import os
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_dataset

from model import CouncilModel
from dataset import get_tokenizer
from config import get_config


class EvalDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        prompt_text = ex["prompt"]
        answer_text = ex["answer"]

        prompt_ids_full = self.tokenizer(prompt_text, truncation=False, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids_full)

        enc = self.tokenizer(
            prompt_text + answer_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if prompt_len < self.max_len:
            labels[:prompt_len] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
        }


def load_council_model(checkpoint_path, config, device):
    model = CouncilModel(config).to(device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    model.eval()
    return model


def build_valid_samples(tokenizer, num_samples=500):
    samples = []
    sources = [
        ("openai/gsm8k", "train", "question", "answer"),
        ("AI-MO/NuminaMath-CoT", "train", "problem", "solution"),
    ]
    per_source = max(1, num_samples // len(sources))
    for ds_name, split, q_key, a_key in sources:
        try:
            ds = load_dataset(ds_name, split=split, streaming=True)
        except Exception:
            continue
        count = 0
        for ex in ds:
            if count >= per_source:
                break
            q = ex.get(q_key)
            a = ex.get(a_key)
            if not q or not a:
                continue
            samples.append({
                "prompt": f"Problem: {q}\nAnswer:",
                "answer": a if ds_name == "openai/gsm8k" else f" {a}",
            })
            count += 1
    return samples


@torch.no_grad()
def evaluate_valid_loss(model, dataloader, device, num_recursion=12):
    total_loss = 0.0
    total_tokens = 0
    for batch in tqdm(dataloader, desc="valid_loss"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(input_ids, labels=labels, num_recursion=num_recursion, force_full_recursion=True)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum"
        )
        n_tokens = (shift_labels != -100).sum().item()
        total_loss += loss_sum.item()
        total_tokens += n_tokens
    avg_loss = total_loss / max(1, total_tokens)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return {"loss": avg_loss, "perplexity": ppl, "n_tokens": total_tokens}


def extract_answer(text):
    for marker in ["Answer:", "answer:", "####", "\\boxed{"]:
        if marker in text:
            part = text.split(marker)[-1].strip()
            return part.split("\n")[0].strip()
    return text.strip().split("\n")[-1].strip()


def compute_reward(pred, gt):
    try:
        from math_verify import parse, verify
        gold = parse(gt)
        p = parse(pred)
        if gold and p and verify(gold, p):
            return 1.0
    except Exception:
        pass
    return 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0


@torch.no_grad()
def evaluate_benchmark(model, tokenizer, device, benchmarks, max_new_tokens=256, temperature=0.0, best_n=None):
    results = {}
    for name, loader_fn in benchmarks.items():
        try:
            ds = loader_fn()
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue

        correct = 0
        total = 0
        for ex in tqdm(ds, desc=f"bench:{name}"):
            prompt = ex["prompt"]
            gt = ex["answer"]
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding=False)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)

            gen_fn = model.generate
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
                gen_fn = torch.compiler.disable(gen_fn)

            old_best = getattr(model.config, "best_N", 0)
            if best_n is not None:
                model.config.best_N = best_n

            out = gen_fn(
                input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.0,
                eos_token_id=tokenizer.eos_token_id,
            )
            model.config.best_N = old_best

            completion = tokenizer.decode(out[0][input_ids.size(1):], skip_special_tokens=True)
            pred = extract_answer(completion)
            correct += compute_reward(pred, gt)
            total += 1

        acc = correct / max(1, total)
        results[name] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"[{name}] acc={acc:.4f} ({correct}/{total})")
    return results


def build_benchmarks():
    def gsm8k_loader():
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [
            {
                "prompt": f"Problem: {ex['question']}\nAnswer:",
                "answer": ex["answer"].split("####")[-1].strip(),
            }
            for ex in ds
        ]

    def math500_loader():
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            {
                "prompt": f"Problem: {ex['problem']}\nAnswer:",
                "answer": ex["answer"],
            }
            for ex in ds
        ]

    return {"gsm8k": gsm8k_loader, "math500": math500_loader}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["valid", "benchmark", "all"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_valid", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--best_n", type=int, default=None)
    parser.add_argument("--num_recursion", type=int, default=12)
    parser.add_argument("--out", type=str, default="eval_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    tokenizer = get_tokenizer()
    config.vocab_size = len(tokenizer)

    model = load_council_model(args.ckpt, config, device)

    report = {}

    if args.mode in ("valid", "all"):
        samples = build_valid_samples(tokenizer, args.num_valid)
        ds = EvalDataset(samples, tokenizer, config.max_seq_len)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        report["valid"] = evaluate_valid_loss(model, dl, device, num_recursion=args.num_recursion)
        print(f"[valid] loss={report['valid']['loss']:.4f} ppl={report['valid']['perplexity']:.2f}")

    if args.mode in ("benchmark", "all"):
        bms = build_benchmarks()
        report["benchmark"] = evaluate_benchmark(
            model, tokenizer, device, bms,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            best_n=args.best_n,
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()