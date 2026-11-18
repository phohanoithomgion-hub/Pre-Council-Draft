import os
import queue
import threading
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ["HF_HUB_DISABLE_HF_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"

DATA_DIR = os.path.expanduser("~/council_data")

class StreamingSFTDataset(IterableDataset):
    def __init__(self, tokenizer, max_seq_len=512, max_tokens=50_000_000_000):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_tokens = max_tokens
        self.datasets = [
            ("HuggingFaceTB/finemath", "finemath-4plus"),
            ("meituan-longcat/R-HORIZON-training-data", None),
            ("open-thoughts/OpenThoughts-114k", None),
            ("AI-MO/NuminaMath-CoT", None),
            ("nvidia/OpenMathReasoning", None),
            ("open-r1/OpenR1-Math-220k", None),
            ("openai/gsm8k", "main"),
            ("DigitalLearningGmbH/MATH-lighteval", None),
            ("TIGER-Lab/MathInstruct", None),
            ("meta-math/MetaMathQA", None),
        ]
        self.prompt_template = "User: {prompt}\nAssistant: "

    def _worker(self, q):
        try:
            total_tokens = 0
            for repo_id, config in self.datasets:
                try:
                    ds = load_dataset(
                        repo_id, config, split="train", 
                        cache_dir=DATA_DIR, streaming=True
                    )
                except Exception:
                    continue

                for example in ds:
                    if total_tokens >= self.max_tokens:
                        q.put(None)
                        return
                    
                    item = self._extract_qa(example)
                    if item is None:
                        continue
                    
                    prompt_raw, answer_raw = item
                    if len(answer_raw.strip()) < 10:
                        continue
                    
                    prompt_text = self.prompt_template.format(prompt=prompt_raw.strip())
                    answer_text = answer_raw.strip() + self.tokenizer.eos_token
                    
                    prompt_ids_full = self.tokenizer(prompt_text, truncation=False, add_special_tokens=False)["input_ids"]
                    prompt_len = len(prompt_ids_full)

                    if prompt_len >= self.max_seq_len - 10:
                        continue

                    enc_prompt = self.tokenizer(prompt_text, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
                    prompt_ids = enc_prompt["input_ids"].squeeze(0)
                    prompt_mask = enc_prompt["attention_mask"].squeeze(0)

                    full_text = prompt_text + answer_text
                    enc_full = self.tokenizer(full_text, truncation=True, max_length=self.max_seq_len, padding="max_length", return_tensors="pt")
                    input_ids = enc_full["input_ids"].squeeze(0)
                    attention_mask = enc_full["attention_mask"].squeeze(0)
                    
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100
                    labels[:prompt_len] = -100
                    
                    total_tokens += (labels != -100).sum().item()

                    q.put({
                        "input_ids": input_ids,
                        "labels": labels,
                        "prompt_ids": prompt_ids,
                        "prompt_mask": prompt_mask,
                    })
            q.put(None)
        except Exception as e:
            q.put(e)

    def __iter__(self):
        q = queue.Queue(maxsize=32)
        t = threading.Thread(target=self._worker, args=(q,), daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def _extract_qa(self, ex):
        if "conversations" in ex:
            user_msg = next((c["content"] for c in ex["conversations"] if c.get("role") == "user"), None)
            assistant_msg = next((c["content"] for c in ex["conversations"] if c.get("role") == "assistant"), None)
            if user_msg and assistant_msg:
                return user_msg, assistant_msg
        for k1, k2 in [("problem", "solution"), ("question", "answer"), ("instruction", "output"), ("input", "output")]:
            if k1 in ex and k2 in ex:
                return str(ex[k1]), str(ex[k2])
        if "text" in ex:
            text = ex["text"]
            for marker in ["\nAnswer:", "\nSolution:", "\nOutput:"]:
                if marker in text:
                    parts = text.split(marker, 1)
                    return parts[0].strip(), parts[1].strip()
        return None


class StreamingRLDataset(IterableDataset):
    def __init__(self, tokenizer, max_prompt_len=256, max_tokens=10_000_000_000):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_tokens = max_tokens
        self.datasets = [
            ("meituan-longcat/R-HORIZON-training-data", None),
            ("openai/gsm8k", "main"),
            ("AI-MO/NuminaMath-CoT", None),
            ("nvidia/OpenMathReasoning", None),
            ("open-thoughts/OpenThoughts-114k", None),
        ]
        self.prompt_template = "User: {prompt}\nAssistant: "

    def __iter__(self):
        total = 0
        for repo_id, config in self.datasets:
            try:
                ds = load_dataset(
                    repo_id, config, split="train",
                    cache_dir=DATA_DIR, streaming=True,
                )
            except Exception:
                continue

            for ex in ds:
                if total >= self.max_tokens:
                    return
                item = self._extract(ex)
                if item is None:
                    continue
                prompt, gt = item
                prompt_text = self.prompt_template.format(prompt=prompt.strip())
                enc = self.tokenizer(
                    prompt_text, truncation=True, max_length=self.max_prompt_len,
                    padding="max_length", return_tensors="pt"
                )
                prompt_ids = enc["input_ids"].squeeze(0)
                prompt_mask = enc["attention_mask"].squeeze(0)
                total += prompt_ids.numel()
                yield {"prompt_ids": prompt_ids, "prompt_mask": prompt_mask, "ground_truth": gt}

    def _extract(self, ex):
        if "question" in ex and "answer" in ex:
            q = ex["question"]
            a = ex["answer"].split("####")[-1].strip() if "####" in ex["answer"] else ex["answer"]
            return q, a
        if "problem" in ex and "solution" in ex:
            return ex["problem"], ex["solution"][-200:]
        if "instruction" in ex and "output" in ex:
            return ex["instruction"], ex["output"]
        return None


class StreamingPretrainDataset(IterableDataset):
    def __init__(self, tokenizer, max_seq_len=512, config_names=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if config_names is None:
            config_names = ["automathtext-v2-ultra", "automathtext-v2-high"]
        if isinstance(config_names, str):
            config_names = [config_names]
        self.config_names = config_names

    def __iter__(self):
        for config_name in self.config_names:
            print(f"[PretrainDataset] Loading config: {config_name}")
            ds = load_dataset(
                "OpenSQZ/AutoMathText-V2",
                config_name,
                split="train",
                cache_dir=DATA_DIR,
                streaming=True,
            )
            for example in ds:
                text = example.get("text", "")
                if not text or len(text) < 50:
                    continue
                enc = self.tokenizer(
                    text, truncation=True, max_length=self.max_seq_len,
                    padding="max_length", return_tensors="pt",
                )
                input_ids = enc["input_ids"].squeeze(0)
                attention_mask = enc["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "n_tokens": (labels != -100).sum().item(),
                }


def get_tokenizer(model_name="gpt2"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok