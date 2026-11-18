import os
import copy
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from model import CouncilModel
from dataset import StreamingRLDataset, get_tokenizer
from config import get_config
from log import TrainingLogger, sample_batch_output


def find_latest_rl_checkpoint(ckpt_dir):
    """Tìm checkpoint RL có step cao nhất."""
    import glob
    ckpts = glob.glob(os.path.join(ckpt_dir, "rl_step_*.pt"))
    if not ckpts:
        return None, 0
    def get_step(path):
        try:
            return int(os.path.basename(path).split("_")[-1].split(".")[0])
        except:
            return 0
    ckpts.sort(key=get_step, reverse=True)
    return ckpts[0], get_step(ckpts[0])

def compute_reward(prediction, ground_truth):
    """
    Binary reward: 1 nếu đúng, 0 nếu sai.
    Dùng math_verify nếu có, fallback về string match.
    """
    try:
        from math_verify import parse, verify
        gold = parse(ground_truth)
        pred = parse(prediction)
        if gold and pred and verify(gold, pred):
            return 1.0
    except Exception:
        pass
    # Fallback: normalized string match
    p = prediction.strip().lower().replace(" ", "")
    g = ground_truth.strip().lower().replace(" ", "")
    return 1.0 if p == g or g in p else 0.0

def compute_self_logprobs(model, prompt_ids, gen_ids):
    logits = model(gen_ids, num_recursion=model.config.rl_generate_recursion)
    gen_start = prompt_ids.size(1)
    gen_logits = logits[:, gen_start-1:gen_start-1 + (gen_ids.size(1) - gen_start), :]
    logprobs = F.log_softmax(gen_logits, dim=-1)
    targets = gen_ids[:, gen_start:]
    return logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

def compute_oracle_logprobs(model, prompt_ids, gt_texts, gen_ids, tokenizer):
    B = prompt_ids.size(0)
    T_p = prompt_ids.size(1)
    T_gen = gen_ids.size(1) - T_p
    device = prompt_ids.device
    results = []
    for i in range(B):
        gt_enc = tokenizer(gt_texts[i], truncation=True, max_length=256, return_tensors="pt")
        gt_ids_i = gt_enc["input_ids"].to(device)
        T_g = gt_ids_i.size(1)
        full = torch.cat([prompt_ids[i:i+1], gt_ids_i, gen_ids[i:i+1, T_p:]], dim=1)
        logits = model(full, num_recursion=model.config.rl_generate_recursion)
        gen_start = T_p + T_g
        gen_logits = logits[:, gen_start-1:gen_start-1+T_gen, :]
        logprobs = F.log_softmax(gen_logits, dim=-1)
        targets = gen_ids[i:i+1, T_p:]
        lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        results.append(lp)
    return torch.cat(results, dim=0)

def compute_oppo_advantages(self_logprobs, oracle_logprobs, rewards, beta0=0.5, scale=4.0):
    """
    OPPO Bayesian Value Recursion.
    
    self_logprobs: (B, T) — log π(ŷ_t | x, ŷ_<t)
    oracle_logprobs: (B, T) — log π(ŷ_t | x, y*, ŷ_<t)
    rewards: (B,) — binary reward
    
    Returns: advantages (B, T)
    """
    B, T = self_logprobs.shape
    device = self_logprobs.device

    # Per-token likelihood ratio (oracle vs self)
    log_ratio = oracle_logprobs - self_logprobs  # (B, T)
    log_ratio = log_ratio.clamp(-10.0, 10.0)  # ổn định số

    # Bayesian recursion
    p = torch.full((B,), beta0, device=device)  # prior success probability
    advantages = torch.zeros(B, T, device=device)

    for t in range(T):
        lr = log_ratio[:, t]
        # Bayesian update: logit(p_t) = logit(p_{t-1}) + lr
        logit_p = torch.log(p / (1 - p + 1e-8))
        p = torch.sigmoid(logit_p + lr)

        # Advantage weight: uncertainty-weighted (peaks at p=0.5)
        # "concentrates credit where outcome remains uncertain"
        w = 4.0 * p * (1.0 - p)  # in [0, 1]
        advantages[:, t] = w * lr

    reward_signal = 2.0 * rewards - 1.0
    advantages = advantages * reward_signal.unsqueeze(1) * scale

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def train_rl():
    config = get_config()
    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    tokenizer = get_tokenizer()
    # SỬA LỖI 1
    config.vocab_size = len(tokenizer)
    config.eos_token_id = tokenizer.eos_token_id
    print(f"Tokenizer vocab_size = {config.vocab_size}")

    logger = TrainingLogger(config.log_dir, "rl_oppo")
    logger.info(f"Config: {config}")

    dataset = StreamingRLDataset(tokenizer, max_prompt_len=256, max_tokens=config.rl_max_tokens)
    dataloader = DataLoader(
        dataset,
        batch_size=config.rl_batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        multiprocessing_context="spawn"
    )

    model = CouncilModel(config).to(device)

    sft_path = f"{config.checkpoint_dir}/sft_final.pt"
    if os.path.exists(sft_path):
        sft_state = torch.load(sft_path, map_location="cpu", weights_only=False)
        sft_best_N = 0
        if isinstance(sft_state, dict) and "model" in sft_state:
            sft_best_N = sft_state.get("best_N", 0)
            sft_state = sft_state["model"]
        sft_state = {k.replace("_orig_mod.", ""): v for k, v in sft_state.items()}
        model.load_state_dict(sft_state, strict=False)
        if sft_best_N > 0:
            config.best_N = sft_best_N
            print(f"Restored best_N={sft_best_N} from SFT checkpoint")
        print(f"Loaded SFT checkpoint from {sft_path}")
        del sft_state
    else:
        print(f"WARNING: No SFT checkpoint at {sft_path}. RL will start from random init.")

    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print("Created ref_model (frozen SFT copy)")

    if os.environ.get("USE_COMPILE", "1") == "1":
        try:
            model = torch.compile(model, mode="default", fullgraph=False)
            print("Compile OK")
        except Exception as e:
            print(f"Compile failed: {e}")

    resume_ckpt, resume_step = find_latest_rl_checkpoint(config.checkpoint_dir)
    resume_state = None
    if resume_ckpt is not None:
        print(f"Found RL checkpoint: {resume_ckpt} (step {resume_step})")
        resume_state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        if "model" in resume_state:
            resume_state["model"] = {
                k.replace("_orig_mod.", ""): v for k, v in resume_state["model"].items()
            }

    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=config.rl_lr, betas=(0.9, 0.95))
    total_steps = config.rl_total_steps
    save_interval = max(1, int(total_steps * config.save_every_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0)))
    )

    model.train()
    optimizer.zero_grad()
    step = 0
    rec_window = []

    if resume_state is not None:
        try:
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            base_model.load_state_dict(resume_state["model"], strict=False)
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            step = resume_state["step"]
            print(f"Resumed RL from step {step}")
        except Exception as e:
            print(f"Resume failed: {e}. Starting RL from scratch (step 0).")
            step = 0

    for epoch in range(config.rl_epochs):
        pbar = tqdm(dataloader, desc=f"RL (OPPO) Epoch {epoch}")
        for batch in pbar:
            prompt_ids = batch["prompt_ids"].to(device)       # (B, T_p)
            ground_truths = batch["ground_truth"]              # list[str]
            B = prompt_ids.size(0)

            model.eval()
            prompt_mask = batch["prompt_mask"].to(device)
            gen_ids = model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=config.rl_max_new_tokens,
                temperature=config.rl_temperature,
                eos_token_id=tokenizer.eos_token_id
            )
            gen_texts = tokenizer.batch_decode(gen_ids[:, prompt_ids.size(1):], skip_special_tokens=True)
            model.train()

            rewards = torch.tensor(
                [compute_reward(g, gt) for g, gt in zip(gen_texts, ground_truths)],
                device=device, dtype=torch.float32
            )

            self_logprobs = compute_self_logprobs(model, prompt_ids, gen_ids)
            base_for_rec = model._orig_mod if hasattr(model, "_orig_mod") else model
            rec_used = getattr(base_for_rec, "_last_n_rec_used", config.rl_generate_recursion)
            rec_window.append(rec_used)

            oracle_logprobs = compute_oracle_logprobs(model, prompt_ids, list(ground_truths), gen_ids, tokenizer)

            T = min(self_logprobs.size(1), oracle_logprobs.size(1))
            self_lp = self_logprobs[:, :T]
            oracle_lp = oracle_logprobs[:, :T]

            advantages = compute_oppo_advantages(
                self_lp, oracle_lp, rewards,
                config.oppo_beta0_prior, config.oppo_advantage_scale
            )

            with torch.no_grad():
                ref_logprobs = compute_self_logprobs(ref_model, prompt_ids, gen_ids)[:, :T]

            ratio = torch.exp(self_lp - ref_logprobs)
            clipped = torch.clamp(ratio, 1 - config.rl_clip_epsilon, 1 + config.rl_clip_epsilon)
            loss_pg = -torch.min(ratio * advantages, clipped * advantages).mean()

            # 10. KL penalty
            kl = (ref_logprobs - self_lp).mean()
            loss = (loss_pg + config.rl_beta * kl) / config.rl_grad_accum

            loss.backward()

            if (step + 1) % config.rl_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            cur_loss = loss.item() * config.rl_grad_accum
            cur_kl = kl.item()
            cur_reward = rewards.mean().item()

            if step % 10 == 0:
                avg_rec = sum(rec_window) / len(rec_window)
                pbar.set_postfix({
                    "loss": f"{cur_loss:.4f}",
                    "kl": f"{cur_kl:.4f}",
                    "reward": f"{cur_reward:.2f}",
                    "avg_rec": f"{avg_rec:.2f}",
                })
                logger.log_step(step, {
                    "loss": cur_loss,
                    "kl": cur_kl,
                    "reward": cur_reward,
                    "avg_rec": avg_rec,
                })
                rec_window.clear()
            else:
                pbar.set_postfix({
                    "loss": f"{cur_loss:.4f}",
                    "kl": f"{cur_kl:.4f}",
                    "reward": f"{cur_reward:.2f}",
                })

            if step % config.sample_every_n_steps == 0:
                try:
                    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                    was_training = base_model.training
                    base_model.eval()
                    idx = torch.randint(0, B, (1,)).item()
                    prompt_i = prompt_ids[idx:idx+1]
                    prompt_mask_i = prompt_mask[idx:idx+1]
                    gen_fn = base_model.generate
                    if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
                        gen_fn = torch.compiler.disable(gen_fn)
                    with torch.no_grad():
                        out_i = gen_fn(
                            prompt_i,
                            attention_mask=prompt_mask_i,
                            max_new_tokens=64,
                            temperature=config.rl_temperature,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    prompt_text = tokenizer.decode(prompt_i[0], skip_special_tokens=True)
                    completion_text = tokenizer.decode(
                        out_i[0][prompt_i.size(1):], skip_special_tokens=True
                    )
                    logger.log_sample(step, prompt_text, completion_text, ground_truths[idx])
                    if was_training:
                        base_model.train()
                except Exception as e:
                    logger.info(f"Sampling failed: {e}")

            if step % save_interval == 0 and step > 0:
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({
                    "model": save_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "config": config,
                }, f"{config.checkpoint_dir}/rl_step_{step}.pt")
                pct = step / total_steps * 100
                logger.info(f"Saved RL checkpoint at step {step} ({pct:.1f}%)")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    final_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(final_model.state_dict(), f"{config.checkpoint_dir}/rl_final.pt")
    logger.info("RL (OPPO) completed.")
    return model

if __name__ == "__main__":
    train_rl()