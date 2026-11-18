import os
import math
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CouncilModel
from dataset import StreamingPretrainDataset, get_tokenizer
from config import get_config
from log import TrainingLogger


def find_latest_pretrain_checkpoint(ckpt_dir):
    import glob
    ckpts = glob.glob(os.path.join(ckpt_dir, "pretrain_step_*.pt"))
    if not ckpts:
        return None, 0
    def get_step(path):
        try:
            return int(os.path.basename(path).split("_")[-1].split(".")[0])
        except Exception:
            return 0
    ckpts.sort(key=get_step, reverse=True)
    return ckpts[0], get_step(ckpts[0])


def warmup_cosine(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_pretrain():
    config = get_config()
    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    tokenizer = get_tokenizer()
    config.vocab_size = len(tokenizer)
    config.eos_token_id = tokenizer.eos_token_id
    print(f"Tokenizer vocab_size = {config.vocab_size}")

    logger = TrainingLogger(config.log_dir, "pretrain")
    logger.info(f"Config: {config}")

    dataset = StreamingPretrainDataset(
        tokenizer,
        max_seq_len=config.max_seq_len,
        config_names=config.pretrain_dataset_configs,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.pretrain_batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        multiprocessing_context="spawn"
    )

    config.backbone_checkpoint = False
    model = CouncilModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_params:,}")

    resume_ckpt, resume_step = find_latest_pretrain_checkpoint(config.checkpoint_dir)
    resume_state = None
    if resume_ckpt is not None:
        logger.info(f"Found pretrain checkpoint: {resume_ckpt} (step {resume_step})")
        resume_state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        resume_state["model"] = {
            k.replace("_orig_mod.", ""): v
            for k, v in resume_state["model"].items()
        }

    use_compile = os.environ.get("USE_COMPILE", "1") == "1"
    if use_compile:
        try:
            logger.info("Compiling model (mode=default)...")
            model = torch.compile(model, mode="default", fullgraph=False)
            logger.info("Compile OK")
        except Exception as e:
            logger.info(f"Compile failed, fallback eager: {e}")
            use_compile = False

    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(), lr=config.pretrain_lr, betas=(0.9, 0.95)
    )

    total_steps = config.pretrain_max_steps
    save_interval = max(1, int(total_steps * config.pretrain_save_every_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: warmup_cosine(s, config.pretrain_warmup, total_steps),
    )

    model.train()
    optimizer.zero_grad()
    step = 0
    total_real_tokens = 0
    resumed_lr = None
    resumed_step = None

    if resume_state is not None:
        try:
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            missing, unexpected = base_model.load_state_dict(resume_state["model"], strict=False)
            if missing:
                logger.info(f"Resume missing keys: {len(missing)}")
            if unexpected:
                logger.info(f"Resume unexpected keys: {len(unexpected)}")
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            resumed_lr = optimizer.param_groups[0]["lr"]
            for pg in optimizer.param_groups:
                pg["initial_lr"] = config.pretrain_lr
            scheduler.base_lrs = [config.pretrain_lr for _ in optimizer.param_groups]
            step = resume_state["step"]
            resumed_step = step
            total_real_tokens = resume_state.get("total_real_tokens", 0)
            logger.info(f"Resumed from step {step}, tokens={total_real_tokens/1e9:.3f}B")
            logger.info(f"LR warmdown: {resumed_lr:.2e} -> {config.pretrain_lr:.2e} over {config.pretrain_lr_warmdown_steps} batches")
        except Exception as e:
            logger.info(f"Resume failed: {e}. Starting from scratch.")

    pbar = tqdm(dataloader, desc="Pretrain")
    for batch in pbar:
        if step >= total_steps or total_real_tokens >= config.pretrain_max_tokens:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        n_tokens = int(batch["n_tokens"].sum().item())
        total_real_tokens += n_tokens

        with autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(
                input_ids,
                labels=labels,
                num_recursion=config.pretrain_num_recursion,
                force_full_recursion=True,
            )
            loss = loss / config.pretrain_grad_accum

        loss.backward()

        if (step + 1) % config.pretrain_grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            next_step = step + 1
            if resumed_lr is not None and next_step <= resumed_step + config.pretrain_lr_warmdown_steps:
                t = (next_step - resumed_step) / config.pretrain_lr_warmdown_steps
                t = max(0.0, min(1.0, t))
                target_lr = scheduler.get_last_lr()[0]
                lr = (1 - t) * resumed_lr + t * target_lr
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

        step += 1
        current_loss = loss.item() * config.pretrain_grad_accum
        current_lr = optimizer.param_groups[0]["lr"]

        if step % 10 == 0:
            pct = step / total_steps * 100
            pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "tokens": f"{total_real_tokens/1e9:.2f}B",
                "pct": f"{pct:.1f}%",
            })
            logger.log_step(step, {
                "loss": current_loss,
                "lr": current_lr,
                "tokens_B": total_real_tokens / 1e9,
            })

        if step % save_interval == 0 and step > 0:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({
                "model": save_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "total_real_tokens": total_real_tokens,
                "config": config,
            }, f"{config.checkpoint_dir}/pretrain_step_{step}.pt")
            pct = step / total_steps * 100
            logger.info(
                f"Saved pretrain checkpoint at step {step} ({pct:.1f}%), "
                f"tokens={total_real_tokens/1e9:.3f}B"
            )

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    final_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model": final_model.state_dict(),
        "total_real_tokens": total_real_tokens,
        "step": step,
        "config": config,
    }, f"{config.checkpoint_dir}/pretrain_final.pt")
    logger.info(
        f"Pretrain completed. step={step}, tokens={total_real_tokens/1e9:.3f}B"
    )
    return model


if __name__ == "__main__":
    train_pretrain()