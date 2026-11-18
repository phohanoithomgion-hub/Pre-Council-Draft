import os
import math
from dataclasses import dataclass

@dataclass
class CouncilConfig:
    # Model
    d_model_backbone: int = 640
    num_mamba: int = 14
    num_attn: int = 1
    vocab_size: int = 50257  # Sẽ được ghi đè bằng len(tokenizer)
    d_model_thought: int = 1024
    num_qubits: int = 8
    dynamic_hidden: int = 16000
    multidim_base: int = 2048
    engram_table_size: int = 50000
    engram_mem_dim: int = 64
    max_recursion: int = 12
    max_seq_len: int = 512

    # SFT
    sft_batch_size: int = 14
    sft_grad_accum: int = 8
    sft_lr: float = 1e-5
    sft_warmup: int = 2000
    sft_epochs: int = 1
    sft_max_tokens: int = 50_000_000_000
    sft_total_steps: int = 5000000
    save_every_ratio: float = 0.01

    # SFT Recursion Schedule
    sft_rec_1_frac: float = 0.05
    sft_rec_2_frac: float = 0.15
    sft_rec_max_frac: float = 0.40
    sft_early_stop_frac: float = 0.50

    # Checkpointing
    backbone_checkpoint: bool = True
    attention_checkpoint: bool = True

    # Pretraining
    pretrain_batch_size: int = 19
    pretrain_grad_accum: int = 8
    pretrain_lr: float = 2e-5
    pretrain_lr_warmdown_steps: int = 5
    pretrain_warmup: int = 2000
    pretrain_max_tokens: int = 30_000_000_000
    pretrain_max_steps: int = 3_000_000
    pretrain_save_every_ratio: float = 0.01
    pretrain_dataset_configs: list = None
    pretrain_num_recursion: int = 1

    # RL (OPPO)
    rl_batch_size: int = 4
    rl_grad_accum: int = 8
    rl_lr: float = 1e-5
    rl_epochs: int = 1
    rl_max_tokens: int = 50_000_000_000
    rl_total_steps: int = 90000
    rl_clip_epsilon: float = 0.2
    rl_beta: float = 0.01
    rl_max_new_tokens: int = 128
    sample_every_n_steps: int = 500
    rl_temperature: float = 0.8
    rl_generate_recursion: int = 12
    use_cache: bool = True
    mamba_ssm_cache_dtype: str = "bfloat16"
    mamba_conv_cache_dtype: str = "bfloat16"

    # OPPO
    oppo_beta0_prior: float = 0.5
    oppo_advantage_scale: float = 4.0

    # Adaptive recursion
    aux_cont_loss_weight: float = 0.05
    probe_every_n: int = 500
    probe_start_step: int = 30000
    probe_max_batch: int = 4
    probe_delta_loss_threshold: float = 0.05
    min_best_N: int = 3
    best_N: int = 0

    # Data
    max_samples_per_dataset: int = 2_000_000
    num_workers: int = 0

    # System
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42

    # Inference generation
    inference_repetition_penalty: float = 1.15
    inference_no_repeat_ngram_size: int = 3

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

def get_config():
    config = CouncilConfig()
    if config.pretrain_dataset_configs is None:
        config.pretrain_dataset_configs = [
            "automathtext-v2-ultra",
            "automathtext-v2-high",
        ]
    return config