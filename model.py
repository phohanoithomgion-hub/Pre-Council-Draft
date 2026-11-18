import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba2

class CustomAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        T_q = q.size(2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v)
        T_kv = k.size(2)

        if past_kv is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T_q == 1:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            offset = T_kv - T_q
            mask = torch.ones(T_q, T_kv, device=x.device, dtype=torch.bool).tril(diagonal=offset)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(B, T_q, D)
        return self.out_proj(out), new_kv

class CouncilBackbone(nn.Module):
    def __init__(self, d_model=640, num_mamba=14, num_attn=1, vocab_size=50257, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mamba_layers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=64, expand=2) for _ in range(num_mamba)
        ])
        self.attn_layers = nn.ModuleList([
            CustomAttention(d_model, num_heads=8) for _ in range(num_attn)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _init_mamba_state(self, mamba, batch_size, device, dtype):
        conv_dim = getattr(mamba, 'conv_dim', None)
        if conv_dim is None:
            conv_dim = mamba.conv1d.weight.shape[0]
        d_conv = getattr(mamba, 'd_conv', mamba.conv1d.weight.shape[-1])
        nheads = getattr(mamba, 'nheads', None) or getattr(mamba, 'num_heads', None)
        headdim = getattr(mamba, 'headdim', None) or getattr(mamba, 'head_dim', None)
        d_state = getattr(mamba, 'd_state', 64)

        conv_state = torch.zeros(
            batch_size, conv_dim, d_conv,
            device=device, dtype=dtype
        )
        ssm_state = torch.zeros(
            batch_size, nheads, headdim, d_state,
            device=device, dtype=dtype
        )
        return (conv_state, ssm_state)

    def _mamba_block(self, mamba, h):
        return mamba(h)

    def _attn_block(self, attn, h):
        out, _ = attn(h, past_kv=None)
        return out

    def forward(self, input_ids, cache=None, return_cache=False):
        B, T = input_ids.shape
        device = input_ids.device

        is_decode = cache is not None
        use_ckpt = self.use_checkpoint and self.training and not is_decode

        if is_decode:
            input_ids_last = input_ids[:, -1:]
            x = self.embedding(input_ids_last)
            mamba_states = list(cache["mamba_states"])
            attn_kvs = list(cache["attn_kvs"])
        else:
            x = self.embedding(input_ids)
            mamba_states = [None] * len(self.mamba_layers)
            attn_kvs = [None] * len(self.attn_layers)

        for i, mamba in enumerate(self.mamba_layers):
            if is_decode:
                conv_state, ssm_state = mamba_states[i]
                conv_state = conv_state.to(x.dtype)
                ssm_state = ssm_state.to(x.dtype)
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                out, new_conv, new_ssm = mamba.step(x, conv_state, ssm_state)
                if out.dim() == 2:
                    out = out.unsqueeze(1)
                x = out
                mamba_states[i] = (new_conv, new_ssm)
            elif return_cache and not self.training:
                conv_state, ssm_state = self._init_mamba_state(mamba, B, device, x.dtype)
                outputs = []
                for t in range(T):
                    xt = x[:, t:t+1, :]
                    xt_out, conv_state, ssm_state = mamba.step(xt, conv_state, ssm_state)
                    if xt_out.dim() == 2:
                        xt_out = xt_out.unsqueeze(1)
                    outputs.append(xt_out)
                x = torch.cat(outputs, dim=1)
                mamba_states[i] = (conv_state, ssm_state)
            elif use_ckpt:
                x = checkpoint(self._mamba_block, mamba, x, use_reentrant=False)
            else:
                x = mamba(x)
            if i == 6 and len(self.attn_layers) > 0:
                need_attn_cache = is_decode or return_cache
                if need_attn_cache:
                    x, new_kv = self.attn_layers[0](x, past_kv=attn_kvs[0])
                    attn_kvs[0] = new_kv
                elif use_ckpt:
                    x = checkpoint(self._attn_block, self.attn_layers[0], x, use_reentrant=False)
                else:
                    x, _ = self.attn_layers[0](x, past_kv=None)

        x = self.norm(x)
        new_cache = {"mamba_states": mamba_states, "attn_kvs": attn_kvs}
        return x, new_cache

class Engram(nn.Module):
    def __init__(self, d_model=640, num_heads=4, table_size=50000, mem_dim=64, max_ngram=3):
        super().__init__()
        self.num_heads = num_heads
        self.mem_dim = mem_dim
        self.max_ngram = max_ngram
        self.tables = nn.ModuleDict()
        for n in range(2, max_ngram + 1):
            for h in range(num_heads):
                prime_size = self._next_prime(table_size + h * 997 + n * 7919)
                self.tables[f"n{n}_h{h}"] = nn.Embedding(prime_size, mem_dim)
        self.mem_proj = nn.Linear(mem_dim, d_model)
        self.gate = nn.Linear(d_model + d_model, 1)

    def _next_prime(self, n):
        def is_prime(x):
            if x < 2: return False
            for i in range(2, int(x**0.5) + 1):
                if x % i == 0: return False
            return True
        while not is_prime(n): n += 1
        return n

    def _poly_hash(self, ngram_ids, table_size):
        base = 31
        h = torch.zeros_like(ngram_ids[..., 0])
        for i in range(ngram_ids.size(-1)):
            h = (h * base + ngram_ids[..., i]) % table_size
        return h

    def forward(self, x, input_ids, T_out=None):
        B, T_in, D = x.shape
        T_out = T_out if T_out is not None else T_in
        mem = torch.zeros(B, T_out, self.mem_dim, device=x.device)
        cnt = torch.zeros(B, T_out, 1, device=x.device)
        T_ng = input_ids.shape[1]
        is_decode = (T_in == 1)
        for n in range(2, self.max_ngram + 1):
            if T_ng < n:
                continue
            if is_decode:
                ngrams = input_ids[:, -n:].unsqueeze(1)
            else:
                ngrams = torch.stack([input_ids[:, i:T_ng - n + 1 + i] for i in range(n)], dim=-1)
            for h in range(self.num_heads):
                table = self.tables[f"n{n}_h{h}"]
                hv = self._poly_hash(ngrams, table.num_embeddings)
                if is_decode:
                    mem[:, -1:, :] += table(hv)
                    cnt[:, -1:, :] += 1
                else:
                    mem[:, n-1:, :] += table(hv)
                    cnt[:, n-1:, :] += 1
        mem = mem / cnt.clamp(min=1)
        mem = self.mem_proj(mem)
        gate = torch.sigmoid(self.gate(torch.cat([x, mem], dim=-1)))
        return x + gate * mem

class HybridMoELayer(nn.Module):
    def __init__(self, d_model=1024, num_t1=3, t2_rank=32, use_checkpoint=False):
        super().__init__()
        self.t2_rank = t2_rank
        self.use_checkpoint = use_checkpoint
        self.t1_experts = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_t1)])
        self.router_t1 = nn.Linear(d_model, num_t1)
        self.hyper_a = nn.Linear(d_model, d_model * t2_rank)
        self.hyper_b = nn.Linear(d_model, t2_rank * d_model)
        self.shared_expert = nn.Linear(d_model, d_model, bias=False)
        self.G = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        self.G_alpha = 0.9
        self.norm = nn.LayerNorm(d_model)

    def _forward(self, h):
        B, T, D = h.shape
        p_t1 = torch.softmax(self.router_t1(h), dim=-1)
        out_t1 = sum(p_t1[:, :, i:i+1] * expert(h) for i, expert in enumerate(self.t1_experts))

        A = self.hyper_a(h).view(B, T, D, self.t2_rank)
        B_mat = self.hyper_b(h).view(B, T, self.t2_rank, D)
        inter = torch.einsum('btd,btdr->btr', h, A)
        inter = F.gelu(inter)
        out_t2 = torch.einsum('btr,btrd->btd', inter, B_mat)

        cur = out_t1 + out_t2 + self.shared_expert(h)
        cur_mean = cur.mean(dim=(0, 1))
        out = self.norm(h + cur + self.G.unsqueeze(0).unsqueeze(0))
        return out, cur_mean

    def forward(self, h):
        if self.use_checkpoint and self.training:
            out, cur_mean = checkpoint(self._forward, h, use_reentrant=False)
        else:
            out, cur_mean = self._forward(h)
        if self.training:
            with torch.no_grad():
                self.G.data = self.G_alpha * self.G.data + (1 - self.G_alpha) * cur_mean.detach()
        return out

class LiquidCfCLayer(nn.Module):
    def __init__(self, d_model=1024, hidden_dim=16000, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.backbone = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model * 3)
        )
        self.w_tau = nn.Parameter(torch.full((d_model,), 1.0))
        self.norm = nn.LayerNorm(d_model)

    def _forward(self, x, t):
        B, T, D = x.shape
        fgh = self.backbone(x).view(B, T, 3, D)
        f, g, h = fgh[:, :, 0], fgh[:, :, 1], fgh[:, :, 2]
        decay = F.softplus(self.w_tau) * torch.sigmoid(f)
        gate = torch.exp(-decay * t)
        return self.norm(x + gate * g + (1 - gate) * h)

    def forward(self, x, t=1.0):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, t, use_reentrant=False)
        return self._forward(x, t)

class MultiDimLayer(nn.Module):
    def __init__(self, d_model=1024, base_dim=2048, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.base_dim = base_dim
        self.curvature = 1.0
        self.eps = 1e-7
        self.euclid = nn.Sequential(nn.Linear(d_model, 1024), nn.GELU(), nn.Linear(1024, base_dim))
        self.hyper = nn.Sequential(nn.Linear(d_model, 1024), nn.GELU(), nn.Linear(1024, base_dim))
        self.sph = nn.Sequential(nn.Linear(d_model, 1024), nn.GELU(), nn.Linear(1024, base_dim))
        self.hyper_shift = nn.Parameter(torch.zeros(base_dim))
        self.gate = nn.Sequential(nn.Linear(base_dim * 3, 1024), nn.GELU(), nn.Linear(1024, 3), nn.Softmax(dim=-1))
        self.out_proj = nn.Linear(base_dim * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _exp_map(self, v):
        n = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return torch.tanh(n) * v / n

    def _mobius_add(self, x, y):
        c = self.curvature
        x2 = (x * x).sum(-1, keepdim=True)
        y2 = (y * y).sum(-1, keepdim=True)
        xy = (x * y).sum(-1, keepdim=True)
        num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
        den = 1 + 2*c*xy + c*c*x2*y2
        return num / den.clamp(min=self.eps)

    def _forward(self, h):
        e = self.euclid(h)
        hyp_v = self._exp_map(self.hyper(h))
        hyp = self._mobius_add(hyp_v, self.hyper_shift.unsqueeze(0).unsqueeze(0))
        sph = F.normalize(self.sph(h), dim=-1)
        comb = torch.cat([e, hyp, sph], dim=-1)
        gw = self.gate(comb)
        gated = torch.cat([
            e * gw[:, :, 0:1], hyp * gw[:, :, 1:2], sph * gw[:, :, 2:3]
        ], dim=-1)
        return self.norm(h + self.out_proj(gated))

    def forward(self, h):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, h, use_reentrant=False)
        return self._forward(h)

class QuantumLayer(nn.Module):
    def __init__(self, d_model=1024, num_qubits=8):
        super().__init__()
        self.dim_hilbert = 2 ** num_qubits
        self.amp_encoder = nn.Sequential(
            nn.Linear(d_model, 4096), nn.GELU(), nn.Linear(4096, self.dim_hilbert * 2)
        )
        self.hamiltonian = nn.Parameter(torch.randn(self.dim_hilbert, self.dim_hilbert) * 0.05)
        self.measurement = nn.Linear(self.dim_hilbert, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            xf = x.float()
            B, T, D = xf.shape
            amp = self.amp_encoder(xf).view(B, T, 2, self.dim_hilbert)
            real, imag = amp[:, :, 0], amp[:, :, 1]
            n = torch.sqrt(real**2 + imag**2).sum(-1, keepdim=True)
            psi = torch.complex(real / n, imag / n)
            H = (self.hamiltonian + self.hamiltonian.t()) / 2
            U = torch.matrix_exp(1j * H.float())
            psi = torch.einsum('ij,btj->bti', U, psi)
            probs = psi.abs().pow(2)
            probs = probs / probs.sum(-1, keepdim=True)
            out = self.norm(xf + self.measurement(probs.to(xf.dtype)))
        return out.to(x.dtype)

class CouncilDecisionHead(nn.Module):
    def __init__(self, d_model=1024):
        super().__init__()
        self.refine = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.confidence = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())
        self.decision = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, h):
        hr = self.refine(h)
        return hr, self.confidence(hr), self.decision(hr)

class CouncilModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = CouncilBackbone(
            config.d_model_backbone, config.num_mamba, config.num_attn, config.vocab_size,
            use_checkpoint=getattr(config, "backbone_checkpoint", False),
        )
        self.engram = Engram(
            config.d_model_backbone, table_size=config.engram_table_size, mem_dim=config.engram_mem_dim
        )
        self.proj_in = nn.Linear(config.d_model_backbone, config.d_model_thought)
        self.proj_final = nn.Linear(config.d_model_thought, config.d_model_backbone)
        self.final_norm = nn.LayerNorm(config.d_model_backbone)
        self.lm_head = nn.Linear(config.d_model_backbone, config.vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embedding.weight

        self.logic = HybridMoELayer(config.d_model_thought, use_checkpoint=False)
        self.dynamic = LiquidCfCLayer(config.d_model_thought, config.dynamic_hidden, use_checkpoint=False)
        self.multidim = MultiDimLayer(config.d_model_thought, config.multidim_base, use_checkpoint=False)
        self.quantum = QuantumLayer(config.d_model_thought, config.num_qubits)
        self.decision = CouncilDecisionHead(config.d_model_thought)

        # SỬA LỖI 6: Residual scaling qua 12 vòng
        self.residual_norm = nn.LayerNorm(config.d_model_thought)

        # Khởi tạo lại tất cả weights với scale đúng
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.backbone.embedding.weight, mean=0, std=0.02)
    def forward(self, input_ids, labels=None, num_recursion=None, cache=None, return_cache=False, force_full_recursion=False):
        self._last_n_rec_used = 0
        self._last_would_stop_at = 0

        x, new_cache = self.backbone(input_ids, cache=cache, return_cache=return_cache)

        if self.training and cache is None and getattr(self.config, "backbone_checkpoint", False):
            def pre_block(h, ids):
                h = self.engram(h, ids)
                h = self.proj_in(h)
                return h
            x = checkpoint(pre_block, x, input_ids, use_reentrant=False)
        else:
            if cache is None:
                x = self.engram(x, input_ids)
            else:
                if "last_tokens" in cache:
                    tokens_for_engram = torch.cat([cache["last_tokens"], input_ids[:, -1:]], dim=1)
                    x = self.engram(x, tokens_for_engram, T_out=x.shape[1])
            x = self.proj_in(x)

        n_rec = num_recursion if num_recursion is not None else self.config.max_recursion
        prev_x = None
        cont_list = []
        delta_list = []
        would_stop_at = None
        last_step_idx = 0

        for step_idx in range(n_rec):
            last_step_idx = step_idx
            x_prev = x.clone()

            if self.training:
                def thought_block(h):
                    h = self.logic(h)
                    h = self.dynamic(h, t=1.0)
                    h = self.multidim(h)
                    h = self.quantum(h)
                    return h
                x = checkpoint(thought_block, x, use_reentrant=False)
            else:
                x = self.logic(x)
                x = self.dynamic(x, t=1.0)
                x = self.multidim(x)
                x = self.quantum(x)

            x = self.residual_norm(x_prev + x)

            if prev_x is not None and prev_x.shape == x.shape:
                delta_i = (x - prev_x).norm(dim=-1)
            else:
                delta_i = None
            delta_list.append(delta_i)

            h_refined, _, cont_i = self.decision(x)
            cont_list.append(cont_i.squeeze(-1))

            cont_mean = cont_i.mean().item()
            delta_mean = delta_i.mean().item() if delta_i is not None else 1.0

            if would_stop_at is None and (cont_mean < 0.5 or delta_mean < 0.01):
                would_stop_at = step_idx + 1

            if not force_full_recursion:
                if would_stop_at == step_idx + 1:
                    x = h_refined
                    break

            prev_x = x.clone()
            x = h_refined

        self._last_n_rec_used = last_step_idx + 1
        self._last_would_stop_at = would_stop_at if would_stop_at is not None else n_rec

        x_out = self.final_norm(self.proj_final(x))
        logits = self.lm_head(x_out)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_lm = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            if self.training and len(cont_list) >= 2:
                aux_losses = []
                for i in range(len(cont_list) - 1):
                    delta_next = delta_list[i + 1]
                    if delta_next is None:
                        continue
                    abs_ok = delta_next > 0.01
                    delta_curr = delta_list[i]
                    if delta_curr is not None:
                        rel = delta_next / (delta_curr + 1e-6)
                        target = (abs_ok & (rel > 0.5)).float()
                    else:
                        target = abs_ok.float()
                    with torch.amp.autocast('cuda', enabled=False):
                        aux_losses.append(
                            F.binary_cross_entropy(cont_list[i].float(), target)
                        )
                if aux_losses:
                    loss_aux = torch.stack(aux_losses).mean().to(loss_lm.dtype)
                    loss = loss_lm + self.config.aux_cont_loss_weight * loss_aux
                else:
                    loss = loss_lm
            else:
                loss = loss_lm
            if return_cache:
                return logits, loss, new_cache
            return logits, loss

        if return_cache:
            return logits, new_cache
        return logits

    @torch.no_grad()
    def probe_recursion(self, input_ids, labels, max_rec=None, delta_loss_threshold=0.05):
        max_rec = max_rec or self.config.max_recursion
        was_training = self.training
        self.eval()
        try:
            x, _ = self.backbone(input_ids)
            x = self.engram(x, input_ids)
            x = self.proj_in(x)

            B = input_ids.size(0)
            prev_x = None
            loss_per_rec = []
            delta_per_rec = []
            per_sample_losses = []

            for step_idx in range(max_rec):
                x_prev = x.clone()
                x = self.logic(x)
                x = self.dynamic(x, t=1.0)
                x = self.multidim(x)
                x = self.quantum(x)
                x = self.residual_norm(x_prev + x)

                if prev_x is not None:
                    delta_i = (x - prev_x).norm(dim=-1).mean(dim=-1)
                    delta_per_rec.append(delta_i.detach().cpu())
                else:
                    delta_per_rec.append(None)

                h_refined, _, _ = self.decision(x)
                prev_x = x.clone()
                x = h_refined

                x_out = self.final_norm(self.proj_final(x))
                logits = self.lm_head(x_out)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_per_sample = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction='none'
                ).view(B, -1).mean(dim=-1)
                per_sample_losses.append(loss_per_sample.detach().cpu())
                loss_per_rec.append(loss_per_sample.mean().detach().cpu())

            per_sample_losses_t = torch.stack(per_sample_losses, dim=1)
            per_sample_optimal = torch.full((B,), max_rec, dtype=torch.long)
            for b in range(B):
                for k in range(max_rec - 1):
                    if (per_sample_losses_t[b, k] - per_sample_losses_t[b, k + 1]) < delta_loss_threshold:
                        per_sample_optimal[b] = k + 1
                        break

            optimal_N = int(per_sample_optimal.median().item())
            return {
                'loss_per_rec': torch.stack(loss_per_rec),
                'delta_per_rec': delta_per_rec,
                'optimal_N': optimal_N,
                'per_sample_optimal_N': per_sample_optimal,
                'per_sample_losses': per_sample_losses_t,
            }
        finally:
            if was_training:
                self.train()

    @staticmethod
    def _apply_repetition_penalty(logits, generated_ids, penalty):
        if penalty is None or penalty <= 1.0:
            return logits
        score = torch.gather(logits, 1, generated_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        return logits.scatter(1, generated_ids, score)

    @staticmethod
    def _apply_no_repeat_ngram(logits, generated_ids, n):
        if n is None or n <= 0:
            return logits
        B = generated_ids.size(0)
        logits = logits.clone()
        for b in range(B):
            seq = generated_ids[b].tolist()
            if len(seq) < n:
                continue
            if n == 1:
                prefix = ()
            else:
                prefix = tuple(seq[-(n - 1):])
            banned = set()
            for i in range(len(seq) - n + 1):
                ng = tuple(seq[i:i + n])
                if ng[:n - 1] == prefix:
                    banned.add(ng[-1])
            if banned:
                idx = torch.tensor(sorted(banned), device=logits.device, dtype=torch.long)
                logits[b, idx] = float("-inf")
        return logits

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=256, temperature=0.8, eos_token_id=None, repetition_penalty=1.0, no_repeat_ngram_size=0):
        self.eval()
        B = input_ids.size(0)
        device = input_ids.device
        k = self.engram.max_ngram - 1
        cfg_best_N = getattr(self.config, "best_N", 0)
        n_rec = cfg_best_N if cfg_best_N > 0 else self.config.rl_generate_recursion

        logits, cache = self.forward(
            input_ids,
            num_recursion=n_rec,
            return_cache=True
        )

        T_p = input_ids.size(1)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1)
            last_tokens = torch.stack(
                [input_ids[
                    torch.arange(B, device=device),
                    (lengths - 1 - j).clamp(min=0, max=T_p - 1),
                ] for j in range(k - 1, -1, -1)],
                dim=1
            )
        else:
            if T_p >= k:
                last_tokens = input_ids[:, -k:]
            else:
                last_tokens = input_ids[:, :1].expand(-1, k)

        out = input_ids
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        nxt_logits = logits[:, -1, :]
        nxt_logits = self._apply_repetition_penalty(nxt_logits, out, repetition_penalty)
        nxt_logits = self._apply_no_repeat_ngram(nxt_logits, out, no_repeat_ngram_size)
        nxt_logits = nxt_logits / max(temperature, 1e-6)
        probs = F.softmax(nxt_logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        if eos_token_id is not None:
            nxt = torch.where(finished.unsqueeze(-1), torch.full_like(nxt, eos_token_id), nxt)
        out = torch.cat([out, nxt], dim=-1)

        if eos_token_id is not None:
            finished = finished | (nxt.squeeze(-1) == eos_token_id)
            if finished.all().item():
                return out

        for _ in range(max_new_tokens - 1):
            cache["last_tokens"] = last_tokens
            logits, cache = self.forward(
                nxt,
                num_recursion=n_rec,
                cache=cache,
                return_cache=True
            )

            last_tokens = torch.cat([last_tokens, nxt], dim=1)[:, -k:]

            nxt_logits = logits[:, -1, :]
            nxt_logits = self._apply_repetition_penalty(nxt_logits, out, repetition_penalty)
            nxt_logits = self._apply_no_repeat_ngram(nxt_logits, out, no_repeat_ngram_size)
            nxt_logits = nxt_logits / max(temperature, 1e-6)
            probs = F.softmax(nxt_logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None:
                nxt = torch.where(finished.unsqueeze(-1), torch.full_like(nxt, eos_token_id), nxt)
            out = torch.cat([out, nxt], dim=-1)

            if eos_token_id is not None:
                finished = finished | (nxt.squeeze(-1) == eos_token_id)
                if finished.all().item():
                    break

        return out