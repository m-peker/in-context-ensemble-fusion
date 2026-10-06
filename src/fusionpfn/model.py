"""FusionPFN network: two-axis attention over a (rows x models) grid of cell tokens.

* model-axis attention: the K cells of a row attend to each other (consensus / disagreement);
* row-axis attention: every cell attends to the context-row cells of the same model
  (the model's track record), so query rows never leak into each other.
The head emits an instance-wise weight w_ik per (row, model); the fused prediction is a
logarithmic opinion pool  log p(y=c|row i) ∝ b + sum_k w_ik log p_k(c|row i).
The network is equivariant to permutations of rows, models and classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import N_CLASS_FEATURES, N_FEATURES


class Block(nn.Module):
    def __init__(self, d, heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.h = heads
        self.ln_m, self.ln_r, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv_m = nn.Linear(d, 3 * d)
        self.out_m = nn.Linear(d, d)
        self.q_r, self.kv_r = nn.Linear(d, d), nn.Linear(d, 2 * d)
        self.out_r = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def _split(self, x):  # (..., L, d) -> (..., h, L, d/h)
        *lead, L, d = x.shape
        return x.view(*lead, L, self.h, d // self.h).transpose(-2, -3)

    def _merge(self, x):
        *lead, h, L, dh = x.shape
        return x.transpose(-2, -3).reshape(*lead, L, h * dh)

    def forward(self, x, n_ctx, model_mask=None):
        # x: (B, N, K, d) ; first n_ctx rows are context rows
        B, N, K, d = x.shape
        # --- model axis ---
        h = self.ln_m(x).reshape(B * N, K, d)
        q, k, v = self.qkv_m(h).chunk(3, -1)
        am = None
        if model_mask is not None:  # (B, K) True = valid
            am = model_mask[:, None, None, :].expand(B, N, 1, K).reshape(B * N, 1, 1, K)
        o = F.scaled_dot_product_attention(self._split(q), self._split(k), self._split(v), attn_mask=am)
        x = x + self.drop(self.out_m(self._merge(o)).view(B, N, K, d))
        # --- row axis (keys/values from context rows only) ---
        h = self.ln_r(x).permute(0, 2, 1, 3).reshape(B * K, N, d)
        q = self.q_r(h)
        k, v = self.kv_r(h[:, :n_ctx]).chunk(2, -1)
        o = F.scaled_dot_product_attention(self._split(q), self._split(k), self._split(v))
        o = self.out_r(self._merge(o)).view(B, K, N, d).permute(0, 2, 1, 3)
        x = x + self.drop(o)
        x = x + self.drop(self.ff(self.ln_f(x)))
        return x


class FusionPFN(nn.Module):
    def __init__(self, d=128, layers=6, heads=4, dropout=0.0, residual=False, dc=32):
        super().__init__()
        self.residual = residual
        self.embed = nn.Sequential(nn.Linear(N_FEATURES, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads, dropout=dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.temp = nn.Linear(d, 1)  # row-level sharpness from mean-pooled cells
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.temp.weight)
        nn.init.zeros_(self.temp.bias)
        if residual:
            # class-equivariant correction r_ic = rho( mean_k phi([h_ik, g_ikc]) )
            self.res_h = nn.Linear(d, dc)
            self.res_g = nn.Linear(N_CLASS_FEATURES, dc)
            self.res_phi = nn.Sequential(nn.GELU(), nn.Linear(dc, dc), nn.GELU())
            self.res_rho = nn.Sequential(nn.Linear(dc, dc), nn.GELU(), nn.Linear(dc, 1))
            nn.init.zeros_(self.res_rho[-1].weight)
            nn.init.zeros_(self.res_rho[-1].bias)

    def forward(self, F_ctx, F_qry, logP_qry, model_mask=None, G_qry=None, return_parts=False):
        """Return fused log-probabilities for query rows (B, m, C) and weights (B, m, K).

        G_qry: (B, m, K, C, N_CLASS_FEATURES) class features (required when residual=True).
        """
        n_ctx = F_ctx.shape[1]
        x = self.embed(torch.cat([F_ctx, F_qry], 1))
        for blk in self.blocks:
            x = blk(x, n_ctx, model_mask)
        x = self.ln(x[:, n_ctx:])  # query rows
        K = x.shape[2]
        w = self.head(x).squeeze(-1)  # (B, m, K)
        if model_mask is not None:
            w = w.masked_fill(~model_mask[:, None, :], 0.0)
            k_eff = model_mask.sum(-1).clamp_min(1).float()[:, None, None]
        else:
            k_eff = float(K)
        # at init: w = 1/K -> geometric mean of base models (a strong, safe default)
        w = (w + 1.0) / k_eff
        pool = torch.einsum("bmk,bmkc->bmc", w, logP_qry)
        r = torch.zeros_like(pool)
        if self.residual:
            z = self.res_phi(self.res_h(x).unsqueeze(3) + self.res_g(G_qry.to(x.dtype)))  # (B,m,K,C,dc)
            if model_mask is not None:
                mk = model_mask[:, None, :, None, None].to(z.dtype)
                z = (z * mk).sum(2) / mk.sum(2).clamp_min(1)
            else:
                z = z.mean(2)
            r = self.res_rho(z).squeeze(-1).to(pool.dtype)  # (B,m,C)
        logp = torch.log_softmax(pool + r, -1)
        if return_parts:
            return logp, w, r
        return logp, w
