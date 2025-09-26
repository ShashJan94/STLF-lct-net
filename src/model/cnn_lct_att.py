# src/models/cnn_lct_att.py
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
# … your TemporalCNN, PointwiseMix, LCTCell, LCTBlock, RelPosMHA, CNN_LCT_Att …

class TemporalCNN(nn.Module):
    def __init__(self, in_feats, out_channels=32, kernel_size=5, activation="gelu", p=0.15):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_feats, out_channels, kernel_size, padding=pad, bias=True)
        self.act  = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ln   = nn.LayerNorm(out_channels)
        self.do   = nn.Dropout(p)
    def forward(self, x):  # [B,T,d]
        y = x.transpose(1,2)          # [B,d,T]
        y = self.conv(y).transpose(1,2)# [B,T,C]
        y = self.ln(y); y = self.act(y); y = self.do(y)
        return y

class PointwiseMix(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.15):
        super().__init__()
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=True)
        self.ln = nn.LayerNorm(out_ch)
        self.do = nn.Dropout(p)
    def forward(self, x):  # [B,T,C]
        y = self.pw(x.transpose(1,2)).transpose(1,2)
        y = self.ln(y); y = self.do(y)
        return y

class LCTCell(nn.Module):
    def __init__(self, in_dim, hidden_dim, dt=0.2):
        super().__init__()
        self.dt = dt
        self.Wx = nn.Linear(in_dim, hidden_dim, bias=True)
        self.Wh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = torch.tanh
        self.tau_raw = nn.Parameter(torch.randn(hidden_dim)*0.1)
        self.softplus = nn.Softplus(beta=1.0)
    def forward(self, x_t, h_t):
        tau = self.softplus(self.tau_raw) + 1e-3  # positive
        f = self.act(self.Wx(x_t) + self.Wh(h_t))
        dh = -h_t / tau + f
        return h_t + self.dt * dh

class LCTBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, p=0.15):
        super().__init__()
        self.pre_ln  = nn.LayerNorm(in_dim)
        self.cell    = LCTCell(in_dim, hidden_dim, dt=0.2)  # dt=0.2 for stability
        self.post_ln = nn.LayerNorm(hidden_dim)
        self.do      = nn.Dropout(p)

    def forward(self, X):  # [B,T,C]
        Xn = self.pre_ln(X)
        B, T, _ = Xn.shape
        H = Xn.new_zeros(B, self.cell.Wx.out_features)   # single, correct init
        outs = []
        for t in range(T):
            H = self.cell(Xn[:, t, :], H)
            outs.append(H)
        Y = torch.stack(outs, dim=1)   # [B,T,H]
        Y = self.post_ln(Y)
        return self.do(Y)

class RelPosMHA(nn.Module):
    def __init__(self, model_dim, num_heads=4, max_len=256, attn_p=0.1, causal=True):
        super().__init__()
        assert model_dim % num_heads == 0
        self.d = model_dim
        self.h = num_heads
        self.dk = model_dim // num_heads
        self.max_len = max_len
        self.causal = causal

        self.q = nn.Linear(model_dim, model_dim, bias=False)
        self.k = nn.Linear(model_dim, model_dim, bias=False)
        self.v = nn.Linear(model_dim, model_dim, bias=False)
        self.o = nn.Linear(model_dim, model_dim, bias=False)

        self.rel = nn.Parameter(torch.empty(num_heads, 2*max_len-1, self.dk))
        nn.init.normal_(self.rel, std=0.02)

        self.pre_ln  = nn.LayerNorm(model_dim)
        self.attn_do = nn.Dropout(attn_p)
        self.out_do  = nn.Dropout(attn_p)

    def forward(self, H):  # [B,T,D]
        B,T,D = H.shape
        assert T <= self.max_len
        X = self.pre_ln(H)

        Q = self.q(X).view(B,T,self.h,self.dk).transpose(1,2) # [B,h,T,dk]
        K = self.k(X).view(B,T,self.h,self.dk).transpose(1,2)
        V = self.v(X).view(B,T,self.h,self.dk).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.dk)  # [B,h,T,T]

        # relative position
        pos = torch.arange(T, device=H.device)
        rel_idx = (pos[None,:] - pos[:,None]) + (self.max_len - 1)          # [T,T]
        rel_emb = self.rel[:, rel_idx, :]                                   # [h,T,T,dk]
        rel_scores = torch.einsum("bhtd,htTd->bhtT", Q, rel_emb)
        scores = scores + rel_scores

        if self.causal:
            mask = torch.ones(T, T, device=H.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float('-inf'))

        A = F.softmax(scores, dim=-1)
        A = self.attn_do(A)
        O = torch.matmul(A, V).transpose(1,2).contiguous().view(B,T,D)
        return self.out_do(self.o(O)), A

class CNN_LCT_Att(nn.Module):
    """
    CNN -> Mix1x1 -> LCT -> (optional) MHA + FFN (both gated residuals)
    Predicts a residual over a learned softmax blend of {last, lag-24, lag-48, lag-168}.
    Works for any window length T >= 1 (lags auto-masked if not available).
    """
    def __init__(
        self,
        in_feats: int,
        cnn_channels: int = 32,
        cnn_kernel: int = 5,
        mix_channels: int = 64,
        lct_hidden: int = 96,      # divisible by num_heads
        use_mha: bool = True,
        num_heads: int = 2,
        max_len: int = 512,        # set to win.T at creation time
        attn_p: float = 0.10,
        p: float = 0.15,
        baseline_lags: tuple = (1, 24, 48, 168),
        baseline_temp: float = 0.8 # hours: last, day, 2-day, week
    ):
        super().__init__()
        assert lct_hidden % max(1, num_heads) == 0, "lct_hidden must be divisible by num_heads"

        self.use_mha = use_mha
        self.baseline_lags = tuple(baseline_lags)

        # CNN over time + 1x1 mixing
        self.cnn = TemporalCNN(in_feats, out_channels=cnn_channels, kernel_size=cnn_kernel, p=p)
        self.mix = PointwiseMix(in_ch=in_feats + cnn_channels, out_ch=mix_channels, p=p)

        # LCT sequence model
        self.lct = LCTBlock(in_dim=mix_channels, hidden_dim=lct_hidden, p=p)

        # Optional Multi-Head Attention (causal) with gated residual
        if use_mha:
            self.mha = RelPosMHA(model_dim=lct_hidden, num_heads=num_heads, max_len=max_len, attn_p=attn_p, causal=True)
            self.mha_gate = nn.Parameter(torch.tensor(-4.59511985))  # sigmoid ≈ 0.01

            # Tiny Transformer-style FFN with its own gated residual
            self.ffn = nn.Sequential(
                nn.LayerNorm(lct_hidden),
                nn.Linear(lct_hidden, 4 * lct_hidden),
                nn.GELU(),
                nn.Dropout(p),
                nn.Linear(4 * lct_hidden, lct_hidden),
                nn.Dropout(p),
            )
            self.ffn_gate = nn.Parameter(torch.tensor(-4.59511985))  # sigmoid ≈ 0.01

        # Residual head (predicts delta in scaled space)
        self.out = nn.Linear(lct_hidden, 1)

        # Multi-lag softmax baseline weights (from the last hidden state z_last)
        self.baseline_proj = nn.Linear(lct_hidden, len(self.baseline_lags), bias=True)
        self.baseline_temp = float(baseline_temp)
        with torch.no_grad():
            self.baseline_proj.weight.zero_()
            if self.baseline_lags == (1, 24, 48, 72, 168):
                # favor last & 24h; keep 48h/72h mild; 168h low unless T>=168
                self.baseline_proj.bias.copy_(torch.tensor([1.2, 1.0, 0.2, 0.0, -1.2]))
            elif self.baseline_lags == (1, 24, 48, 168):
                self.baseline_proj.bias.copy_(torch.tensor([1.2, 1.0, -1.2, -2.0]))
            else:
                b = torch.full((len(self.baseline_lags),), -0.5)
                if 1 in self.baseline_lags:
                    b[list(self.baseline_lags).index(1)] = 0.8
                self.baseline_proj.bias.copy_(b)


    def forward(self, X):  # X: [B,T,d] ; channel 0 = scaled load
        # CNN -> Mix -> LCT
        y_cnn  = self.cnn(X)                           # [B,T,Cc]
        merged = torch.cat([X, y_cnn], dim=-1)         # [B,T,d+Cc]
        mixed  = self.mix(merged)                      # [B,T,F]
        Hseq   = self.lct(mixed)                       # [B,T,H]

        # Optional: MHA (causal) + FFN, both with gated residuals
        if self.use_mha:
            Hattn, _ = self.mha(Hseq)                  # pre-LN & causal mask inside MHA
            gate_a = torch.sigmoid(self.mha_gate)      # scalar in (0,1)
            Hseq   = Hseq + gate_a * Hattn

            Hffn   = self.ffn(Hseq)
            gate_f = torch.sigmoid(self.ffn_gate)
            Hseq   = Hseq + gate_f * Hffn

        # Residual over multi-lag softmax baseline
        z_last = Hseq[:, -1, :]                        # [B,H]
        delta  = self.out(z_last).squeeze(-1)          # [B]

        B, T, _ = X.shape
        # Gather candidate baselines for requested lags
        cand_vals, avail_mask = [], []
        for L in self.baseline_lags:
            idx = T - L                                # correct index for horizon=1
            if idx >= 0:
                cand_vals.append(X[:, idx, 0])         # [B]
                avail_mask.append(1)
            else:
                cand_vals.append(torch.zeros(B, device=X.device, dtype=X.dtype))
                avail_mask.append(0)
        C = torch.stack(cand_vals, dim=1)              # [B, K]
        avail = torch.tensor(avail_mask, device=X.device, dtype=torch.bool)[None, :]  # [1,K]

        logits = self.baseline_proj(z_last)            # [B, K]
        logits = torch.where(avail, logits, torch.full_like(logits, -1e9))  # mask missing lags
        w = torch.softmax(logits / self.baseline_temp, dim=1)  # [B, K]
        baseline = (w * C).sum(dim=1)                  # [B]

        pred = baseline + delta                        # [B]
        return pred.unsqueeze(-1)                      # [B,1]

def trim_baseline_lags_inplace(model: CNN_LCT_Att, T: int) -> None:
    """Drop any lags > T and slice baseline head to match."""
    keep = [L for L in model.baseline_lags if L <= T]
    if tuple(keep) == tuple(model.baseline_lags):
        return
    idx = [model.baseline_lags.index(L) for L in keep]
    with torch.no_grad():
        model.baseline_proj.weight = nn.Parameter(model.baseline_proj.weight[idx].clone())
        model.baseline_proj.bias   = nn.Parameter(model.baseline_proj.bias[idx].clone())
    model.baseline_lags = tuple(keep)

def build_cnn_lct_att(in_feats: int, T: int, **kw) -> CNN_LCT_Att:
    """Factory that sets max_len=T and trims lags > T."""
    m = CNN_LCT_Att(in_feats=in_feats, max_len=T, **kw)
    trim_baseline_lags_inplace(m, T)
    return m
