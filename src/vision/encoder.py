import torch, torch.nn as nn, torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=256, patch=16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                        # [B, C, H', W']
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        return self.norm(x)                     # [B, N, D]

class TransformerEncoder(nn.Module):
    def __init__(self, dim=256, depth=6, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, heads, batch_first=True),
                "ln2": nn.LayerNorm(dim),
                "mlp": nn.Sequential(
                    nn.Linear(dim, int(dim*mlp_ratio)),
                    nn.GELU(),
                    nn.Linear(int(dim*mlp_ratio), dim),
                ),
            }) for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.layers:
            h = blk["ln1"](x)
            attn_out, _ = blk["attn"](h, h, h, need_weights=False)
            x = x + attn_out
            h = blk["ln2"](x)
            x = x + blk["mlp"](h)
        return x

class SimpleJEPAEncoder(nn.Module):
    def __init__(self, embed_dim=256, depth=6, heads=4, patch=16):
        super().__init__()
        self.embed = PatchEmbed(embed_dim=embed_dim, patch=patch)
        self.enc = TransformerEncoder(dim=embed_dim, depth=depth, heads=heads)
        self.proj = nn.LayerNorm(embed_dim)

    def forward(self, x):                       # x: [B,3,H,W]
        z = self.embed(x)                       # [B,N,D]
        z = self.enc(z)                         # [B,N,D]
        return self.proj(z)                     # [B,N,D]

class Predictor(nn.Module):
    def __init__(self, dim=256, pred_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, pred_dim), nn.GELU(), nn.Linear(pred_dim, dim)
        )
    def forward(self, x): return self.net(x)

def jepa_loss(context_tokens, target_tokens, mask_indices):
    """
    Hardened cosine loss with eps and safe token filtering.
    Prevents numerical instability from zero/near-zero vectors in bf16.
    """
    eps = 1e-6
    ctx_raw = context_tokens
    tgt_raw = target_tokens.detach()
    
    # Compute norms and identify safe (non-degenerate) tokens
    ctx_norm = ctx_raw.norm(dim=-1, keepdim=True)  # [B, N, 1]
    tgt_norm = tgt_raw.norm(dim=-1, keepdim=True)  # [B, N, 1]
    
    safe_ctx = ctx_norm.squeeze(-1) > eps  # [B, N]
    safe_tgt = tgt_norm.squeeze(-1) > eps  # [B, N]
    safe = safe_ctx & safe_tgt & mask_indices  # [B, N] - keep only valid, non-degenerate tokens
    
    # Normalize with eps for numerical stability
    ctx = F.normalize(ctx_raw, dim=-1, eps=eps)  # [B, N, D]
    tgt = F.normalize(tgt_raw, dim=-1, eps=eps)  # [B, N, D]
    
    # Compute similarity only for safe tokens
    sim = 1.0 - (ctx * tgt).sum(-1)  # [B, N]
    
    # Mask and normalize per sample
    mask_f = safe.float()  # [B, N]
    num_per_sample = mask_f.sum(dim=1).clamp_min(1.0)  # [B] - number of valid tokens per sample
    
    # Average per sample, then mean over batch
    loss_per_sample = (sim * mask_f).sum(dim=1) / num_per_sample  # [B]
    loss = loss_per_sample.mean()
    
    # Return loss and stats for telemetry
    num_valid = safe.sum().item()
    num_total = mask_indices.sum().item()
    pct_filtered = (1.0 - num_valid / max(num_total, 1)) * 100.0
    
    return loss, {
        "num_valid_tokens": num_valid,
        "num_total_tokens": num_total,
        "pct_filtered": pct_filtered,
        "loss_per_sample": loss_per_sample.detach(),
    }
