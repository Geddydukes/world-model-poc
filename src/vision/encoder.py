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
    ctx = context_tokens[mask_indices]          # [M,D]
    tgt = target_tokens[mask_indices].detach()
    ctx = F.normalize(ctx, dim=-1)
    tgt = F.normalize(tgt, dim=-1)
    return (1.0 - (ctx * tgt).sum(dim=-1)).mean()
