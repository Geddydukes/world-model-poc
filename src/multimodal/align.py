import torch.nn as nn, torch.nn.functional as F

def proj_head(in_dim, out_dim):
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))

def clip_like_contrastive_loss(a, v, temp=0.07):  # a,v: [B,D]
    a = F.normalize(a, dim=-1); v = F.normalize(v, dim=-1)
    logits = a @ v.t() / temp
    import torch
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5*(F.cross_entropy(logits, labels)+F.cross_entropy(logits.t(), labels))
