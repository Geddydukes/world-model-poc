import torch

def autocast_enabled(device: str):
    return device in ("cuda", "mps")

def amp_dtype():
    return torch.bfloat16

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
