import numpy as np, torch, random
from tqdm import tqdm
from src.memory.episodic import EpisodicMemory
from src.multimodal.align import proj_head, clip_like_contrastive_loss

def train_align(cfg, date):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mem = EpisodicMemory(cfg["memory"]["sqlite_path"], cfg["memory"]["embed_dir"])
    imgs = mem.all_for_date(date, modality="image")
    auds = mem.all_for_date(date, modality="audio")
    if not imgs or not auds: return 0.0
    B = min(len(imgs), len(auds), cfg["train"]["micro_batch"]*8)
    vi = np.stack([x[1] for x in random.sample(imgs, B)], 0)
    ai = np.stack([x[1] for x in random.sample(auds, B)], 0)
    v = torch.tensor(vi, dtype=torch.float32, device=device)
    a = torch.tensor(ai, dtype=torch.float32, device=device)

    D = v.shape[-1]; proj_dim = cfg["vision"]["pred_dim"]//2
    pv = proj_head(D, proj_dim).to(device); pa = proj_head(D, proj_dim).to(device)
    opt = torch.optim.AdamW(list(pv.parameters())+list(pa.parameters()),
                            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    last = 0.0
    for _ in tqdm(range(cfg["train"]["steps_align"]), desc="[align]"):
        opt.zero_grad()
        loss = clip_like_contrastive_loss(pa(a), pv(v))
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(pv.parameters())+list(pa.parameters()), 1.0)
        opt.step(); last = float(loss.item())
    return last
