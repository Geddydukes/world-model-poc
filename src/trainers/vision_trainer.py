import os, random, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.vision.encoder import SimpleJEPAEncoder, Predictor, jepa_loss
from src.vision.augment import get_vision_transforms
from src.data.datasets import ImageGlobDataset
from src.utils.tensor import autocast_enabled, amp_dtype, count_params
from src.memory.episodic import EpisodicMemory

def random_token_mask(B, N, ratio=0.6, device="cpu"):
    keep = int(N*(1.0-ratio))
    mask = torch.zeros(B,N, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(N, device=device)[keep:]
        mask[b, idx] = True
    return mask

def train_vision(cfg, date):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    prec = amp_dtype()
    tr = get_vision_transforms(cfg["data"]["image_size"])
    today_glob = cfg["data"]["frame_glob_today"].format(date=date)

    ds = ImageGlobDataset(today_glob, transform=tr)
    dl = DataLoader(ds, batch_size=cfg["train"]["micro_batch"], shuffle=True,
                    num_workers=cfg["data"]["num_workers"], drop_last=True)

    enc_s = SimpleJEPAEncoder(cfg["vision"]["embed_dim"], cfg["vision"]["depth"], cfg["vision"]["heads"]).to(device)
    enc_t = SimpleJEPAEncoder(cfg["vision"]["embed_dim"], cfg["vision"]["depth"], cfg["vision"]["heads"]).to(device)
    enc_t.load_state_dict(enc_s.state_dict()); [p.requires_grad_(False) for p in enc_t.parameters()]
    pred = Predictor(cfg["vision"]["embed_dim"], cfg["vision"]["pred_dim"]).to(device)

    opt = torch.optim.AdamW(list(enc_s.parameters())+list(pred.parameters()),
                            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    ema = cfg["train"]["ema_momentum"]
    steps = cfg["train"]["steps_vision"]; accum = max(1, cfg["train"]["batch_size"]//cfg["train"]["micro_batch"])
    scal = autocast_enabled(str(device))
    mem = EpisodicMemory(cfg["memory"]["sqlite_path"], cfg["memory"]["embed_dir"])

    it = iter(dl); last = 0.0
    for step in tqdm(range(steps), desc="[vision]"):
        try: x, paths = next(it)
        except StopIteration: it = iter(dl); x, paths = next(it)
        x = x.to(device)
        with torch.autocast(device_type=str(device), dtype=prec, enabled=scal):
            z_s = enc_s(x); z_t = enc_t(x)
            mask = random_token_mask(z_s.size(0), z_s.size(1), cfg["vision"]["mask_ratio"], device)
            loss = jepa_loss(pred(z_s), z_t, mask)

        loss.backward()
        if (step+1)%accum==0:
            torch.nn.utils.clip_grad_norm_(list(enc_s.parameters())+list(pred.parameters()), 1.0)
            opt.step(); opt.zero_grad()
            with torch.no_grad():
                for ps, pt in zip(enc_s.parameters(), enc_t.parameters()):
                    pt.data = pt.data*ema + ps.data*(1.0-ema)
        last = float(loss.item())

        if (step+1)%200==0:  # write some embeddings to memory (teacher for stability)
            enc_t.eval()
            with torch.no_grad():
                z = enc_t(x).mean(dim=1).cpu().numpy()
            for i,p in enumerate(paths): mem.add_embedding(date, p, "image", z[i])
            enc_t.train()

    os.makedirs(cfg["outputs"]["ckpt_dir"], exist_ok=True)
    torch.save(enc_s.state_dict(), os.path.join(cfg["outputs"]["ckpt_dir"], f"{date}_student.pt"))
    torch.save(enc_t.state_dict(), os.path.join(cfg["outputs"]["ckpt_dir"], f"{date}_teacher.pt"))
    return last
