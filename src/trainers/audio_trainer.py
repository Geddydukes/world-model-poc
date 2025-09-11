import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.datasets import AudioGlobDataset
from src.audio.cpc import LogMel, CPCEncoder, CPCContext, CPCPredictor, cpc_infonce_loss
from src.utils.tensor import autocast_enabled, amp_dtype
from src.memory.episodic import EpisodicMemory

def train_audio(cfg, date):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    prec = amp_dtype()

    ds = AudioGlobDataset(cfg["data"]["audio_glob_today"].format(date=date),
                          sample_rate=cfg["audio"]["sample_rate"])
    dl = DataLoader(ds, batch_size=cfg["train"]["micro_batch"], shuffle=True,
                    num_workers=cfg["data"]["num_workers"], drop_last=True)

    fea = LogMel(cfg["audio"]["sample_rate"], cfg["audio"]["mel_bins"],
                 cfg["audio"]["win_length"], cfg["audio"]["hop_length"]).to(device)
    enc = CPCEncoder(cfg["audio"]["mel_bins"], cfg["audio"]["cpc_hidden"]).to(device)
    ctx = CPCContext(cfg["audio"]["cpc_context"]).to(device)
    pred = CPCPredictor(cfg["audio"]["cpc_context"], cfg["audio"]["cpc_pred_steps"]).to(device)

    opt = torch.optim.AdamW(list(enc.parameters())+list(ctx.parameters())+list(pred.parameters()),
                            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scal = autocast_enabled(str(device))
    steps = cfg["train"]["steps_audio"]; mem = EpisodicMemory(cfg["memory"]["sqlite_path"], cfg["memory"]["embed_dir"])

    it = iter(dl); last = 0.0
    for step in tqdm(range(steps), desc="[audio]"):
        try: wav, paths = next(it)
        except StopIteration: it = iter(dl); wav, paths = next(it)
        wav = wav.to(device)
        with torch.autocast(device_type=str(device), dtype=prec, enabled=scal):
            mel = fea(wav)               # [B,M,T]
            h = enc(mel)                 # [B,T,H]
            c = ctx(h)                   # [B,T,H]
            loss = cpc_infonce_loss(c, h, pred(c))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(ctx.parameters())+list(pred.parameters()), 1.0)
        opt.step(); last = float(loss.item())
        if (step+1)%200==0:
            with torch.no_grad():
                z = c.mean(dim=1).cpu().numpy()
            for i,p in enumerate(paths): mem.add_embedding(date, p, "audio", z[i])
    return last
