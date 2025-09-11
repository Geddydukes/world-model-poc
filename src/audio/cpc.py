import math, torch, torch.nn as nn, torch.nn.functional as F, torchaudio

class LogMel(nn.Module):
    def __init__(self, sr=16000, mels=64, win=640, hop=320):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, win_length=win, hop_length=hop, n_mels=mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
    def forward(self, wav):                      # wav: [B,T]
        S = self.melspec(wav)                    # [B,M,Frames]
        return self.to_db(S)

class CPCEncoder(nn.Module):
    def __init__(self, in_mels=64, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_mels, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
        )
        self.proj = nn.Linear(hidden, hidden)
    def forward(self, mel_db):                   # [B,M,T]
        h = self.conv(mel_db).permute(0,2,1)     # [B,T,H]
        return self.proj(h)

class CPCContext(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
    def forward(self, h):                        # [B,T,H]
        out,_ = self.gru(h); return out

class CPCPredictor(nn.Module):
    def __init__(self, hidden=256, pred_steps=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(pred_steps)])
    def forward(self, c):
        return [L(c) for L in self.layers]       # list of [B,T,H]

def cpc_infonce_loss(context, future_feats, preds):
    total = 0.0; K = len(preds)
    B,T,H = context.shape
    for k in range(1, K+1):
        pred = preds[k-1][:, :-k, :]            # [B,T-k,H]
        targ = future_feats[:, k:, :]           # [B,T-k,H]
        pb = pred.reshape(-1,H); tb = targ.reshape(-1,H)
        logits = (pb @ tb.t()) / math.sqrt(H)   # [BT,BT]
        labels = torch.arange(pb.size(0), device=pb.device)
        total += F.cross_entropy(logits, labels)
    return total / K
