import argparse, numpy as np
from PIL import Image
import torch, torchvision.transforms as T
from src.memory.episodic import EpisodicMemory
from src.vision.encoder import SimpleJEPAEncoder

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_path", required=True)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = SimpleJEPAEncoder().to(device).eval()
    tr = T.Compose([T.Resize(224, antialias=True), T.CenterCrop(224), T.ToTensor()])
    img = tr(Image.open(args.query_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        z = enc(img).mean(dim=1).squeeze(0).cpu().numpy()

    mem = EpisodicMemory()
    for path, sim in mem.nearest(z, modality="image", topk=args.topk):
        print(f"{sim: .3f}  {path}")
