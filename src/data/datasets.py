import glob, random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchaudio

class ImageGlobDataset(Dataset):
    def __init__(self, globs, transform=None, max_items=None):
        if isinstance(globs, str): globs = [globs]
        files = []
        for g in globs: files.extend(glob.glob(g, recursive=True))
        self.files = sorted(files)
        if max_items and len(self.files) > max_items:
            self.files = random.sample(self.files, max_items)
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, path

class AudioGlobDataset(Dataset):
    def __init__(self, globs, sample_rate=16000, max_items=None):
        if isinstance(globs, str): globs = [globs]
        files = []
        for g in globs: files.extend(glob.glob(g, recursive=True))
        self.files = sorted(files)
        if max_items and len(self.files) > max_items:
            self.files = random.sample(self.files, max_items)
        self.sr = sample_rate

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav.mean(dim=0), path  # mono [T]
