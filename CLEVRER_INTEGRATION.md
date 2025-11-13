# CLEVRER Dataset Integration

This document explains how to integrate and use the CLEVRER dataset with the world model training pipeline.

## Overview

CLEVRER (CoLlision Events for Video REpresentation and Reasoning) is a dataset of 20,000 videos (10k train, 5k val, 5k test) with annotations for causal reasoning tasks. Each video is 5 seconds long and includes questions about descriptive, explanatory, predictive, and counterfactual reasoning.

## Setup

### 1. Organize Uploaded Files

If you've uploaded CLEVRER files (videos and annotations), first organize them:

```bash
python scripts/organize_clevrer.py \
    --source-dir /path/to/uploaded/files \
    --target-dir data/clevrer
```

This script will:
- Extract any archive files (zip, tar, tar.gz)
- Organize videos into `data/clevrer/videos/`
- Organize annotations into `data/clevrer/annotations/`
- Attempt to separate by train/val/test splits if filenames indicate it

### 2. Ingest CLEVRER Videos

Ingest the CLEVRER videos into the pipeline format:

```bash
# Ingest training set
python scripts/ingest_clevrer.py \
    --video-dir data/clevrer/videos/train \
    --annotation-dir data/clevrer/annotations \
    --split train \
    --date clevrer_train \
    --target-fps 8.0 \
    --max-workers 4

# Ingest validation set
python scripts/ingest_clevrer.py \
    --video-dir data/clevrer/videos/val \
    --annotation-dir data/clevrer/annotations \
    --split val \
    --date clevrer_val \
    --target-fps 8.0 \
    --max-workers 4

# Ingest test set
python scripts/ingest_clevrer.py \
    --video-dir data/clevrer/videos/test \
    --annotation-dir data/clevrer/annotations \
    --split test \
    --date clevrer_test \
    --target-fps 8.0 \
    --max-workers 4
```

This will:
- Decode videos into frames
- Extract audio
- Store sequences in `data/sequences/clevrer_*`
- Save frames in `data/frames/clevrer_*`
- Save audio in `data/audio/clevrer_*`
- Register clips in episodic memory
- Save annotations alongside each clip

## Usage

### Using CLEVRER Dataset in Training

The CLEVRER dataset can be used in two ways:

#### 1. As Pre-Ingested Data (Recommended)

After ingestion, CLEVRER data is available through the standard glob patterns. Update your config:

```yaml
data:
  frame_glob_today: "data/frames/clevrer_train/**/*.jpg"
  audio_glob_today: "data/audio/clevrer_train/**/*.wav"
  # Or mix with other data:
  frame_glob_replay: "data/frames/clevrer_train/**/*.jpg"
  audio_glob_replay: "data/audio/clevrer_train/**/*.wav"
```

#### 2. Using CLEVRER Dataset Classes

For direct access to videos and annotations:

```python
from src.data.clevrer import CLEVRERVideoDataset, CLEVRERFrameDataset
from torchvision import transforms

# Load videos directly (decodes on-the-fly)
dataset = CLEVRERVideoDataset(
    video_dir="data/clevrer/videos/train",
    annotation_dir="data/clevrer/annotations",
    split="train",
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    target_fps=8.0,
)

# Or use pre-extracted frames
frame_dataset = CLEVRERFrameDataset(
    frame_dir="data/frames/clevrer_train",
    annotation_dir="data/clevrer/annotations",
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
)

# Access sample
sample = dataset[0]
frames = sample["frames"]  # [T, C, H, W] tensor
questions = sample.get("questions", [])
answers = sample.get("answers", [])
```

## Dataset Structure

After ingestion, the structure will be:

```
data/
├── clevrer/
│   ├── videos/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
├── sequences/
│   ├── clevrer_train/
│   │   ├── {clip_id}/
│   │   │   ├── frames.npy
│   │   │   ├── metadata.json
│   │   │   ├── annotation.json  # CLEVRER annotation
│   │   │   ├── audio.wav
│   │   │   └── audio_logmel.npz
│   ├── clevrer_val/
│   └── clevrer_test/
├── frames/
│   ├── clevrer_train/
│   │   ├── {clip_id}/
│   │   │   └── frame_*.jpg
│   ├── clevrer_val/
│   └── clevrer_test/
└── audio/
    ├── clevrer_train/
    │   └── segments/
    │       └── {clip_id}.wav
    ├── clevrer_val/
    └── clevrer_test/
```

## Annotation Format

CLEVRER annotations are stored as JSON. Each annotation includes:

- `video_id`: Identifier for the video
- `questions`: List of question dictionaries
- `answers`: List of answer dictionaries
- `events`: (Optional) Event annotations
- `objects`: (Optional) Object annotations

The annotation is saved alongside each clip in `data/sequences/{date}/{clip_id}/annotation.json`.

## Configuration

Update `configs/default.yaml` to enable CLEVRER:

```yaml
data:
  clevrer:
    video_dir: "data/clevrer/videos"
    annotation_dir: "data/clevrer/annotations"
    train_dir: "data/clevrer/train"
    val_dir: "data/clevrer/val"
    test_dir: "data/clevrer/test"
    use_clevrer: true
    clevrer_frame_glob: "data/frames/clevrer/**/*.jpg"
    clevrer_audio_glob: "data/audio/clevrer/**/*.wav"
```

## Training with CLEVRER

Once ingested, you can train on CLEVRER data using the standard training pipeline:

```bash
# Train vision SSL on CLEVRER
python sleep.py --config configs/default.yaml --date clevrer_train

# Or use the individual trainers
python -c "
from src.trainers.vision_trainer import train_vision
import yaml
cfg = yaml.safe_load(open('configs/default.yaml'))
cfg['data']['frame_glob_today'] = 'data/frames/clevrer_train/**/*.jpg'
train_vision(cfg, 'clevrer_train')
"
```

## Notes

- CLEVRER videos are 5 seconds long at 30 FPS (150 frames each)
- With `target_fps=8.0`, each video yields ~40 frames
- Annotations are preserved and accessible during training
- The dataset supports both video decoding and pre-extracted frames
- Use `max_videos` parameter to limit processing for testing

