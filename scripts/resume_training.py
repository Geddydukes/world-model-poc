"""Resume training from a checkpoint around step 2000-3000."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from src.memory.episodic import EpisodicMemory
from src.trainers.ssl_vision import VisionSSLTrainer
from src.utils.device import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description="Resume vision training from checkpoint")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--date", default="clevrer_train")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    ap.add_argument("--start-step", type=int, default=0, help="Starting step (if different from checkpoint)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found at {cfg_path}")
    
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = resolve_device(cfg.get("device", "auto"))
    ckpt_dir = Path(cfg["outputs"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    memory = EpisodicMemory(cfg["memory"]["sqlite_path"], cfg["memory"]["embed_dir"])
    try:
        vision_trainer = VisionSSLTrainer(cfg, device=device)
        loss = vision_trainer.run(
            date=args.date,
            checkpoint_dir=ckpt_dir,
            memory=memory,
            resume_from=ckpt_path,
            start_step=args.start_step,
        )
        print(f"Final loss: {loss:.6f}")
    finally:
        memory.close()


if __name__ == "__main__":
    main()

