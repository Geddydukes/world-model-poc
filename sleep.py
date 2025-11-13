"""Nightly orchestration for the world model pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict

import yaml

from src.memory.episodic import EpisodicMemory
from src.trainers.align_multimodal import AlignmentTrainer
from src.trainers.ssl_audio import AudioSSLTrainer
from src.trainers.ssl_vision import VisionSSLTrainer
from src.trainers.world_predict import WorldPredictTrainer
from src.utils.device import resolve_device


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run nightly training tasks")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found at {cfg_path}")
    cfg = load_config(cfg_path)

    run_date = args.date or cfg.get("data", {}).get("today_date") or dt.date.today().isoformat()
    device_str = cfg.get("device", "auto")
    device = resolve_device(device_str)

    ckpt_dir = Path(cfg["outputs"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(cfg["outputs"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    memory = EpisodicMemory(cfg["memory"]["sqlite_path"], cfg["memory"]["embed_dir"])
    metrics: list[tuple[str, float]] = []
    try:
        if cfg["train"].get("run_ssl_vision", False):
            vision_trainer = VisionSSLTrainer(cfg, device=device)
            loss = vision_trainer.run(date=run_date, checkpoint_dir=ckpt_dir, memory=memory)
            metrics.append(("vision_ssl_loss", loss))

        if cfg["train"].get("run_ssl_audio", False):
            audio_trainer = AudioSSLTrainer(cfg, device=device)
            loss = audio_trainer.run(date=run_date, checkpoint_dir=ckpt_dir, memory=memory)
            metrics.append(("audio_ssl_loss", loss))

        if cfg["train"].get("run_align", False):
            align_trainer = AlignmentTrainer(cfg)
            loss = align_trainer.run(date=run_date, checkpoint_dir=ckpt_dir, memory=memory)
            metrics.append(("align_loss", loss))

        if cfg["train"].get("run_world_predict", False):
            world_trainer = WorldPredictTrainer(cfg)
            loss = world_trainer.run(date=run_date, checkpoint_dir=ckpt_dir, memory=memory)
            metrics.append(("world_predict_loss", loss))
    finally:
        memory.close()

    report_lines = [f"# Nightly Report {run_date}", ""]
    if metrics:
        for name, value in metrics:
            report_lines.append(f"- {name}: {value:.4f}")
    else:
        report_lines.append("- No tasks executed.")

    report_path = report_dir / f"nightly_{run_date}.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
