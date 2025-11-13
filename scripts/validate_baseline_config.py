"""Validate baseline config against schema to ensure reproducibility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


# Expected schema for baseline config
BASELINE_SCHEMA = {
    "version": str,
    "run_name": str,
    "seed": int,
    "device": str,
    "precision": str,
    "batch": {
        "micro": int,
        "grad_accum": int,
    },
    "data": {
        "drop_last": bool,
        "log_manifest": bool,
        "seed": int,
    },
    "optim": {
        "lr": float,
        "weight_decay": float,
        "weight_decay_norm_bias": float,
        "clip_norm": float,
    },
    "schedule": {
        "lr_cut_step": int,
        "lr_after_cut": float,
        "cosine_to_zero": bool,
        "warmup_steps": int,
    },
    "loss": {
        "type": str,
        "eps": float,
        "per_sample_mean": bool,
        "safe_token_filter": bool,
    },
    "outliers": {
        "enable_tripwire": bool,
        "sigma": float,
        "skip_on_trigger": bool,
    },
    "logging": {
        "log_every": int,
        "grad_every": int,
        "fields": list,
    },
    "eval": {
        "interval_steps": int,
        "rollout_steps": int,
        "seed": int,
        "save_best_by": str,
        "ema_tiebreak_window": int,
    },
    "checkpoints": {
        "final_path": str,
    },
}


def validate_config(cfg: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> list[str]:
    """Recursively validate config against schema."""
    errors = []
    
    for key, expected_type in schema.items():
        full_path = f"{path}.{key}" if path else key
        
        if key not in cfg:
            errors.append(f"Missing required key: {full_path}")
            continue
        
        value = cfg[key]
        
        if isinstance(expected_type, dict):
            # Nested dict - recurse
            if not isinstance(value, dict):
                errors.append(f"{full_path}: expected dict, got {type(value).__name__}")
            else:
                errors.extend(validate_config(value, expected_type, full_path))
        elif isinstance(expected_type, type):
            # Type check
            if not isinstance(value, expected_type):
                errors.append(f"{full_path}: expected {expected_type.__name__}, got {type(value).__name__}")
        elif isinstance(expected_type, list):
            # List of allowed types
            if not any(isinstance(value, t) for t in expected_type):
                errors.append(f"{full_path}: expected one of {[t.__name__ for t in expected_type]}, got {type(value).__name__}")
    
    # Check for extra keys
    if isinstance(cfg, dict) and isinstance(schema, dict):
        for key in cfg:
            if key not in schema:
                full_path = f"{path}.{key}" if path else key
                errors.append(f"Unexpected key: {full_path} (not in baseline schema)")
    
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate baseline config against schema")
    ap.add_argument("--config", default="configs/baseline_vision_ssl.yaml", help="Config file to validate")
    ap.add_argument("--strict", action="store_true", help="Reject configs with extra keys")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    errors = validate_config(cfg, BASELINE_SCHEMA)
    
    if errors:
        print(f"❌ Validation failed for {config_path}:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print(f"✅ Config {config_path} is valid and matches baseline schema")
        print(f"   Version: {cfg.get('version', 'unknown')}")
        print(f"   Run name: {cfg.get('run_name', 'unknown')}")
        sys.exit(0)


if __name__ == "__main__":
    main()

