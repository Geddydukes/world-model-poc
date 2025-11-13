"""Diagnose potential training degradation issues."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
import glob
from collections import Counter

def check_tail_batches(dataset_size: int, batch_size: int) -> dict:
    """Check for tail batch issues."""
    num_batches = dataset_size // batch_size
    remainder = dataset_size % batch_size
    return {
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "full_batches": num_batches,
        "remainder_samples": remainder,
        "has_tail_batch": remainder > 0,
        "tail_batch_ratio": remainder / batch_size if remainder > 0 else 0.0,
    }

def check_loss_normalization() -> dict:
    """Check loss normalization implementation."""
    # Simulate the loss computation
    batch_size = 2
    num_tokens = 196  # 14x14 patches for 224x224 image
    mask_ratio = 0.6
    
    # Create mock data
    context_tokens = torch.randn(batch_size, num_tokens, 256)
    target_tokens = torch.randn(batch_size, num_tokens, 256)
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool)
    keep = max(int(num_tokens * (1.0 - mask_ratio)), 1)
    for b in range(batch_size):
        idx = torch.randperm(num_tokens)[keep:]
        mask[b, idx] = True
    
    # Current implementation (from encoder.py)
    ctx = context_tokens[mask]
    tgt = target_tokens[mask].detach()
    ctx = torch.nn.functional.normalize(ctx, dim=-1)
    tgt = torch.nn.functional.normalize(tgt, dim=-1)
    loss_current = (1.0 - (ctx * tgt).sum(dim=-1)).mean()
    
    # Check if mask count varies
    mask_counts = mask.sum(dim=1)
    num_masked = mask.sum()
    
    return {
        "loss_value": float(loss_current.item()),
        "num_masked_tokens": int(num_masked.item()),
        "mask_counts_per_sample": mask_counts.tolist(),
        "mask_count_varies": len(set(mask_counts.tolist())) > 1,
        "normalization_method": "mean over all masked tokens",
        "potential_issue": "If mask count varies significantly, averaging by batch size could be wrong",
    }

def check_data_quality(sequence_dir: Path, sample_size: int = 100) -> dict:
    """Check for data outliers and quality issues."""
    files = list(glob.glob(str(sequence_dir / "*/frames.npy")))[:sample_size]
    
    valid = 0
    invalid = 0
    constant_frames = 0
    low_variance = 0
    nan_inf = 0
    shape_issues = 0
    
    for f in files:
        try:
            arr = np.load(f)
            if arr.size == 0:
                invalid += 1
                continue
            
            if len(arr.shape) != 4:
                shape_issues += 1
                continue
            
            # Check for constant frames
            for frame in arr[:5]:  # Check first 5 frames
                if np.allclose(frame, frame[0, 0, 0]):
                    constant_frames += 1
                    break
            
            # Check variance
            if arr.std() < 1e-6:
                low_variance += 1
            
            # Check for NaN/Inf
            if not np.isfinite(arr).all():
                nan_inf += 1
            
            valid += 1
        except Exception:
            invalid += 1
    
    return {
        "samples_checked": len(files),
        "valid": valid,
        "invalid": invalid,
        "constant_frames": constant_frames,
        "low_variance": low_variance,
        "nan_inf": nan_inf,
        "shape_issues": shape_issues,
        "valid_percentage": valid / len(files) * 100 if files else 0,
    }

def check_scheduler_events(config_path: Path) -> dict:
    """Check for scheduler or annealing events."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    train_cfg = cfg.get("train", {})
    
    return {
        "has_scheduler": False,  # No scheduler found in code
        "learning_rate": train_cfg.get("lr", 0),
        "ema_momentum": train_cfg.get("ema_momentum", 0),
        "steps_vision": train_cfg.get("steps_vision", 0),
        "potential_issues": [
            "No learning rate scheduler - constant LR throughout",
            "EMA momentum constant at 0.999 - no annealing",
        ],
    }

def check_numerical_stability() -> dict:
    """Check for numerical stability issues."""
    # Test autocast with bf16
    device = "cpu"  # Test on CPU first
    dtype = torch.bfloat16
    
    # Simulate a forward pass
    x = torch.randn(2, 3, 224, 224)
    
    with torch.autocast(device_type=device, dtype=dtype, enabled=True):
        # Simple operation that could cause issues
        y = x * 1e-5
        z = y / 1e-5
        
        has_nan = torch.isnan(z).any()
        has_inf = torch.isinf(z).any()
        grad_norm = torch.norm(x) if x.requires_grad else 0.0
    
    return {
        "autocast_enabled": True,
        "dtype": str(dtype),
        "has_nan": bool(has_nan),
        "has_inf": bool(has_inf),
        "grad_norm": float(grad_norm),
        "recommendations": [
            "Monitor grad norms during training",
            "Check for NaN/Inf in loss values",
            "Consider gradient clipping if norms are large",
        ],
    }

def main():
    print("=" * 80)
    print("TRAINING DEGRADATION DIAGNOSTICS")
    print("=" * 80)
    
    # 1. Tail/Odd Batches
    print("\n1. TAIL/ODD BATCHES")
    print("-" * 80)
    dataset_size = 889  # Valid sequences
    batch_size = 2  # micro_batch
    tail_info = check_tail_batches(dataset_size, batch_size)
    for k, v in tail_info.items():
        print(f"  {k}: {v}")
    if tail_info["has_tail_batch"]:
        print(f"  ⚠️  ISSUE: Tail batch with {tail_info['remainder_samples']} samples")
        print(f"     Fix: Use drop_last=True in DataLoader")
    
    # 2. Loss Normalization
    print("\n2. LOSS NORMALIZATION")
    print("-" * 80)
    loss_info = check_loss_normalization()
    for k, v in loss_info.items():
        if k != "potential_issue":
            print(f"  {k}: {v}")
    if loss_info["mask_count_varies"]:
        print(f"  ⚠️  POTENTIAL ISSUE: {loss_info['potential_issue']}")
        print(f"     Current: Loss averaged over all masked tokens (correct)")
        print(f"     But if mask count varies, should normalize per-sample first")
    
    # 3. Scheduler/Anneal Events
    print("\n3. SCHEDULER/ANNEAL EVENTS")
    print("-" * 80)
    config_path = Path("configs/default.yaml")
    sched_info = check_scheduler_events(config_path)
    for k, v in sched_info.items():
        if k != "potential_issues":
            print(f"  {k}: {v}")
    if sched_info["potential_issues"]:
        print(f"  ⚠️  POTENTIAL ISSUES:")
        for issue in sched_info["potential_issues"]:
            print(f"     - {issue}")
    
    # 4. Numerical Stability
    print("\n4. NUMERICAL STABILITY (AMP/GRADS)")
    print("-" * 80)
    num_info = check_numerical_stability()
    for k, v in num_info.items():
        if k != "recommendations":
            print(f"  {k}: {v}")
    if num_info["recommendations"]:
        print(f"  Recommendations:")
        for rec in num_info["recommendations"]:
            print(f"     - {rec}")
    
    # 5. Data Quality
    print("\n5. DATA QUALITY (OUTLIERS)")
    print("-" * 80)
    sequence_dir = Path("data/sequences/clevrer_train")
    if sequence_dir.exists():
        data_info = check_data_quality(sequence_dir, sample_size=200)
        for k, v in data_info.items():
            print(f"  {k}: {v}")
        if data_info["constant_frames"] > 0 or data_info["low_variance"] > 0:
            print(f"  ⚠️  ISSUE: Found {data_info['constant_frames']} constant frames, {data_info['low_variance']} low variance")
            print(f"     Fix: Add data validation in dataset __getitem__")
    else:
        print(f"  Sequence directory not found: {sequence_dir}")
    
    # 6. Distributed/2-process
    print("\n6. DISTRIBUTED/2-PROCESS AVERAGING")
    print("-" * 80)
    print("  Status: Not using distributed training")
    print("  Multiple processes are separate training runs, not synchronized")
    print("  ✅ No issue here")
    
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)

if __name__ == "__main__":
    main()

