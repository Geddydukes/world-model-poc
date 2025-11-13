# Baseline Vision SSL Configuration

## Overview

This baseline configuration (`configs/baseline_vision_ssl.yaml`) represents a **locked, reproducible setup** for training vision SSL models on the CLEVRER dataset. All training stability issues have been resolved, and this config ensures bitwise reproducibility across runs.

## Key Features

### Reproducibility
- **Explicit seeding**: All random number generators seeded to 1337
- **Deterministic DataLoader**: Seeded generator ensures consistent shuffling
- **Versioned config**: Version 1.0.0 with schema validation
- **Complete environment tracking**: PyTorch versions, hardware, commit hash documented

### Stability Guarantees
- **Hardened cosine loss**: eps=1e-6, safe token filtering, per-sample normalization
- **Tripwire**: 3σ outlier detection with automatic step skipping
- **QA assertions**: Hard checks for NaN/Inf, empty batches, constant inputs
- **Gradient clipping**: Always clips to 1.0, monitors pre-clip norms

### Training Configuration
- **Effective batch size**: 16 (micro_batch=2, grad_accum=8)
- **Learning rate**: 1e-4 → 5e-5 @ step 2501
- **Weight decay**: 0.05 (excludes bias/LayerNorm)
- **Drop last batch**: Always enabled for consistent batch statistics

## Quick Start

### 1. Validate Config
```bash
python scripts/validate_baseline_config.py --config configs/baseline_vision_ssl.yaml
```

### 2. Run Training
```bash
python sleep.py --config configs/baseline_vision_ssl.yaml --date clevrer_train
```

### 3. Monitor Training
```bash
tail -f /tmp/training_hardened.log | grep "\[vision\]"
```

## Expected Results

- **Final loss** (train objective): ~0.0011 @ step 5000
- **Best loss range** (train objective): 0.0005-0.0009 (steps 3.2k-4.9k)
- **Outlier rate**: ~0.16% (8/5000 steps)
- **Gradient norms**: Stable 0.005-0.03 throughout

## Configuration Schema

The baseline config must match the expected schema. Use `scripts/validate_baseline_config.py` to verify.

### Required Sections
- `version`: Config version (e.g., "1.0.0")
- `run_name`: Descriptive run identifier
- `seed`: Global random seed
- `batch`: Batch size configuration
- `data`: Data loading configuration (includes seed for shuffling)
- `optim`: Optimizer settings (includes separate weight decay for norm/bias)
- `schedule`: Learning rate schedule
- `loss`: Loss function configuration
- `outliers`: Tripwire configuration
- `logging`: Telemetry configuration
- `eval`: Evaluation configuration (includes seed)
- `checkpoints`: Checkpoint paths

## Reproducibility Checklist

- [x] Explicit seeds set (1337)
- [x] Deterministic DataLoader shuffling
- [x] Environment versions documented
- [x] Hardware configuration documented
- [x] Git commit hash recorded
- [x] Data manifest with sequence IDs
- [x] Config schema validation
- [x] QA assertions enabled

## Validation

After training, run validation:
```bash
python scripts/run_validation.py \
  --checkpoint checkpoints/daily/clevrer_train_vision_ssl.pt \
  --rollout-steps 20
```

## Modifications

**⚠️ Warning**: Modifying the baseline config may break reproducibility. If you need to experiment:

1. Create a new config file (e.g., `configs/experiment_v1.yaml`)
2. Document all changes from baseline
3. Re-run validation to ensure stability

## See Also

- `TRAINING_FINAL_SUMMARY.md` - Complete training report
- `RunPlan.md` - Detailed execution and evaluation plan
- `configs/baseline_vision_ssl.yaml` - The baseline configuration

