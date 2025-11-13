# Baseline Run Plan: Vision SSL on CLEVRER

## Prerequisites

1. **Data**: CLEVRER training sequences ingested (889 valid sequences)
2. **Environment**: Python 3.11+, PyTorch 2.9.0, torchvision 0.24.0
3. **Hardware**: Apple Silicon (MPS) or CUDA-capable GPU
4. **Config**: `configs/baseline_vision_ssl.yaml` validated

## Pre-Flight Checks

### 1. Validate Configuration
```bash
python scripts/validate_baseline_config.py --config configs/baseline_vision_ssl.yaml
```

Expected output:
```
✅ Config configs/baseline_vision_ssl.yaml is valid and matches baseline schema
   Version: 1.0.0
   Run name: vision_ssl_clevrer_baseline
```

### 2. Verify Data
```bash
# Check sequence count
find data/sequences/clevrer_train -type d -mindepth 1 | wc -l
# Should be 889

# Check manifest
wc -l data/clevrer_manifest_train.txt
# Should be 889
```

### 3. Check Disk Space
```bash
df -h .
# Ensure sufficient space for checkpoints (~80MB) and logs
```

## Execution

### Step 1: Start Training
```bash
cd /path/to/world-model-poc
source .venv/bin/activate
PYTHONUNBUFFERED=1 PYTHONPATH=. python sleep.py \
  --config configs/baseline_vision_ssl.yaml \
  --date clevrer_train \
  2>&1 | tee /tmp/training_baseline.log
```

### Step 2: Monitor Progress
```bash
# Watch training telemetry (every 100 steps)
tail -f /tmp/training_baseline.log | grep "\[vision\]"

# Check for warnings
tail -f /tmp/training_baseline.log | grep "Warning"

# Monitor gradient norms
tail -f /tmp/training_baseline.log | grep "grad_norm"
```

### Step 3: Verify Key Milestones
- **Step 1000**: Loss should be ~0.003-0.004
- **Step 2500**: LR cut should occur, loss ~0.0010-0.0013
- **Step 3300-3900**: Best performance window (loss ~0.0005-0.0009)
- **Step 5000**: Final loss ~0.0011

## Post-Training Validation

### 1. Check Training Summary
```bash
tail -20 /tmp/training_baseline.log | grep -A 10 "Training Summary"
```

Expected:
- Final loss: ~0.0011
- Best loss: ~0.0005-0.0009
- Outliers: ~8 (0.16%)

### 2. Verify Checkpoints
```bash
ls -lh checkpoints/daily/*clevrer_train*.pt
# Should see final checkpoint (~78MB)
```

### 3. Load and Inspect Checkpoint
```bash
python -c "
from src.trainers.checkpoint import load_checkpoint
ckpt = load_checkpoint('checkpoints/daily/clevrer_train_vision_ssl.pt')
print(f'Step: {ckpt.get(\"step\")}')
print(f'Loss: {ckpt.get(\"loss\")}')
print(f'Outliers: {ckpt.get(\"outlier_count\", 0)}')
"
```

## Evaluation

### 1. Run Validation Rollouts
```bash
python scripts/run_validation.py \
  --checkpoint checkpoints/daily/clevrer_train_vision_ssl.pt \
  --rollout-steps 20 \
  --config configs/baseline_vision_ssl.yaml
```

### 2. Measure Metrics
- **Val rollout MSE @20 steps**: Record value
- **Drift metric** (ΔPSNR/ΔSSIM t=1→20): Record Δ in dB
- **Train vs val correlation**: Check alignment

### 3. Qualitative Assessment
- **Reconstructions**: Check crispness
- **Rollout stability**: Verify no drift/collapse
- **EMA vs raw**: Compare sharpness at ≥10 steps

## Troubleshooting

### Issue: Non-finite loss
- **Check**: QA assertions should catch this
- **Action**: Review data quality, check for corrupted sequences

### Issue: High outlier rate (>1%)
- **Check**: Review tripwire logs
- **Action**: May indicate data issues or numerical instability

### Issue: Gradient explosions
- **Check**: Grad norms should stay <0.1
- **Action**: Verify clipping is working, check LR

### Issue: Loss not decreasing
- **Check**: Verify data loading, check batch statistics
- **Action**: Ensure valid_tokens > 0, inputs_std > 1e-6

## Reproducibility Verification

### Run 1 vs Run 2 Comparison
To verify reproducibility, run training twice and compare:

```bash
# Run 1
python sleep.py --config configs/baseline_vision_ssl.yaml --date clevrer_train 2>&1 | tee run1.log

# Run 2 (should produce identical results)
python sleep.py --config configs/baseline_vision_ssl.yaml --date clevrer_train 2>&1 | tee run2.log

# Compare key metrics
grep "step 1000" run1.log run2.log
grep "step 5000" run1.log run2.log
```

Expected: Loss values should match exactly (bitwise reproducibility).

## Success Criteria

✅ **Training Stability**
- No late-run blowups (loss stays <0.002 after step 3000)
- Outlier rate <0.5%
- Gradient norms stable (0.005-0.03)

✅ **Performance**
- Final loss <0.0015
- Best loss in 3.2k-4.9k range <0.0010

✅ **Reproducibility**
- Identical results across runs (same seed)
- Config validation passes
- All QA assertions pass

✅ **Validation**
- Rollout MSE reasonable for task
- No collapse-to-mean in rollouts
- EMA improves over raw weights

## Next Steps After Baseline

1. **Lock baseline**: Mark config as immutable
2. **Document results**: Update `TRAINING_FINAL_SUMMARY.md` with validation metrics
3. **Create variants**: For experiments, copy baseline and modify
4. **Monitor**: Track any deviations from baseline behavior

## References

- Baseline config: `configs/baseline_vision_ssl.yaml`
- Training report: `TRAINING_FINAL_SUMMARY.md`
- Config validator: `scripts/validate_baseline_config.py`
- Validation script: `scripts/run_validation.py`

