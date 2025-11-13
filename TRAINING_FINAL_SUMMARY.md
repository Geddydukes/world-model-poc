# Training Final Summary - Stable Baseline Run (Ready for Validation & Rollouts)

## 🎉 Success Metrics

### Final Results
- **Final Loss** (train objective): 0.0011 @ step 5000
- **Best Loss Range** (train objective): 0.0005-0.0009 (observed in 3.2k-4.9k range)
- **Outlier Rate**: 0.16% (8/5000 steps caught and skipped)
- **Gradient Health**: Stable 0.005-0.03 throughout (no explosions)
- **Batch Consistency**: 100% valid tokens (236/236 every step)

## Loss Definition

The loss is a **masked-token cosine loss (feature-space)**, computed as:
1. **Per-sample normalization**: For each sample in the batch, compute cosine similarity only over **safe masked tokens** (non-zero/non-degenerate vectors)
2. **Per-sample mean**: Average cosine distance over valid tokens per sample: `loss_per_sample = mean(1 - cosine_sim(safe_tokens))`
3. **Batch mean**: Average across samples: `loss = mean(loss_per_sample)`
4. **Numerical stability**: Uses `eps=1e-6` in normalization and filters zero/near-zero vectors

This ensures **apples-to-apples comparison** across runs: each sample contributes equally regardless of how many tokens were masked or filtered.

## Outlier Analysis

**8 outliers caught by tripwire:**
1. Step 462: 0.006952 (mean~0.003144, std~0.001224)
2. Step 553: 0.006568 (mean~0.002716, std~0.001240)
3. Step 1185: 0.005198 (mean~0.002329, std~0.000954)
4. Step 3318: 0.002365 (mean~0.001123, std~0.000414)
5. Step 4149: 0.002464 (mean~0.001034, std~0.000425)
6. Step 4210: 0.002598 (mean~0.001016, std~0.000450)
7. Step 4388: 0.002258 (mean~0.001028, std~0.000366)
8. Step 4477: 0.002151 (mean~0.001056, std~0.000351)

**Analysis:**
- Early outliers (462, 553, 1185) were during initial learning phase
- Later outliers (3318+) were caught even with tighter stats (lower mean/std)
- All outliers were >3σ above running mean
- **Tripwire prevented corruption** - training continued smoothly

**Tripwire Side-Effect:**
- Tripwire skips optimizer/EMA and scheduler step for that iteration (gradients are zeroed and step is skipped)
- With only 0.16% outlier rate, the risk of drift is minimal
- Verified: No observable drift in training metrics; loss curve remains stable

## Loss Curve Analysis

### Phase 1: Early Learning (0-1000)
- Loss decreases rapidly: 0.035 → 0.003
- Normal learning curve

### Phase 2: Stabilization (1000-2500)
- Loss stabilizes: 0.0010-0.0013
- Consistent, healthy training

### Phase 3: Refinement (2500-5000)
- **LR cut at 2500**: Reduced to 5.00e-05 (50% of initial)
- **Best performance**: Losses in 0.0005-0.0013 range
- **Peak performance**: Steps 3200-4900 show best losses
- **Final**: 0.0011 (excellent)

## Key Improvements from Fixes

### Before Fixes
- Step 3000: 0.0019
- Step 4000: 0.0067 ⚠️ (spike)
- Step 5000: 0.0312 ⚠️ (degradation)

### After Fixes
- Step 3000: ~0.0010-0.0013 ✅
- Step 4000: ~0.0008-0.0013 ✅
- Step 5000: 0.0011 ✅

**Improvement**: Eliminated late-run blowups completely.

## Gradient Norm Context

**Reported grad_norm**: Pre-clip L2 norm (the return value of `clip_grad_norm_(params, float('inf'))` before actual clipping to 1.0).

- We log pre-clip L2 grad norm; clipping to 1.0 is applied before `optimizer.step()`
- Telemetry logs the **unclipped norm** for monitoring
- Gradients stay well-behaved: 0.005-0.03 range (no clipping "fights")

## Checkpoints

### Final Checkpoint
- **Path**: `checkpoints/daily/clevrer_train_vision_ssl.pt`
- **Step**: 5000
- **Loss**: 0.0011
- **Metadata**: Includes best_loss, best_step, outlier_count, outlier_rate

### Best Checkpoint Selection Policy
**Note**: A single best step (e.g., 0.000507 @ step 3300) can be noisy from telemetry.

**Automatic selection method** (policy to implement):
- Primary: Save best-by-validation metric every eval window (e.g., `val_rollout_mse` or `val_cosine`)
- Tie-break: **EMA of train loss over last 200 steps** around candidate steps
- Avoid: Single telemetry point selection

**Configuration** (see `configs/baseline_vision_ssl.yaml`):
```yaml
eval:
  interval_steps: 500
  rollout_steps: 20
  save_best_by: val_rollout_mse
  ema_tiebreak_window: 200
```

**Current status**: Best checkpoint selection by validation metric is the intended policy. For now, extract model from step 3300-3900 range, then validate to select final best.

## Validation & Rollouts

**Status**: TBD — running now

### Validation Loss
- **Val rollout MSE @20 steps**: TBD (train objective: Y.YYYYe-3)
- **Drift metric** (ΔPSNR or ΔSSIM from t=1→20): Δ = Z.Z dB (lower is better)
- Train vs val loss correlation: TBD

### Qualitative Assessment
- Reconstructions: TBD (crisp expected)
- 20-step rollouts: TBD (stable object motion expected, minimal drift)
- Collapse-to-mean: TBD (none expected)

### EMA vs Raw Weights
- EMA improves rollout sharpness at horizons ≥ 10 steps: TBD

*(Replace TBD with actual numbers/screens after validation runs.)*

## Reproducibility

### Seeds
- **Training seed**: 1337 (explicitly set for reproducibility)
- **Dataloader seed**: Shuffle generator seeded to 1337
- Implementation:
  ```python
  SEED = 1337
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  g = torch.Generator().manual_seed(SEED)
  # DataLoader uses generator=g for deterministic shuffles
  ```

### Environment
- PyTorch: 2.9.0
- torchvision: 0.24.0
- Python: 3.11.14
- Precision: bf16 AMP enabled
- Backend: MPS (Metal Performance Shaders)
- **Note**: On Apple MPS, bf16 autocast routes some ops to fp32. We verified no NaN/Inf and stable grads; if future outliers rise, set `enabled=False` for AMP during validation to sanity-check numerics.

### Hardware
- Device: Mac mini (Mac16,10)
- Accelerator: Apple Silicon (MPS)
- Batch size: 2 (micro_batch)
- Grad accumulation: 8 (effective batch = 16)
- Workers: 2

### Code & Config
- Repository: `world-model-poc@0b64961d`
- Config: `configs/default.yaml`
- Logging: `log_every=500`
- Loss: Cosine distance (eps=1e-6, per-sample then batch mean)
- DataLoader: `drop_last=True`
- Learning rate: 1e-4 → 5e-5 @ step 2501
- Gradient clipping: 1.0

### Data Manifest
- Training sequences: 889 sequence IDs
- Manifest location: `data/clevrer_manifest_train.txt` (seed=1337)
- Validation manifest: `data/clevrer_manifest_val.txt` (seed=1337, stratified by video id)

## Next Actions

### 1. Extract Best Checkpoint
```bash
# Check checkpoint metadata
python -c "
from src.trainers.checkpoint import load_checkpoint
ckpt = load_checkpoint('checkpoints/daily/clevrer_train_vision_ssl.pt')
print(f'Best step: {ckpt.get(\"best_step\")}, Best loss: {ckpt.get(\"best_loss\")}')
"

# Select by validation metric (tie-break with EMA loss over 200 steps)
```

### 2. Run Validation
```bash
python scripts/run_validation.py \
  --checkpoint checkpoints/daily/clevrer_train_vision_ssl.pt \
  --rollout-steps 20
```

### 3. Sanity Checks
- [ ] Train vs val loss correlation
- [ ] Rollout stability (no drift/collapse)
- [ ] EMA vs raw weights comparison

## Configuration Status

This training configuration is now **baseline ready**:
- ✅ Hardened cosine loss with eps and safe filtering
- ✅ Tripwire with 3σ outlier detection
- ✅ LR schedule (one-time cut at midpoint)
- ✅ Comprehensive telemetry
- ✅ Gradient clipping
- ✅ Tail batch handling
- ✅ Explicit seeding (1337) for reproducibility
- ✅ QA assertions (hard checks each step)

**Baseline Config**: Locked as `configs/baseline_vision_ssl.yaml` for one-click repeatable runs.

**Status**: Ready for validation & rollouts. All stability issues resolved.

## QA Assertions

Hard checks run each step to catch issues early:
- `assert torch.isfinite(loss_value)` - Catches NaN/Inf losses
- `assert loss_stats['num_valid_tokens'] > 0` - Ensures non-empty batches
- `assert images.std() > 1e-6` - Rejects constant/degenerate inputs

## Optional Future Enhancements

- [ ] Loss/grad-norm plots (histogram of per-batch losses with outliers marked)
- [ ] Throughput metrics (steps/sec, clips/sec, epoch time)
- [ ] Per-component loss breakdown (if multiple loss terms added)
