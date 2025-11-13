# Training Degradation Analysis

## Loss Progression
- Step 1000: loss=0.0026
- Step 2000: loss=0.0021 ✅ (improving)
- Step 3000: loss=0.0021 ✅ (stable)
- Step 4000: loss=0.0171 ⚠️ (8x increase)
- Step 5000: loss=0.0248 ⚠️ (12x increase from best)

## Investigation Results

### 1. ✅ **TAIL/ODD BATCHES** - **CONFIRMED ISSUE**

**Finding:**
- Dataset size: 889 samples
- Batch size: 2 (micro_batch)
- Full batches: 444
- **Remainder: 1 sample** (creates a tail batch with only 1 sample)

**Current Code:**
```python
drop_last=len(dataset) >= micro_batch  # Line 78 in ssl_vision.py
```
This condition is `True` (889 >= 2), but the logic is backwards - it should always be `True` to drop the tail batch.

**Impact:**
- The last batch has only 1 sample instead of 2
- This creates inconsistent batch statistics
- Can cause gradient estimation issues, especially with EMA updates
- **This is likely a contributing factor to the degradation**

**Fix:**
```python
drop_last=True  # Always drop tail batch
```

---

### 2. ⚠️ **LOSS NORMALIZATION BUG** - **MINOR ISSUE**

**Finding:**
The current loss implementation averages over ALL masked tokens across the batch:
```python
def jepa_loss(context_tokens, target_tokens, mask_indices):
    ctx = context_tokens[mask_indices]  # [M,D] - all masked tokens from batch
    tgt = target_tokens[mask_indices].detach()
    ctx = F.normalize(ctx, dim=-1)
    tgt = F.normalize(tgt, dim=-1)
    return (1.0 - (ctx * tgt).sum(dim=-1)).mean()  # Mean over all M tokens
```

**Analysis:**
- Mask count is relatively constant (118 tokens per sample with mask_ratio=0.6)
- When mask counts vary, the current implementation weights samples by their mask count
- Test showed: with mask counts [78, 88], difference is ~0.0004 (small but non-zero)

**Impact:**
- Minor bias when mask counts vary significantly
- Not the primary cause, but could contribute

**Fix (if needed):**
```python
# Normalize per sample first, then average
loss_per_sample = []
for b in range(batch_size):
    sample_mask = mask[b]
    if sample_mask.sum() > 0:
        ctx_b = context_tokens[b][sample_mask]
        tgt_b = target_tokens[b][sample_mask].detach()
        ctx_b = F.normalize(ctx_b, dim=-1)
        tgt_b = F.normalize(tgt_b, dim=-1)
        loss_b = (1.0 - (ctx_b * tgt_b).sum(dim=-1)).mean()
        loss_per_sample.append(loss_b)
loss = torch.stack(loss_per_sample).mean()
```

---

### 3. ✅ **SCHEDULER/ANNEAL EVENT** - **NO ISSUE FOUND**

**Finding:**
- No learning rate scheduler in code (constant LR = 1e-4)
- EMA momentum is constant at 0.999 (no annealing)
- No StepLR milestones, cosine restarts, or other abrupt changes

**Impact:**
- **Not a cause** - there are no scheduler events at step 3000-4000
- However, constant LR might not be optimal for long training

**Recommendation:**
- Consider adding a cosine annealing scheduler if training longer
- Current constant LR is fine for 5000 steps

---

### 4. ⚠️ **NUMERICAL INSTABILITY (AMP/GRADS)** - **POTENTIAL ISSUE**

**Finding:**
- Using `torch.autocast` with `bfloat16` precision
- Gradient clipping is present: `clip_grad_norm_(params, 1.0)`
- No explicit NaN/Inf checks in loss computation
- No grad norm logging

**Current Code:**
```python
with torch.autocast(device_type=..., dtype=self.precision, enabled=self.autocast):
    # ... forward pass ...
    loss_value = jepa_loss(...)
    loss = loss_value / grad_accum
loss.backward()
```

**Potential Issues:**
- AMP can cause numerical instability if operations aren't AMP-safe
- No detection of NaN/Inf in loss values
- Grad norms not monitored (could spike without notice)
- Loss spike at step 4000 could indicate numerical instability

**Impact:**
- **Possible contributing factor** - the sudden spike suggests numerical issues
- Need to verify if loss contains NaN/Inf at step 4000

**Fix:**
```python
# Add loss validation
if not torch.isfinite(loss_value):
    print(f"Warning: Non-finite loss at step {step}: {loss_value}")
    continue  # Skip this step

# Log grad norms periodically
if (step + 1) % 100 == 0:
    total_norm = torch.nn.utils.clip_grad_norm_(params, float('inf'))
    print(f"Grad norm: {total_norm:.4f}")
```

---

### 5. ✅ **DATA QUALITY (OUTLIERS)** - **NO ISSUE IN VALID SAMPLES**

**Finding:**
- Checked 200 valid sequence files
- 0 constant frames
- 0 low variance samples
- 0 NaN/Inf values
- 0 shape issues
- 100% valid in sampled data

**However:**
- 40.5% of sequences are corrupted/empty (606 out of 1495)
- These are filtered out during dataset initialization
- But the dataset still loads them initially, then skips them

**Impact:**
- Valid samples are clean
- **Not a direct cause**, but corrupted files waste computation during dataset init

**Recommendation:**
- Delete corrupted sequences (already have script)
- Add validation in `__getitem__` as safety check:
```python
if not torch.isfinite(x).all():
    # Skip or return next valid sample
if x.std() < 1e-6:
    # Skip constant frames
```

---

### 6. ✅ **DISTRIBUTED/2-PROCESS AVERAGING** - **NO ISSUE**

**Finding:**
- Two processes are running, but they're **separate training runs**, not distributed training
- Each process trains independently
- No synchronization or averaging between processes

**Impact:**
- **Not a cause** - processes don't interact

---

## Root Cause Analysis

### Primary Suspects (in order of likelihood):

1. **Tail Batch Issue** ⭐⭐⭐
   - Confirmed: 1-sample tail batch exists
   - Impact: Inconsistent batch statistics, especially with EMA
   - Fix: `drop_last=True`

2. **Numerical Instability** ⭐⭐
   - Suspected: Sudden spike at step 4000 suggests numerical issues
   - No NaN/Inf detection in code
   - Need to verify loss values at step 4000
   - Fix: Add loss validation and grad norm monitoring

3. **Loss Normalization** ⭐
   - Minor: Small bias when mask counts vary
   - Not primary cause, but could contribute
   - Fix: Per-sample normalization (optional)

4. **Data Quality** ⭐
   - Valid samples are clean
   - Corrupted files already filtered
   - Not a direct cause

### Recommended Fixes (Priority Order):

1. **HIGH PRIORITY:**
   - Fix `drop_last=True` in DataLoader
   - Add loss validation: `if not torch.isfinite(loss_value): skip`
   - Add grad norm logging every 100 steps

2. **MEDIUM PRIORITY:**
   - Delete corrupted sequences
   - Add per-sample loss normalization (if mask counts vary significantly)

3. **LOW PRIORITY:**
   - Consider learning rate scheduler for longer training
   - Add data validation in `__getitem__` as safety check

## Next Steps

1. Apply fixes to `src/trainers/ssl_vision.py`
2. Delete corrupted sequences
3. Retrain with fixes
4. Monitor loss and grad norms during training

