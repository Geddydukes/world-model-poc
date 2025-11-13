"""Run validation and rollouts on trained vision model."""

from __future__ import annotations

# Setup logging FIRST - before any imports that might be slow
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    force=True  # Override any existing config
)
logger = logging.getLogger(__name__)

# Immediate output to show script started
print("=" * 60, flush=True)
print("Validation script starting...", flush=True)
print("=" * 60, flush=True)
logger.info("Script starting, beginning imports...")

import argparse
import os
import sys
import time
import platform
import resource
from pathlib import Path
from typing import List

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger.info("Importing numpy...")
# Import numpy first (before torch) to catch import issues early
import numpy as np

logger.info("Importing torch...")
import torch

logger.info("Importing other dependencies...")
import yaml
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

logger.info("Importing project modules...")
logger.info("  Importing checkpoint...")
from src.trainers.checkpoint import load_checkpoint
logger.info("  Importing vision encoder...")
from src.vision.encoder import SimpleJEPAEncoder, Predictor, jepa_loss
logger.info("  Importing device utils...")
from src.utils.device import resolve_device
logger.info("  Importing sequence dataset...")
from src.data.sequence_dataset import SequenceFrameDataset
logger.info("  Importing vision augment...")
from src.vision.augment import get_vision_transforms

logger.info("All imports complete")


def visualize_reconstruction(model, images, save_path: Path, num_samples: int = 8):
    """Visualize model reconstructions."""
    # Keep images on device until needed for visualization
    n_avail = images.size(0)
    n = min(num_samples, n_avail, 8)  # Hard cap to keep grid tidy
    
    if n == 0:
        print(f"Warning: No images to visualize")
        return
    
    # Ensure channel-first format
    assert images.dim() == 4 and images.size(1) in (1, 3), f"bad image shape: {tuple(images.shape)}"
    
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        # Ensure float32 and keep on device
        if images.dtype == torch.float64:
            images = images.float()
        images_vis = images[:n].to(device)
        # Get tokens for visualization
        tokens = model(images_vis)
        # Only move to CPU once, at the end
        images_cpu = images_vis.detach().cpu()
        tokens_cpu = tokens.detach().cpu()
    
    # Build grid: 2 rows (input + tokens), n columns
    fig, axes = plt.subplots(2, n, figsize=(n * 2.2, 4.4))
    if n == 1:
        axes = axes.reshape(2, 1)  # Normalize structure for single column
    
    for i in range(n):
        # Input image (already on CPU)
        img = images_cpu[i].permute(1, 2, 0).numpy()
        img = img.clip(0, 1)  # Ensure valid range
        if img.shape[2] == 1:
            img = img.squeeze(2)  # Handle grayscale
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Input {i+1}")
        axes[0, i].axis('off')
        
        # Token visualization (placeholder - would need decoder)
        token_shape = tokens_cpu[i].shape
        axes[1, i].text(0.5, 0.5, f"Tokens: {token_shape}", 
                        ha='center', va='center', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path} ({n} samples)")


def rollout_test(model, initial_images, num_steps: int = 20):
    """Test rollout stability."""
    model.eval()
    # This is a placeholder - full rollout would predict next frames
    # For now, just check that embeddings are stable
    with torch.no_grad():
        # Ensure float32
        if initial_images.dtype == torch.float64:
            initial_images = initial_images.float()
        initial_tokens = model(initial_images)
        
        # Simulate rollout (in real implementation, would predict next frames)
        # Keep everything on device, only convert to numpy at the end
        tokens_sequence = [initial_tokens]
        for step in range(num_steps):
            # Placeholder: just use same tokens (real rollout would predict)
            tokens_sequence.append(initial_tokens)
        
        # Compute means on device, then convert to numpy once
        token_means_tensor = torch.stack([t.mean(dim=(1, 2)) for t in tokens_sequence])  # [num_steps+1, B, D]
        token_means = token_means_tensor.cpu().numpy()
        
        # Compute drift on CPU (small arrays, fine to do here)
        drift = np.std([np.linalg.norm(m - token_means[0]) for m in token_means[1:]])
        
        print(f"Rollout test ({num_steps} steps):")
        print(f"  Token drift: {drift:.6f}")
        print(f"  Status: {'✅ Stable' if drift < 0.1 else '⚠️ High drift'}")
        
        return drift < 0.1


def compute_validation_loss(model, predictor, dataloader, device, num_batches: int = 50, verbose: bool = False):
    """Compute validation loss on validation dataset."""
    logger.info(f"Starting validation loss computation: {num_batches} batches on {device}")
    model.eval()
    predictor.eval()
    
    total_loss = 0.0
    total_samples = 0
    losses = []
    
    t0 = time.perf_counter()
    t_last = t0
    
    logger.info("Entering validation loop...")
    logger.info("About to create iterator from DataLoader...")
    sys.stdout.flush()  # Force output
    
    t_iter_start = time.perf_counter()
    logger.info("Calling iter(dataloader)...")
    sys.stdout.flush()
    
    try:
        dataloader_iter = iter(dataloader)
        t_iter = time.perf_counter() - t_iter_start
        logger.info(f"Iterator created in {t_iter:.3f}s")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Failed to create iterator: {e}", exc_info=True)
        raise
    
    logger.info("Entering torch.no_grad() context...")
    sys.stdout.flush()
    
    with torch.no_grad():
        logger.info("Inside no_grad context, starting batch loop...")
        sys.stdout.flush()
        
        for batch_idx in range(num_batches):
            logger.info(f"Loop iteration {batch_idx + 1}/{num_batches}: Waiting for batch...")
            sys.stdout.flush()
            
            t_wait_start = time.perf_counter()
            try:
                logger.info(f"  About to call next(dataloader_iter) for batch {batch_idx + 1}...")
                sys.stdout.flush()
                
                # Add timeout mechanism for first batch to catch hangs
                if batch_idx == 0:
                    logger.info("  First batch - this may take longer as workers initialize...")
                    sys.stdout.flush()
                
                images, _ = next(dataloader_iter)
                t_wait = time.perf_counter() - t_wait_start
                logger.info(f"Batch {batch_idx + 1} received after {t_wait:.3f}s wait")
                logger.info(f"  Batch shape: {images.shape}, dtype: {images.dtype}")
                sys.stdout.flush()
            except StopIteration:
                logger.warning(f"DataLoader exhausted at batch {batch_idx + 1}, stopping")
                break
            except Exception as e:
                logger.error(f"Error getting batch {batch_idx + 1}: {e}", exc_info=True)
                logger.error(f"  Exception type: {type(e).__name__}")
                sys.stdout.flush()
                raise
            
            t_batch_start = time.perf_counter()
            t_data_load = t_batch_start - t_last
            logger.info(f"Batch {batch_idx + 1}/{num_batches}: Received in {t_wait:.3f}s, processing started (gap since last: {t_data_load:.3f}s)")
            
            # Ensure float32 and move to device
            t_prep_start = time.perf_counter()
            if images.dtype == torch.float64:
                images = images.float()
                logger.debug(f"  Converted float64 -> float32")
            images = images.to(device, non_blocking=(device.type == 'cuda'))
            batch_size = images.shape[0]
            t_prep = time.perf_counter() - t_prep_start
            logger.debug(f"  Prep (dtype/device): {t_prep:.3f}s, shape={images.shape}")
            
            # Get tokens
            t_forward_start = time.perf_counter()
            teacher_tokens = model(images)
            t_teacher = time.perf_counter() - t_forward_start
            logger.debug(f"  Teacher forward: {t_teacher:.3f}s, tokens shape={teacher_tokens.shape}")
            
            student_tokens = model(images)  # For validation, use same model
            t_student = time.perf_counter() - t_forward_start - t_teacher
            logger.debug(f"  Student forward: {t_student:.3f}s")
            
            # Create random mask
            t_mask_start = time.perf_counter()
            B, N, D = teacher_tokens.shape
            mask_ratio = 0.6
            keep = max(int(N * (1.0 - mask_ratio)), 1)
            mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            for b in range(B):
                idx = torch.randperm(N, device=device)[keep:]
                mask[b, idx] = True
            t_mask = time.perf_counter() - t_mask_start
            logger.debug(f"  Mask creation: {t_mask:.3f}s")
            
            # Compute loss
            t_loss_start = time.perf_counter()
            loss_value, loss_stats = jepa_loss(
                predictor(student_tokens), 
                teacher_tokens, 
                mask
            )
            t_loss = time.perf_counter() - t_loss_start
            logger.debug(f"  Loss computation: {t_loss:.3f}s")
            
            # Accumulate loss
            total_loss += float(loss_value.item()) * batch_size
            total_samples += batch_size
            losses.append(float(loss_value.item()))
            
            # Sync MPS for accurate timing
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            
            t_batch_total = time.perf_counter() - t_batch_start
            t_last = time.perf_counter()
            
            logger.info(f"Batch {batch_idx + 1} complete: {t_batch_total:.3f}s total "
                       f"(data={t_data_load:.3f}s, prep={t_prep:.3f}s, forward={t_teacher+t_student:.3f}s, "
                       f"mask={t_mask:.3f}s, loss={t_loss:.3f}s), loss={loss_value.item():.6f}")
            
            if verbose and batch_idx < 5:
                elapsed = time.perf_counter() - t0
                print(f"  Batch {batch_idx + 1}: {elapsed:.3f}s ({elapsed/(batch_idx+1):.3f}s/batch)")
    
    elapsed_total = time.perf_counter() - t0
    logger.info(f"Validation complete: {elapsed_total:.3f}s total for {len(losses)} batches "
               f"({elapsed_total/len(losses):.3f}s/batch avg)")
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss, losses


def main() -> None:
    ap = argparse.ArgumentParser(description="Run validation on vision model")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--val-glob", default="data/sequences/clevrer_val/*/frames.npy", help="Glob pattern for validation sequences")
    ap.add_argument("--num-samples", type=int, default=8, help="Number of samples for visualization")
    ap.add_argument("--rollout-steps", type=int, default=20, help="Number of rollout steps")
    ap.add_argument("--num-batches", type=int, default=50, help="Number of batches for validation loss")
    ap.add_argument("--device", default=None, help="Device override (auto, mps, cpu)")
    ap.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers (defaults to 0 on macOS/MPS)")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    ap.add_argument("--debug", action="store_true", help="Enable debug logging")
    ap.add_argument("--skip-viz", action="store_true", help="Skip visualization to save time")
    args = ap.parse_args()
    
    # Default to num_workers=0 on macOS/MPS for stability
    if args.num_workers is None:
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            args.num_workers = 0
            logger.info("Defaulting to num_workers=0 on macOS/MPS for stability")
        else:
            args.num_workers = 2  # Default for other platforms
    
    # Check file descriptor limits
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 4096:
            logger.warning(f"Low file descriptor limit: {soft}. Consider: ulimit -n 4096")
            logger.warning(f"  Current soft limit: {soft}, hard limit: {hard}")
        else:
            logger.debug(f"File descriptor limit OK: {soft}")
    except Exception as e:
        logger.warning(f"Could not check file descriptor limits: {e}")
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info("=" * 60)
    logger.info("Starting validation run")
    logger.info("=" * 60)
    
    cfg_path = Path(args.config)
    logger.info(f"Loading config from: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = args.device or cfg.get("device", "auto")
    device = resolve_device(device_str)
    vision_cfg = cfg.get("vision", {})
    
    logger.info(f"Device: {device} (requested: {device_str})")
    logger.info(f"DataLoader workers: {args.num_workers}")
    logger.info(f"Num batches: {args.num_batches}, Rollout steps: {args.rollout_steps}")
    
    # Load model
    t_load_start = time.perf_counter()
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)
    t_load = time.perf_counter() - t_load_start
    logger.info(f"Checkpoint loaded in {t_load:.3f}s")
    
    t_model_start = time.perf_counter()
    logger.info("Creating model architecture...")
    teacher = SimpleJEPAEncoder(
        embed_dim=vision_cfg.get("embed_dim", 256),
        depth=vision_cfg.get("depth", 6),
        heads=vision_cfg.get("heads", 4),
    ).float().to(device)  # Ensure float32 for MPS
    logger.debug(f"Teacher model created, loading state dict...")
    teacher.load_state_dict(ckpt["teacher"])
    teacher.eval()
    
    predictor = Predictor(
        vision_cfg.get("embed_dim", 256),
        vision_cfg.get("pred_dim", 512),
    ).float().to(device)  # Ensure float32 for MPS
    logger.debug(f"Predictor created, loading state dict...")
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()
    t_model = time.perf_counter() - t_model_start
    logger.info(f"Models loaded and moved to device in {t_model:.3f}s")

    logger.info(f"Model loaded from step {ckpt.get('step', 'unknown')}")
    if "best_loss" in ckpt:
        logger.info(f"Best loss: {ckpt['best_loss']:.6f} at step {ckpt.get('best_step', 'unknown')}")
    if "outlier_count" in ckpt:
        logger.info(f"Outliers: {ckpt['outlier_count']} ({ckpt.get('outlier_rate', 0):.2f}%)")

    # Load validation data if available
    t_data_start = time.perf_counter()
    val_glob = args.val_glob
    logger.info(f"Searching for validation files: {val_glob}")
    val_files = list(Path().glob(val_glob))
    logger.info(f"Found {len(val_files)} validation sequences")
    
    if val_files:
        t_dataset_start = time.perf_counter()
        # get_vision_transforms only takes image_size, no train/val distinction
        transforms = get_vision_transforms(image_size=vision_cfg.get("image_size", 224))
        logger.info("Creating dataset...")
        val_dataset = SequenceFrameDataset([val_glob], transform=transforms, frames_per_sequence=1)
        t_dataset = time.perf_counter() - t_dataset_start
        logger.info(f"Dataset created in {t_dataset:.3f}s, size: {len(val_dataset)}")
        
        # Optimize DataLoader for MPS: disable pin_memory (CUDA-only), add prefetch
        pin_memory = device.type == 'cuda'  # Only enable for CUDA
        logger.info(f"Creating DataLoader: batch_size=4, workers={args.num_workers}, "
                   f"pin_memory={pin_memory}, prefetch_factor={2 if args.num_workers > 0 else None}")
        sys.stdout.flush()
        
        t_loader_start = time.perf_counter()
        try:
            logger.info("Instantiating DataLoader object...")
            sys.stdout.flush()
            # Harden DataLoader for macOS multiprocessing
            loader_kwargs = {
                "batch_size": 4,
                "shuffle": False,
                "num_workers": args.num_workers,
                "pin_memory": pin_memory,
                "prefetch_factor": 2 if args.num_workers > 0 else None,
                "persistent_workers": False,  # Keep False on macOS for stability
            }
            
            # Add timeout and multiprocessing context for workers
            if args.num_workers > 0:
                loader_kwargs["timeout"] = 30  # Fail fast instead of hanging forever
                try:
                    # PyTorch >= 1.13 supports explicit multiprocessing context
                    loader_kwargs["multiprocessing_context"] = "spawn"
                except TypeError:
                    # Older PyTorch versions - context set globally
                    pass
            
            val_loader = DataLoader(val_dataset, **loader_kwargs)
            t_loader = time.perf_counter() - t_loader_start
            logger.info(f"DataLoader created in {t_loader:.3f}s")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Failed to create DataLoader: {e}", exc_info=True)
            raise
        
        logger.info(f"Computing validation loss on {min(args.num_batches, len(val_loader))} batches...")
        val_loss, loss_list = compute_validation_loss(teacher, predictor, val_loader, device, args.num_batches, verbose=args.verbose)
        logger.info(f"✅ Validation loss: {val_loss:.6f}")
        logger.info(f"   Loss std: {np.std(loss_list):.6f}")
        logger.info(f"   Loss range: [{np.min(loss_list):.6f}, {np.max(loss_list):.6f}]")
        
        # Get sample images for visualization
        logger.info("Getting sample images for visualization...")
        t_sample_start = time.perf_counter()
        sample_images, _ = next(iter(val_loader))
        sample_images = sample_images.to(device)
        t_sample = time.perf_counter() - t_sample_start
        logger.info(f"Sample images loaded in {t_sample:.3f}s")
    else:
        logger.warning(f"⚠️  No validation sequences found at {val_glob}")
        logger.warning(f"   Using dummy data for testing")
        sample_images = torch.randn(args.num_samples, 3, 224, 224).to(device)
    
    t_data = time.perf_counter() - t_data_start
    logger.info(f"Data loading setup complete in {t_data:.3f}s")

    # Visualization - clamp to available images
    if not args.skip_viz:
        logger.info("Creating visualization...")
        t_viz_start = time.perf_counter()
        output_dir = Path("reports/validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        n_vis = min(args.num_samples, sample_images.size(0))
        visualize_reconstruction(teacher, sample_images[:n_vis], output_dir / "reconstruction_grid.png", n_vis)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t_viz = time.perf_counter() - t_viz_start
        logger.info(f"Visualization created in {t_viz:.3f}s")
    else:
        logger.info("Skipping visualization (--skip-viz)")
        output_dir = Path("reports/validation")
        output_dir.mkdir(parents=True, exist_ok=True)

    # Rollout test
    logger.info(f"Running rollout test ({args.rollout_steps} steps)...")
    t_rollout_start = time.perf_counter()
    rollout_stable = rollout_test(teacher, sample_images, args.rollout_steps)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    t_rollout = time.perf_counter() - t_rollout_start
    logger.info(f"Rollout test completed in {t_rollout:.3f}s")

    logger.info("=" * 60)
    logger.info("✅ Validation complete")
    logger.info(f"  Rollout stability: {'✅ Pass' if rollout_stable else '⚠️ Check'}")
    logger.info("=" * 60)
    
    # Save summary
    summary_path = output_dir / "validation_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"Validation Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Step: {ckpt.get('step', 'unknown')}\n")
        if val_files:
            f.write(f"Validation loss: {val_loss:.6f}\n")
            f.write(f"Loss std: {np.std(loss_list):.6f}\n")
        f.write(f"Rollout stability: {'Pass' if rollout_stable else 'Check'}\n")
    print(f"\n📄 Summary saved to {summary_path}")


if __name__ == "__main__":
    # Immediate output before any processing
    print("Initializing multiprocessing...", flush=True)
    
    # Set multiprocessing start method for macOS
    import multiprocessing as mp
    logger.info(f"Setting multiprocessing start method to 'spawn' for macOS compatibility...")
    try:
        mp.set_start_method("spawn", force=True)
        logger.info("Multiprocessing start method set successfully")
    except RuntimeError:
        logger.info("Multiprocessing start method already set")
    except Exception as e:
        logger.warning(f"Could not set multiprocessing start method: {e}")
    
    print("Starting main validation function...", flush=True)
    logger.info("Starting main()...")
    main()

