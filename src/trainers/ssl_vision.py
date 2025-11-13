"""Refactored JEPA-style vision trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import random
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import ImageGlobDataset
from src.data.sequence_dataset import SequenceFrameDataset
from src.memory.episodic import EpisodicMemory
from src.trainers.checkpoint import save_checkpoint
from src.utils.device import device_type_str
from src.utils.tensor import amp_dtype, autocast_enabled
from src.vision.augment import get_vision_transforms
from src.vision.encoder import Predictor, SimpleJEPAEncoder, jepa_loss
from src.trainers.checkpoint import load_checkpoint

# Reproducibility: Seeds will be set from config
def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # True on CUDA can slow; MPS ignores


def random_token_mask(batch: int, tokens: int, ratio: float, *, device: torch.device) -> torch.Tensor:
    keep = max(int(tokens * (1.0 - ratio)), 1)
    mask = torch.zeros(batch, tokens, dtype=torch.bool, device=device)
    for b in range(batch):
        idx = torch.randperm(tokens, device=device)[keep:]
        mask[b, idx] = True
    return mask


class VisionSSLTrainer:
    """Modular vision self-supervised trainer."""

    def __init__(self, cfg: Dict[str, Any], *, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.precision = amp_dtype()
        self.autocast = autocast_enabled(device_type_str(device))
        # Loss statistics for tripwire
        self.loss_stats = None

        
        # Set seeds from config for reproducibility
        seed = cfg.get("seed") or cfg.get("data", {}).get("seed", 1337)
        set_seeds(seed)

    # ------------------------------------------------------------------ helpers
    def _collect_globs(self, date: str) -> List[str]:
        data_cfg = self.cfg["data"]
        globs: List[str] = []
        today_glob = data_cfg.get("frame_glob_today", "")
        if today_glob:
            globs.append(today_glob.format(date=date))
        replay_glob = data_cfg.get("frame_glob_replay", "")
        if replay_glob:
            globs.append(replay_glob)
        return globs

    def _build_dataloader(self, globs: Iterable[str]) -> DataLoader | None:
        globs = [g for g in globs if g]
        if not globs:
            return None
        transforms = get_vision_transforms(self.cfg["data"]["image_size"])
        
        # Check if we're using numpy sequence files or JPEG files
        import glob as glob_module
        sample_files = []
        for g in globs:
            sample_files.extend(glob_module.glob(g, recursive=True)[:1])
        
        if sample_files and sample_files[0].endswith('.npy'):
            # Use sequence dataset for numpy files
            dataset = SequenceFrameDataset(globs, transform=transforms, frames_per_sequence=1)
        else:
            # Use regular image glob dataset for JPEG files
            dataset = ImageGlobDataset(globs, transform=transforms)
        
        if len(dataset) == 0:
            return None
        # Support both old config format (train.micro_batch) and new format (batch.micro)
        micro_batch = self.cfg.get("batch", {}).get("micro") or self.cfg.get("train", {}).get("micro_batch", 2)
        # Deterministic shuffling with seeded generator from config
        data_seed = self.cfg.get("data", {}).get("seed", self.cfg.get("seed", 1337))
        g = torch.Generator(device='cpu').manual_seed(data_seed)
        drop_last = self.cfg.get("data", {}).get("drop_last", True)
        return DataLoader(
            dataset,
            batch_size=micro_batch,
            shuffle=True,
            generator=g,  # Seeded generator for deterministic shuffles
            num_workers=self.cfg["data"]["num_workers"],
            drop_last=drop_last,  # Always drop tail batch to avoid inconsistent batch statistics
        )

    # --------------------------------------------------------------------- train
    def run(
        self,
        *,
        date: str,
        checkpoint_dir: Path,
        memory: EpisodicMemory,
        resume_from: Path | None = None,
        start_step: int = 0,
    ) -> float:
        dataloader = self._build_dataloader(self._collect_globs(date))
        if dataloader is None:
            print(f"[vision] No frames found for date {date}; skipping")
            return 0.0

        vision_cfg = self.cfg.get("vision", {})
        train_cfg = self.cfg.get("train", {})

        student = SimpleJEPAEncoder(
            embed_dim=vision_cfg["embed_dim"],
            depth=vision_cfg["depth"],
            heads=vision_cfg["heads"],
        ).to(self.device)
        teacher = SimpleJEPAEncoder(
            embed_dim=vision_cfg["embed_dim"],
            depth=vision_cfg["depth"],
            heads=vision_cfg["heads"],
        ).to(self.device)
        teacher.load_state_dict(student.state_dict())
        for param in teacher.parameters():
            param.requires_grad_(False)
        predictor = Predictor(vision_cfg["embed_dim"], vision_cfg["pred_dim"]).to(self.device)

        # Separate parameter groups for weight decay (exclude bias and LayerNorm)
        optim_cfg = self.cfg.get("optim", {})
        weight_decay = optim_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.05))
        weight_decay_norm_bias = optim_cfg.get("weight_decay_norm_bias", 0.0)
        
        # Group parameters: regular params vs bias/LayerNorm
        decay_params = []
        no_decay_params = []
        for module in [student, predictor]:
            for name, param in module.named_parameters():
                if param.requires_grad:
                    if 'bias' in name or 'norm' in name.lower() or 'ln' in name.lower():
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)
        
        params = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": weight_decay_norm_bias},
        ]
        # Support both old config format and new format
        lr = optim_cfg.get("lr") or train_cfg.get("lr", 1e-4)
        optim = torch.optim.AdamW(params, lr=lr)
        grad_accum = self.cfg.get("batch", {}).get("grad_accum") or train_cfg.get("grad_accum", 8)
        grad_accum = max(1, int(grad_accum))
        log_every = self.cfg.get("logging", {}).get("log_every") or train_cfg.get("log_every", 1000)
        log_every = max(1, int(log_every))
        telemetry_every = self.cfg.get("logging", {}).get("grad_every", 100)
        ema = float(train_cfg.get("ema_momentum", 0.999))
        steps = int(train_cfg.get("steps_vision", 5000))
        initial_lr = lr
        
        # LR schedule from config
        schedule_cfg = self.cfg.get("schedule", {})
        lr_cut_step = schedule_cfg.get("lr_cut_step", steps // 2)
        lr_after_cut = schedule_cfg.get("lr_after_cut", initial_lr * 0.5)
        warmup_steps = schedule_cfg.get("warmup_steps", 0)
        lr_cut_applied = False
        
        # Load checkpoint if resuming
        if resume_from and resume_from.exists():
            print(f"[vision] Resuming from checkpoint: {resume_from}")
            ckpt = load_checkpoint(resume_from)
            student.load_state_dict(ckpt["student"])
            teacher.load_state_dict(ckpt["teacher"])
            predictor.load_state_dict(ckpt["predictor"])
            if "optimizer" in ckpt:
                optim.load_state_dict(ckpt["optimizer"])
            print(f"[vision] Loaded checkpoint from step {ckpt.get('step', 0)}")

        iterator = iter(dataloader)
        last_loss = 0.0
        best_loss = float('inf')
        best_step = 0
        outlier_count = 0
        optim.zero_grad(set_to_none=True)
        
        for step in range(start_step, steps):
            try:
                images, paths = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                images, paths = next(iterator)
            images = images.to(self.device, non_blocking=True)

            # LR warmup (if configured)
            if warmup_steps > 0 and step < warmup_steps:
                warmup_lr = initial_lr * (step + 1) / warmup_steps
                for g in optim.param_groups:
                    g["lr"] = warmup_lr
            
            # Lower LR for back half
            if step >= lr_cut_step and not lr_cut_applied:
                for g in optim.param_groups:
                    g["lr"] = lr_after_cut
                print(f"[vision] Reduced LR to {optim.param_groups[0]['lr']:.2e} at step {step + 1}")
                lr_cut_applied = True
            
            # Optional: Test AMP A/B (disabled by default, enable via config)
            use_amp = self.cfg.get("train", {}).get("use_amp", True)
            if step < 500 and not use_amp:
                # Test without AMP for first 500 steps
                amp_enabled = False
            else:
                amp_enabled = self.autocast
            
            with torch.autocast(
                device_type=device_type_str(self.device),
                dtype=self.precision,
                enabled=amp_enabled,
            ):
                student_tokens = student(images)
                teacher_tokens = teacher(images)
                mask = random_token_mask(
                    student_tokens.size(0),
                    student_tokens.size(1),
                    vision_cfg["mask_ratio"],
                    device=self.device,
                )
                loss_value, loss_stats = jepa_loss(predictor(student_tokens), teacher_tokens, mask)
                loss = loss_value / grad_accum
            
            # QA Assertions (hard checks each step)
            assert torch.isfinite(loss_value), f"Non-finite loss at step {step + 1}: {loss_value}"
            assert loss_stats['num_valid_tokens'] > 0, f"No valid tokens at step {step + 1}"
            assert images.std() > 1e-6, f"Constant input at step {step + 1}"
            
            # Tripwire: Check for non-finite loss (redundant with assert, but provides warning)
            if not torch.isfinite(loss_value):
                print(f"[vision] Warning: Non-finite loss at step {step + 1}: {loss_value}")
                optim.zero_grad(set_to_none=True)
                continue
            
            # Tripwire: Running stats and 3σ outlier detection
            if self.loss_stats is None:
                self.loss_stats = {"m": float(loss_value.detach().cpu().item()), "v": 0.0}
            else:
                alpha = 0.05
                loss_val = float(loss_value.detach().cpu().item())
                delta = loss_val - self.loss_stats["m"]
                self.loss_stats["m"] += alpha * delta
                self.loss_stats["v"] = (1 - alpha) * (self.loss_stats["v"] + alpha * delta * delta)
            
            # Check for outlier (3σ)
            loss_std = math.sqrt(max(self.loss_stats["v"], 1e-12))
            loss_threshold = self.loss_stats["m"] + 3.0 * loss_std
            loss_val = float(loss_value.detach().cpu().item())
            if loss_val > loss_threshold:
                outlier_count += 1
                print(f"[vision] Warning: Outlier loss at step {step + 1}: {loss_val:.6f} "
                      f"(mean~{self.loss_stats['m']:.6f}, std~{loss_std:.6f}) — skipping step")
                # Tripwire skips optimizer/EMA and scheduler step for this iteration
                optim.zero_grad(set_to_none=True)
                continue
            
            loss.backward()

            if (step + 1) % grad_accum == 0 or step == steps - 1:
                # Get grad norm before clipping for telemetry
                total_norm = torch.nn.utils.clip_grad_norm_(params, float('inf'))
                # Always clip gradients to 1.0
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for ps, pt in zip(student.parameters(), teacher.parameters()):
                        pt.data.mul_(ema).add_(ps.data, alpha=1.0 - ema)
            else:
                # For telemetry when not updating
                total_norm = torch.nn.utils.clip_grad_norm_(params, float('inf'))

            last_loss = float(loss_value.detach().cpu().item())
            
            # Track best loss and save checkpoint
            if last_loss < best_loss:
                best_loss = last_loss
                best_step = step + 1
                # Save best checkpoint if in the sweet spot (3.3k-3.9k) or if it's the best overall
                if (3300 <= best_step <= 3900) or (step + 1) % 1000 == 0:
                    best_ckpt_path = checkpoint_dir / f"{date}_vision_ssl_best_step{best_step}.pt"
                    best_checkpoint = {
                        "student": student.state_dict(),
                        "teacher": teacher.state_dict(),
                        "predictor": predictor.state_dict(),
                        "optimizer": optim.state_dict(),
                        "step": best_step,
                        "loss": best_loss,
                    }
                    save_checkpoint(best_checkpoint, best_ckpt_path)
                    print(f"[vision] Saved best checkpoint at step {best_step} with loss {best_loss:.6f}")
            
            # Richer telemetry every 100 steps
            if (step + 1) % telemetry_every == 0:
                current_lr = optim.param_groups[0]["lr"]
                inputs_std = images.std().item()
                print(f"[vision] step {step + 1}/{steps} | "
                      f"loss={last_loss:.6f} | "
                      f"grad_norm={total_norm:.4f} | "
                      f"lr={current_lr:.2e} | "
                      f"valid_tokens={loss_stats['num_valid_tokens']}/{loss_stats['num_total_tokens']} "
                      f"({100-loss_stats['pct_filtered']:.1f}%) | "
                      f"inputs_std={inputs_std:.4f}")

            if (step + 1) % log_every == 0:
                print(f"[vision] step {step + 1}/{steps} loss={last_loss:.4f}")
                teacher.eval()
                with torch.no_grad():
                    teacher_tokens = teacher(images).mean(dim=1).cpu().numpy()
                for i, path in enumerate(paths):
                    path_obj = Path(path)
                    clip_id = path_obj.parent.name or path_obj.stem
                    frame_idx = 0
                    stem = path_obj.stem
                    if "_" in stem:
                        try:
                            frame_idx = int(stem.split("_")[-1]) - 1
                        except ValueError:
                            frame_idx = 0
                    memory.add_embedding(
                        clip_id=clip_id,
                        vec=teacher_tokens[i],
                        modality="vision",
                        model_tag=vision_cfg.get("model_tag", "vision_jepa"),
                        frame_idx_start=frame_idx,
                        frame_idx_end=frame_idx,
                        mean_pool=True,
                    )
                teacher.train()

        checkpoint = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optim.state_dict(),
            "step": steps,
            "loss": last_loss,
            "best_loss": best_loss,
            "best_step": best_step,
            "outlier_count": outlier_count,
            "outlier_rate": outlier_count / steps * 100.0,
        }
        save_checkpoint(checkpoint, checkpoint_dir / f"{date}_vision_ssl.pt")
        
        # Print training summary
        print(f"\n[vision] Training Summary:")
        print(f"  Final loss: {last_loss:.6f}")
        print(f"  Best loss: {best_loss:.6f} at step {best_step}")
        print(f"  Outliers caught: {outlier_count} ({outlier_count/steps*100:.2f}% of steps)")
        print(f"  Checkpoint saved: {checkpoint_dir / f'{date}_vision_ssl.pt'}")
        if best_step != steps:
            print(f"  Best checkpoint: {checkpoint_dir / f'{date}_vision_ssl_best_step{best_step}.pt'}")
        
        return last_loss
