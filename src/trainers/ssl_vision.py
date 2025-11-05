"""Refactored JEPA-style vision trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader

from src.data.datasets import ImageGlobDataset
from src.memory.episodic import EpisodicMemory
from src.trainers.checkpoint import save_checkpoint
from src.utils.device import device_type_str
from src.utils.tensor import amp_dtype, autocast_enabled
from src.vision.augment import get_vision_transforms
from src.vision.encoder import Predictor, SimpleJEPAEncoder, jepa_loss


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
        self.precision = amp_dtype(device)
        self.autocast = autocast_enabled(device)

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
        dataset = ImageGlobDataset(globs, transform=transforms)
        if len(dataset) == 0:
            return None
        micro_batch = self.cfg["train"]["micro_batch"]
        return DataLoader(
            dataset,
            batch_size=micro_batch,
            shuffle=True,
            num_workers=self.cfg["data"]["num_workers"],
            drop_last=len(dataset) >= micro_batch,
        )

    # --------------------------------------------------------------------- train
    def run(
        self,
        *,
        date: str,
        checkpoint_dir: Path,
        memory: EpisodicMemory,
    ) -> float:
        dataloader = self._build_dataloader(self._collect_globs(date))
        if dataloader is None:
            print(f"[vision] No frames found for date {date}; skipping")
            return 0.0

        vision_cfg = self.cfg["vision"]
        train_cfg = self.cfg["train"]

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

        params = list(student.parameters()) + list(predictor.parameters())
        optim = torch.optim.AdamW(params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
        grad_accum = max(1, int(train_cfg.get("grad_accum", 1)))
        log_every = max(1, int(train_cfg.get("log_every", 1000)))
        ema = float(train_cfg.get("ema_momentum", 0.999))
        steps = int(train_cfg["steps_vision"])

        iterator = iter(dataloader)
        last_loss = 0.0
        optim.zero_grad(set_to_none=True)
        for step in range(steps):
            try:
                images, paths = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                images, paths = next(iterator)
            images = images.to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=device_type_str(self.device),
                dtype=self.precision,
                enabled=self.autocast,
            ):
                student_tokens = student(images)
                teacher_tokens = teacher(images)
                mask = random_token_mask(
                    student_tokens.size(0),
                    student_tokens.size(1),
                    vision_cfg["mask_ratio"],
                    device=self.device,
                )
                loss_value = jepa_loss(predictor(student_tokens), teacher_tokens, mask)
                loss = loss_value / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0 or step == steps - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for ps, pt in zip(student.parameters(), teacher.parameters()):
                        pt.data.mul_(ema).add_(ps.data, alpha=1.0 - ema)

            last_loss = float(loss_value.detach().cpu().item())

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
        }
        save_checkpoint(checkpoint, checkpoint_dir / f"{date}_vision_ssl.pt")
        return last_loss
