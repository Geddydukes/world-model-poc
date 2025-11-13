"""Refactored entry point for audio self-supervision."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader

from src.audio.cpc import CPCContext, CPCEncoder, CPCPredictor, LogMel, cpc_infonce_loss
from src.data.datasets import AudioGlobDataset
from src.memory.episodic import EpisodicMemory
from src.trainers.checkpoint import save_checkpoint
from src.utils.device import device_type_str
from src.utils.tensor import amp_dtype, autocast_enabled


class AudioSSLTrainer:
    """Modular audio CPC trainer with checkpointing and memory sync."""

    def __init__(self, cfg: Dict[str, Any], *, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.precision = amp_dtype()
        self.autocast = autocast_enabled(device_type_str(device))

    def _collect_globs(self, date: str) -> List[str]:
        data_cfg = self.cfg["data"]
        globs: List[str] = []
        today_glob = data_cfg.get("audio_glob_today", "")
        if today_glob:
            globs.append(today_glob.format(date=date))
        replay_glob = data_cfg.get("audio_glob_replay", "")
        if replay_glob:
            globs.append(replay_glob)
        return globs

    def _build_dataloader(self, globs: Iterable[str]) -> DataLoader | None:
        globs = [g for g in globs if g]
        if not globs:
            return None
        dataset = AudioGlobDataset(globs, sample_rate=self.cfg["audio"]["sample_rate"])
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

    def run(
        self,
        *,
        date: str,
        checkpoint_dir: Path,
        memory: EpisodicMemory,
    ) -> float:
        dataloader = self._build_dataloader(self._collect_globs(date))
        if dataloader is None:
            print(f"[audio] No audio files found for date {date}; skipping")
            return 0.0

        audio_cfg = self.cfg["audio"]
        train_cfg = self.cfg["train"]

        logmel = LogMel(
            audio_cfg["sample_rate"],
            audio_cfg["mel_bins"],
            audio_cfg["win_length"],
            audio_cfg["hop_length"],
        ).to(self.device)
        encoder = CPCEncoder(audio_cfg["mel_bins"], audio_cfg["cpc_hidden"]).to(self.device)
        context = CPCContext(audio_cfg["cpc_context"]).to(self.device)
        predictor = CPCPredictor(audio_cfg["cpc_context"], audio_cfg["cpc_pred_steps"]).to(self.device)

        params = list(encoder.parameters()) + list(context.parameters()) + list(predictor.parameters())
        optim = torch.optim.AdamW(params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
        grad_accum = max(1, int(train_cfg.get("grad_accum", 1)))
        log_every = max(1, int(train_cfg.get("log_every", 1000)))
        steps = int(train_cfg["steps_audio"])

        iterator = iter(dataloader)
        last_loss = 0.0
        optim.zero_grad(set_to_none=True)
        for step in range(steps):
            try:
                waveforms, paths = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                waveforms, paths = next(iterator)
            waveforms = waveforms.to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=device_type_str(self.device),
                dtype=self.precision,
                enabled=self.autocast,
            ):
                mel = logmel(waveforms)
                encoded = encoder(mel)
                context_vec = context(encoded)
                prediction = predictor(context_vec)
                loss_value = cpc_infonce_loss(context_vec, encoded, prediction)
                loss = loss_value / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0 or step == steps - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
            last_loss = float(loss_value.detach().cpu().item())

            if (step + 1) % log_every == 0:
                print(f"[audio] step {step + 1}/{steps} loss={last_loss:.4f}")
                with torch.no_grad():
                    pooled = context_vec.mean(dim=1).cpu().numpy()
                for i, path in enumerate(paths):
                    clip_id = Path(path).stem
                    memory.add_embedding(
                        clip_id=clip_id,
                        vec=pooled[i],
                        modality="audio",
                        model_tag=audio_cfg.get("model_tag", "audio_cpc"),
                        frame_idx_start=0,
                        frame_idx_end=0,
                        mean_pool=True,
                    )

        checkpoint = {
            "encoder": encoder.state_dict(),
            "context": context.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optim.state_dict(),
            "step": steps,
        }
        save_checkpoint(checkpoint, checkpoint_dir / f"{date}_audio_ssl.pt")
        return last_loss
