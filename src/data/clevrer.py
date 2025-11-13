"""CLEVRER dataset loader for video and annotation data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from src.ingest.decode import decode_video, FrameStack

if TYPE_CHECKING:
    import torchvision.transforms as T


@dataclass
class CLEVRERAnnotation:
    """Container for CLEVRER video annotations."""
    
    video_id: str
    questions: List[Dict[str, Any]]
    answers: List[Dict[str, Any]]
    events: Optional[List[Dict[str, Any]]] = None
    objects: Optional[List[Dict[str, Any]]] = None


class CLEVRERVideoDataset(Dataset):
    """Dataset for CLEVRER videos with optional annotations."""
    
    def __init__(
        self,
        video_dir: str | Path,
        annotation_dir: Optional[str | Path] = None,
        split: str = "train",
        transform: Optional[Any] = None,  # T.Compose, but lazy import
        max_videos: Optional[int] = None,
        load_frames: bool = True,
        target_fps: Optional[float] = None,
    ):
        """
        Args:
            video_dir: Directory containing CLEVRER video files
            annotation_dir: Directory containing annotation JSON files (optional)
            split: Dataset split ('train', 'val', 'test')
            transform: Image transforms to apply
            max_videos: Maximum number of videos to load
            load_frames: If True, decode frames; if False, return video paths only
            target_fps: Target FPS for frame extraction
        """
        self.video_dir = Path(video_dir)
        self.annotation_dir = Path(annotation_dir) if annotation_dir else None
        self.split = split
        self.transform = transform
        self.load_frames = load_frames
        self.target_fps = target_fps
        
        # Find all video files
        video_extensions = {".mp4", ".avi", ".mov"}
        self.video_paths = sorted([
            p for p in self.video_dir.rglob("*")
            if p.suffix.lower() in video_extensions
        ])
        
        if max_videos and len(self.video_paths) > max_videos:
            import random
            random.seed(42)
            self.video_paths = random.sample(self.video_paths, max_videos)
        
        # Load annotations if available
        self.annotations: Dict[str, CLEVRERAnnotation] = {}
        if self.annotation_dir and self.annotation_dir.exists():
            self._load_annotations()
    
    def _load_annotations(self) -> None:
        """Load all annotation files from the annotation directory."""
        annotation_files = sorted(self.annotation_dir.glob("*.json"))
        
        for ann_file in annotation_files:
            try:
                with ann_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Handle different annotation formats
                if isinstance(data, list):
                    # List of annotations
                    for item in data:
                        video_id = item.get("video_id") or item.get("id") or Path(ann_file).stem
                        self.annotations[video_id] = CLEVRERAnnotation(
                            video_id=video_id,
                            questions=item.get("questions", []),
                            answers=item.get("answers", []),
                            events=item.get("events"),
                            objects=item.get("objects"),
                        )
                elif isinstance(data, dict):
                    # Single annotation or dict of annotations
                    if "video_id" in data or "id" in data:
                        video_id = data.get("video_id") or data.get("id") or Path(ann_file).stem
                        self.annotations[video_id] = CLEVRERAnnotation(
                            video_id=video_id,
                            questions=data.get("questions", []),
                            answers=data.get("answers", []),
                            events=data.get("events"),
                            objects=data.get("objects"),
                        )
                    else:
                        # Dict mapping video_id to annotation
                        for video_id, ann_data in data.items():
                            self.annotations[video_id] = CLEVRERAnnotation(
                                video_id=video_id,
                                questions=ann_data.get("questions", []),
                                answers=ann_data.get("answers", []),
                                events=ann_data.get("events"),
                                objects=ann_data.get("objects"),
                            )
            except Exception as e:
                print(f"Warning: Failed to load annotation {ann_file}: {e}")
                continue
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a video sample with optional annotations."""
        video_path = self.video_paths[idx]
        video_id = video_path.stem
        
        result: Dict[str, Any] = {
            "video_id": video_id,
            "video_path": str(video_path),
        }
        
        # Load frames if requested
        if self.load_frames:
            try:
                frame_stack = decode_video(
                    video_path,
                    target_fps=self.target_fps,
                )
                
                # Lazy import PIL and torchvision only when needed
                from PIL import Image
                import torchvision.transforms as T
                
                # Apply transforms to frames if provided
                frames = []
                for frame in frame_stack.frames:
                    img = Image.fromarray(frame)
                    if self.transform:
                        img = self.transform(img)
                    else:
                        # Default: convert to tensor
                        img = T.ToTensor()(img)
                    frames.append(img)
                
                result["frames"] = torch.stack(frames)  # [T, C, H, W]
                result["timestamps_ms"] = torch.from_numpy(frame_stack.timestamps_ms)
                result["fps"] = frame_stack.fps
            except Exception as e:
                raise RuntimeError(f"Failed to decode video {video_path}: {e}")
        
        # Add annotations if available
        if video_id in self.annotations:
            ann = self.annotations[video_id]
            result["questions"] = ann.questions
            result["answers"] = ann.answers
            result["events"] = ann.events
            result["objects"] = ann.objects
        
        return result


class CLEVRERFrameDataset(Dataset):
    """Dataset for pre-extracted CLEVRER frames."""
    
    def __init__(
        self,
        frame_dir: str | Path,
        annotation_dir: Optional[str | Path] = None,
        transform: Optional[Any] = None,  # T.Compose, but lazy import
        max_clips: Optional[int] = None,
    ):
        """
        Args:
            frame_dir: Directory containing frame directories (one per video)
            annotation_dir: Directory containing annotation JSON files (optional)
            transform: Image transforms to apply
            max_clips: Maximum number of clips to load
        """
        self.frame_dir = Path(frame_dir)
        self.annotation_dir = Path(annotation_dir) if annotation_dir else None
        self.transform = transform
        
        # Find all frame directories
        self.clip_dirs = sorted([
            d for d in self.frame_dir.iterdir()
            if d.is_dir() and any(d.glob("*.jpg")) or any(d.glob("*.png"))
        ])
        
        if max_clips and len(self.clip_dirs) > max_clips:
            import random
            random.seed(42)
            self.clip_dirs = random.sample(self.clip_dirs, max_clips)
        
        # Load annotations if available
        self.annotations: Dict[str, CLEVRERAnnotation] = {}
        if self.annotation_dir and self.annotation_dir.exists():
            self._load_annotations()
    
    def _load_annotations(self) -> None:
        """Load all annotation files from the annotation directory."""
        annotation_files = sorted(self.annotation_dir.glob("*.json"))
        
        for ann_file in annotation_files:
            try:
                with ann_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Handle different annotation formats (same as CLEVRERVideoDataset)
                if isinstance(data, list):
                    for item in data:
                        video_id = item.get("video_id") or item.get("id") or Path(ann_file).stem
                        self.annotations[video_id] = CLEVRERAnnotation(
                            video_id=video_id,
                            questions=item.get("questions", []),
                            answers=item.get("answers", []),
                            events=item.get("events"),
                            objects=item.get("objects"),
                        )
                elif isinstance(data, dict):
                    if "video_id" in data or "id" in data:
                        video_id = data.get("video_id") or data.get("id") or Path(ann_file).stem
                        self.annotations[video_id] = CLEVRERAnnotation(
                            video_id=video_id,
                            questions=data.get("questions", []),
                            answers=data.get("answers", []),
                            events=data.get("events"),
                            objects=data.get("objects"),
                        )
                    else:
                        for video_id, ann_data in data.items():
                            self.annotations[video_id] = CLEVRERAnnotation(
                                video_id=video_id,
                                questions=ann_data.get("questions", []),
                                answers=ann_data.get("answers", []),
                                events=ann_data.get("events"),
                                objects=ann_data.get("objects"),
                            )
            except Exception as e:
                print(f"Warning: Failed to load annotation {ann_file}: {e}")
                continue
    
    def __len__(self) -> int:
        return len(self.clip_dirs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a frame sequence sample."""
        clip_dir = self.clip_dirs[idx]
        clip_id = clip_dir.name
        
        # Load frames
        frame_files = sorted(clip_dir.glob("*.jpg")) + sorted(clip_dir.glob("*.png"))
        if not frame_files:
            raise ValueError(f"No frames found in {clip_dir}")
        
        # Lazy import PIL and torchvision only when needed
        from PIL import Image
        import torchvision.transforms as T
        
        frames = []
        for frame_file in frame_files:
            img = Image.open(frame_file).convert("RGB")
            if self.transform:
                img = self.transform(img)
            else:
                img = T.ToTensor()(img)
            frames.append(img)
        
        result: Dict[str, Any] = {
            "clip_id": clip_id,
            "frames": torch.stack(frames),  # [T, C, H, W]
            "frame_paths": [str(f) for f in frame_files],
        }
        
        # Add annotations if available
        if clip_id in self.annotations:
            ann = self.annotations[clip_id]
            result["questions"] = ann.questions
            result["answers"] = ann.answers
            result["events"] = ann.events
            result["objects"] = ann.objects
        
        return result

