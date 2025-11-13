"""Dataset for loading frame sequences from numpy files."""

from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SequenceFrameDataset(Dataset):
    """Dataset that loads individual frames from sequence numpy files."""
    
    def __init__(
        self,
        globs: str | List[str],
        transform: Optional[T.Compose] = None,
        max_items: Optional[int] = None,
        frames_per_sequence: int = 1,
    ):
        """
        Args:
            globs: Glob pattern(s) to find frames.npy files
            transform: Image transforms to apply
            max_items: Maximum number of frames to load
            frames_per_sequence: How many frames to sample from each sequence
        """
        if isinstance(globs, str):
            globs = [globs]
        
        sequence_files = []
        for g in globs:
            sequence_files.extend(glob.glob(g, recursive=True))
        
        self.sequence_files = sorted(sequence_files)
        self.transform = transform
        self.frames_per_sequence = frames_per_sequence
        
        # Build index of (sequence_file, frame_idx) pairs
        self.frame_index: List[tuple[Path, int]] = []
        for seq_file in self.sequence_files:
            try:
                # Just check file shape, don't load or mmap here (multiprocessing issue)
                # We'll use mmap in __getitem__ per-worker
                temp = np.load(seq_file, mmap_mode='r')
                num_frames = temp.shape[0]
                del temp  # Close the mmap
                # Sample frames_per_sequence frames from this sequence
                if frames_per_sequence >= num_frames:
                    indices = list(range(num_frames))
                else:
                    indices = sorted(random.sample(range(num_frames), frames_per_sequence))
                for idx in indices:
                    self.frame_index.append((Path(seq_file), idx))
            except Exception as e:
                print(f"Warning: Failed to load {seq_file}: {e}")
                continue
        
        if max_items and len(self.frame_index) > max_items:
            self.frame_index = random.sample(self.frame_index, max_items)
    
    def __len__(self) -> int:
        return len(self.frame_index)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """Get a frame from a sequence."""
        seq_file, frame_idx = self.frame_index[idx]
        
        # For multiprocessing workers, mmap can cause hangs on macOS
        # Use regular load instead - it's slower but more reliable
        # Check if we're in a worker process (multiprocessing context)
        import multiprocessing as mp
        import logging
        logger = logging.getLogger(__name__)
        
        process_name = mp.current_process().name
        use_mmap = process_name == 'MainProcess'
        
        if idx < 3:  # Log first few items for debugging
            logger.info(f"Dataset __getitem__[{idx}]: process={process_name}, use_mmap={use_mmap}, file={seq_file.name}")
        
        try:
            if use_mmap:
                # Main process: use mmap for speed
                frames_mmap = np.load(seq_file, mmap_mode='r')
                frame = np.array(frames_mmap[frame_idx])  # Shape: [H, W, 3] - copy to array for processing
                del frames_mmap  # Release mmap reference
            else:
                # Worker process: use regular load to avoid hangs
                frames = np.load(seq_file)
                frame = frames[frame_idx]
        except Exception as e:
            logger.error(f"Error loading {seq_file}[{frame_idx}]: {e}", exc_info=True)
            raise
        
        # Validate frame data
        if not np.isfinite(frame).all():
            # Return next valid frame if this one has NaN/Inf
            if idx + 1 < len(self.frame_index):
                return self.__getitem__(idx + 1)
            # If last item, return zeros (shouldn't happen with drop_last=True)
            frame = np.zeros_like(frame)
        
        # Check for constant/low variance frames
        if frame.std() < 1e-6:
            # Return next valid frame if this one is constant
            if idx + 1 < len(self.frame_index):
                return self.__getitem__(idx + 1)
            frame = np.zeros_like(frame)
        
        # Convert to PIL Image
        img = Image.fromarray(frame.astype(np.uint8))
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            # Default: resize and convert to tensor
            img = T.Compose([
                T.Resize(224, antialias=True),
                T.CenterCrop(224),
                T.ToTensor(),
            ])(img)
        
        # Ensure float32 (not float64) for MPS compatibility
        if img.dtype == torch.float64:
            img = img.float()
        
        return img, str(seq_file)


class SequenceStackDataset(Dataset):
    """Dataset that loads entire frame stacks from sequence numpy files."""
    
    def __init__(
        self,
        globs: str | List[str],
        transform: Optional[T.Compose] = None,
        max_items: Optional[int] = None,
    ):
        """
        Args:
            globs: Glob pattern(s) to find frames.npy files
            transform: Image transforms to apply to each frame
            max_items: Maximum number of sequences to load
        """
        if isinstance(globs, str):
            globs = [globs]
        
        sequence_files = []
        for g in globs:
            sequence_files.extend(glob.glob(g, recursive=True))
        
        self.sequence_files = sorted(sequence_files)
        if max_items and len(self.sequence_files) > max_items:
            self.sequence_files = random.sample(self.sequence_files, max_items)
        
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.sequence_files)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """Get a full sequence stack."""
        seq_file = self.sequence_files[idx]
        
        # Use memory-mapped loading to avoid loading entire array
        # Create mmap per-access (safe for multiprocessing)
        frames_mmap = np.load(seq_file, mmap_mode='r')
        frames = np.array(frames_mmap)  # Shape: [T, H, W, 3] - copy to array for processing
        del frames_mmap  # Release mmap reference
        
        # Convert each frame to tensor
        frame_tensors = []
        for frame in frames:
            img = Image.fromarray(frame.astype(np.uint8))
            if self.transform:
                img = self.transform(img)
            else:
                img = T.Compose([
                    T.Resize(224, antialias=True),
                    T.CenterCrop(224),
                    T.ToTensor(),
                ])(img)
            frame_tensors.append(img)
        
        # Stack frames: [T, C, H, W]
        stack = torch.stack(frame_tensors)
        
        # Ensure float32 (not float64) for MPS compatibility
        if stack.dtype == torch.float64:
            stack = stack.float()
        
        return stack, str(seq_file)

