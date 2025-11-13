"""Configuration loading utilities with caching."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict


@lru_cache(maxsize=8)
def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load YAML config file with caching.
    
    Uses lru_cache to avoid re-parsing the same config file multiple times.
    Cache size of 8 should be sufficient for most use cases.
    
    Args:
        path: Path to YAML config file
        
    Returns:
        Parsed config dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")
    
    # Lazy import yaml to avoid heavy import cost
    import yaml
    
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

