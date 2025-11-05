"""Symbolic rule scaffolding for LLM integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SymbolicTriple:
    subj: str
    pred: str
    obj: str
    confidence: float = 1.0
    provenance: str = "manual"


def triples_to_json(triples: List[SymbolicTriple]) -> List[Dict[str, object]]:
    return [
        {
            "subj": triple.subj,
            "pred": triple.pred,
            "obj": triple.obj,
            "confidence": triple.confidence,
            "provenance": triple.provenance,
        }
        for triple in triples
    ]
