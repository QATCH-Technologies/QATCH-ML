"""
QModel-MOIRAI pilot — Cache builder + Dataset
=============================================

Preprocessing a run (read CSV -> uniform-dt resample -> Difference ->
median smooth -> normalised SEQ_LEN tensors + soft targets) is the slow
part. We do it once per run and cache to a compact ``.npz``, mirroring the
"render once, train many" philosophy of the v6 pipeline (but far smaller —
one ~50 KB npz per run instead of thousands of images).

Each cache file stores:
    sequence (SEQ_LEN, C_in) f32 | targets (N_POI, SEQ_LEN) f32
    poi_pos (N_POI,) f32 | poi_mask (N_POI,) bool
    viscosity_cP f32 | tier i64 | run_id (str, in the filename)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import config as C
from data import RunSpec, build_run_sample

LOG = logging.getLogger("moirai.cache")


def cache_path_for(run_id: str) -> Path:
    safe = run_id.replace("/", "_").replace("\\", "_")
    return C.CACHE_ROOT / f"{safe}.npz"


def build_cache(specs: List[RunSpec], overwrite: bool = False) -> List[Path]:
    """Materialise one .npz per run. Returns the list of cache paths written."""
    C.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    n_skip = 0
    for i, spec in enumerate(specs, 1):
        out = cache_path_for(spec.run_id)
        if out.exists() and not overwrite:
            written.append(out)
            continue
        sample = build_run_sample(spec)
        if sample is None:
            n_skip += 1
            continue
        np.savez_compressed(
            out,
            sequence=sample.sequence,
            targets=sample.targets,
            poi_pos=sample.poi_pos,
            poi_mask=sample.poi_mask,
            viscosity_cP=np.float32(sample.viscosity_cP),
            tier=np.int64(sample.tier),
        )
        written.append(out)
        if i % 25 == 0:
            LOG.info("cached %d/%d runs (skipped %d)", i, len(specs), n_skip)
    LOG.info("Cache build done: %d files, %d skipped", len(written), n_skip)
    return written


class POIDataset(Dataset):
    """Loads cached .npz samples. Cheap enough to keep on-the-fly (no RAM cache),
    matching the v6 CACHE_MODE=None decision that avoided the 470 GB sidecar
    blow-up."""

    def __init__(self, cache_paths: List[Path]):
        self.paths = [p for p in cache_paths if p.exists()]
        if not self.paths:
            raise RuntimeError("POIDataset: no cache files found")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        d = np.load(self.paths[idx])
        return {
            "sequence": torch.from_numpy(d["sequence"]).float(),  # (L, C_in)
            "targets": torch.from_numpy(d["targets"]).float(),  # (P, L)
            "poi_pos": torch.from_numpy(d["poi_pos"]).float(),  # (P,)
            "poi_mask": torch.from_numpy(d["poi_mask"]).bool(),  # (P,)
            "viscosity_cP": float(d["viscosity_cP"]),
            "tier": int(d["tier"]),
        }


def collate(batch):
    return {
        "sequence": torch.stack([b["sequence"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
        "poi_pos": torch.stack([b["poi_pos"] for b in batch]),
        "poi_mask": torch.stack([b["poi_mask"] for b in batch]),
        "viscosity_cP": torch.tensor([b["viscosity_cP"] for b in batch]),
        "tier": torch.tensor([b["tier"] for b in batch]),
    }
