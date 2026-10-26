"""Load cached, clean-structure atom features without importing teacher models."""
from dataclasses import replace
from pathlib import Path

import numpy as np

from mattergen.common.data.dataset import CrystalDataset

TEACHER_DIMS = {
    "dpa_3_1_3m": 128, "dpa_3_1_3m_ft": 128, "dpa_3_1_mptrj": 128,
    "mace_mpa_0": 256, "mace_mp_0": 256, "orb_v3": 256, "orb_v2": 256,
    "sevennet_omni_i12": 64, "sevennet_l3i5": 64, "mattersim_v1_5m": 256,
}


def from_cache_path(cache_path, teacher, transforms=None, properties=None, dataset_transforms=None):
    if teacher not in TEACHER_DIMS:
        raise ValueError(f"Unknown teacher: {teacher}")
    dataset = CrystalDataset.from_cache_path(cache_path, transforms=transforms, properties=properties)
    path = Path(cache_path) / f"{teacher}_rep.npy"
    features = np.load(path, mmap_mode="r", allow_pickle=False)
    expected = (len(dataset.atomic_numbers), TEACHER_DIMS[teacher])
    if features.shape != expected:
        raise ValueError(f"{path}: expected {expected}, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: non-finite teacher features")
    dataset = replace(dataset, representations={f"{teacher}_rep": features})
    for transform in dataset_transforms or []:
        dataset = transform(dataset)
    return dataset
