"""Linear-probe MatterGen representations for formation energy per atom.

This script evaluates clean MP-20 structures (diffusion time ``t=0``), captures
the atomic representation after each GemNet interaction block, mean-pools the
atoms in each structure, and fits a standardized Ridge regressor.

Example
-------
python -m mattergen.scripts.linear_probe_formation_energy \
    --batch-size 64 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch
from tqdm import tqdm

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.diffusion.lightning_module import DiffusionLightningModule

LOGGER = logging.getLogger(__name__)

MODEL_NAMES = ('baseline', 'dpa_3_1_3m', 'dpa_3_1_3m_ft', 'dpa_3_1_mptrj', 'mace_mpa_0', 'mace_mp_0', 'orb_v3', 'orb_v2', 'sevennet_omni_i12', 'sevennet_l3i5', 'mattersim_v1_5m')


SPLITS = ("train", "val", "test")
DEFAULT_ALPHAS = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)


def _checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch=(\d+)", path.name)
    return int(match.group(1)) if match else -1


def find_checkpoint(run_dir: Path) -> Path:
    checkpoint = run_dir / "checkpoints" / "epoch=1899.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def load_denoiser(run_dir: Path, device: torch.device) -> torch.nn.Module:
    """Restore a run and return its GemNet-based score model."""
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = OmegaConf.load(config_path)
    module = instantiate(config.lightning_module)
    if not isinstance(module, DiffusionLightningModule):
        raise TypeError(f"Expected DiffusionLightningModule, got {type(module)}")

    checkpoint_path = find_checkpoint(run_dir)
    LOGGER.info("Loading %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    incompatible = module.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    return module.diffusion_module.model.to(device).eval()


class CachedSplit:
    """Memory-mapped MP-20 arrays and structure-level regression targets."""

    def __init__(self, split_dir: Path):
        self.split_dir = split_dir
        self.pos = np.load(split_dir / "pos.npy", mmap_mode="r")
        self.cell = np.load(split_dir / "cell.npy", mmap_mode="r")
        self.atomic_numbers = np.load(split_dir / "atomic_numbers.npy", mmap_mode="r")
        self.num_atoms = np.load(split_dir / "num_atoms.npy", mmap_mode="r")
        self.offsets = np.concatenate(([0], np.cumsum(self.num_atoms, dtype=np.int64)))
        with (split_dir / "formation_energy_per_atom.json").open() as handle:
            target_data = json.load(handle)
        self.targets = np.asarray(target_data["values"], dtype=np.float64)

        if len(self.targets) != len(self.num_atoms):
            raise ValueError(
                f"{split_dir}: {len(self.targets)} targets for {len(self.num_atoms)} structures"
            )
        if self.offsets[-1] != len(self.pos) or len(self.pos) != len(self.atomic_numbers):
            raise ValueError(f"{split_dir}: inconsistent atom-array lengths")

    def __len__(self) -> int:
        return len(self.num_atoms)

    def graph(self, index: int) -> ChemGraph:
        start, end = self.offsets[index : index + 2]
        n_atoms = int(self.num_atoms[index])
        return ChemGraph(
            pos=torch.from_numpy(np.array(self.pos[start:end], copy=True)).float().remainder(1.0),
            cell=torch.from_numpy(np.array(self.cell[index], copy=True)).float().unsqueeze(0),
            atomic_numbers=torch.from_numpy(
                np.array(self.atomic_numbers[start:end], copy=True)
            ).long(),
            num_atoms=torch.tensor(n_atoms, dtype=torch.long),
            num_nodes=n_atoms,
        )

    def batches(self, batch_size: int) -> Iterator[Batch]:
        for start in range(0, len(self), batch_size):
            yield Batch.from_data_list(
                [self.graph(i) for i in range(start, min(start + batch_size, len(self)))]
            )


def mean_pool_atoms(node_features: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
    pooled = node_features.new_zeros((size, node_features.shape[-1]))
    pooled.index_add_(0, batch, node_features)
    counts = torch.bincount(batch, minlength=size).clamp_min(1).unsqueeze(1)
    return pooled / counts


@contextmanager
def capture_interaction_layers(
    denoiser: torch.nn.Module, layers: Sequence[int]
) -> Iterator[dict[int, torch.Tensor]]:
    """Capture ``h`` from selected 1-indexed GemNet interaction blocks."""
    gemnet = denoiser.gemnet
    num_layers = len(gemnet.int_blocks)
    bad_layers = [layer for layer in layers if not 1 <= layer <= num_layers]
    if bad_layers:
        raise ValueError(f"Layers {bad_layers} are invalid; model has {num_layers} blocks")

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def hook(layer: int):
        def save_output(_module, _inputs, output):
            captured[layer] = output[0]

        return save_output

    try:
        for layer in layers:
            handles.append(gemnet.int_blocks[layer - 1].register_forward_hook(hook(layer)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def extract_representations(
    denoiser: torch.nn.Module,
    dataset: CachedSplit,
    layers: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    chunks: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    with capture_interaction_layers(denoiser, layers) as captured:
        for batch in tqdm(
            dataset.batches(batch_size),
            total=(len(dataset) + batch_size - 1) // batch_size,
            leave=False,
            desc=dataset.split_dir.name,
        ):
            batch = batch.to(device)
            captured.clear()
            # The structures are uncorrupted and t=0 is the clean end of the SDE.
            denoiser(batch, torch.zeros(batch.num_graphs, device=device))
            for layer in layers:
                pooled = mean_pool_atoms(captured[layer], batch.batch, batch.num_graphs)
                chunks[layer].append(pooled.float().cpu().numpy())
    return {layer: np.concatenate(chunks[layer], axis=0) for layer in layers}


def probe(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    alphas: Sequence[float],
) -> dict[str, float]:
    """Tune Ridge alpha on validation RMSE and report held-out test metrics."""
    train_ok = np.isfinite(train_y) & np.isfinite(train_x).all(axis=1)
    val_ok = np.isfinite(val_y) & np.isfinite(val_x).all(axis=1)
    test_ok = np.isfinite(test_y) & np.isfinite(test_x).all(axis=1)
    train_x, train_y = train_x[train_ok], train_y[train_ok]
    val_x, val_y = val_x[val_ok], val_y[val_ok]
    test_x, test_y = test_x[test_ok], test_y[test_ok]

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    best_alpha, best_val_rmse = None, float("inf")
    for alpha in alphas:
        candidate = Ridge(alpha=alpha).fit(train_x, train_y)
        val_rmse = mean_squared_error(val_y, candidate.predict(val_x)) ** 0.5
        if val_rmse < best_val_rmse:
            best_alpha, best_val_rmse = alpha, val_rmse

    assert best_alpha is not None
    model = Ridge(alpha=best_alpha).fit(train_x, train_y)
    prediction = model.predict(test_x)
    baseline = np.full_like(test_y, train_y.mean())
    return {
        "alpha": float(best_alpha),
        "val_rmse": float(best_val_rmse),
        "test_rmse": float(mean_squared_error(test_y, prediction) ** 0.5),
        "test_mae": float(mean_absolute_error(test_y, prediction)),
        "test_r2": float(r2_score(test_y, prediction)),
        "baseline_rmse": float(mean_squared_error(test_y, baseline) ** 0.5),
        "baseline_mae": float(mean_absolute_error(test_y, baseline)),
        "n_train": int(len(train_y)),
        "n_val": int(len(val_y)),
        "n_test": int(len(test_y)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("checkpoints/mattergen"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/cache/mp_20"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/linear_probe_formation_energy"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    # MatterGen uses four GemNet interaction blocks.
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(1, 5)))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-cache", action="store_true", help="Do not reuse/save representation arrays")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    selected = set(args.models or MODEL_NAMES)
    datasets = {split: CachedSplit(args.data_root / split) for split in SPLITS}
    rows = []

    for label in MODEL_NAMES:
        run_name = label
        if label not in selected:
            continue
        LOGGER.info("Evaluating %s (%s)", label, run_name)
        run_dir = args.runs_root / run_name
        model_cache = args.output_dir / label
        model_cache.mkdir(parents=True, exist_ok=True)
        representations: dict[str, dict[int, np.ndarray]] = {}
        missing_splits = []
        for split in SPLITS:
            cache_path = model_cache / f"{split}_representations.npz"
            if cache_path.is_file() and not args.no_cache:
                with np.load(cache_path) as cached:
                    representations[split] = {
                        layer: cached[f"layer_{layer}"] for layer in args.layers
                    }
            else:
                missing_splits.append(split)

        if missing_splits:
            denoiser = load_denoiser(run_dir, device)
            for split in missing_splits:
                split_reps = extract_representations(
                    denoiser, datasets[split], args.layers, args.batch_size, device
                )
                representations[split] = split_reps
                if not args.no_cache:
                    np.savez(
                        model_cache / f"{split}_representations.npz",
                        **{f"layer_{layer}": values for layer, values in split_reps.items()},
                    )
            del denoiser
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for layer in args.layers:
            metrics = probe(
                representations["train"][layer],
                representations["val"][layer],
                representations["test"][layer],
                datasets["train"].targets,
                datasets["val"].targets,
                datasets["test"].targets,
                args.alphas,
            )
            row = {"model": label, "run_name": run_name, "layer": layer, **metrics}
            rows.append(row)
            LOGGER.info(
                "%s layer %d: RMSE %.5f, MAE %.5f, R2 %.5f (alpha=%g)",
                label,
                layer,
                metrics["test_rmse"],
                metrics["test_mae"],
                metrics["test_r2"],
                metrics["alpha"],
            )

        with (args.output_dir / "results.json").open("w") as handle:
            json.dump(rows, handle, indent=2)
        with (args.output_dir / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
