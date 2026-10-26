# Data and checkpoints

Large data files and model weights are managed separately from the source code.
They are hosted in [CrystalREPA-Data](https://huggingface.co/datasets/CyrilPKU/CrystalREPA-Data)
and [CrystalREPA-Models](https://huggingface.co/CyrilPKU/CrystalREPA-Models).
Artifacts are grouped by generative framework; the current release contains MatterGen.

## Download

Run these commands from the CrystalREPA repository root with the Hugging Face
`hf` CLI installed. Sampling requires only the model download:

```bash
hf download CyrilPKU/CrystalREPA-Models \
  --repo-type model --include "mattergen/*" --local-dir checkpoints
```

This places each model at `checkpoints/mattergen/<experiment>/`, including its
`config.yaml` and `checkpoints/epoch=1899.ckpt`. To download only one model,
use `--include "mattergen/dpa_3_1_3m/*"` instead.

For training or representation probing, also download the data:

```bash
hf download CyrilPKU/CrystalREPA-Data \
  --repo-type dataset --include "mattergen/datasets/cache/mp_20/*" --local-dir .
export CRYSTALREPA_DATA_ROOT="$PWD/mattergen/datasets/cache"
```

The CLI preserves the remote directory hierarchy, so this command places data
under `mattergen/datasets/cache/mp_20/`. If you already have the cache under
`datasets/cache/mp_20/`, no second download is needed; instead set
`CRYSTALREPA_DATA_ROOT="$PWD/datasets/cache"`. For probing, pass
`--data-root "$CRYSTALREPA_DATA_ROOT/mp_20"`.

For a fixed release snapshot, append `--revision` with these verified commit IDs:

| Repository | Revision |
| --- | --- |
| CrystalREPA-Data | `f29ad928e1def201b41eb3ec0ebea21c35f39ca2` |
| CrystalREPA-Models | `09615e61b107431c8f07fd283eaa261ea0911a99` |

## Training caches

Set `CRYSTALREPA_DATA_ROOT` to the parent of the following cache directories:

```text
cache/
└── mp_20/
    ├── train/
    ├── val/
    └── test/
```

All eleven configurations use this shared cache, including all ten teacher
representations. Structures use Niggli reduction without primitive-cell
conversion. Teacher features follow the same structures and atom ordering.

The released epoch-1899 `baseline` and `dpa_3_1_3m` checkpoints were trained with
primitive-cell preprocessing; the other nine checkpoints used the released
preprocessing. This provenance is recorded by
`training_preprocessing_matches_release` in the model catalog. Using the shared
cache for new training or continuation does not change a checkpoint's training
history.

Each split contains:

| File | Shape / content |
| --- | --- |
| `pos.npy` | Fractional coordinates, `(total_atoms, 3)` |
| `cell.npy` | Lattice matrices, `(num_structures, 3, 3)` |
| `atomic_numbers.npy` | Atomic numbers, `(total_atoms,)` |
| `num_atoms.npy` | Atoms per structure, `(num_structures,)` |
| `structure_id.npy` | Structure identifiers, `(num_structures,)` |
| `<teacher>_rep.npy` | Frozen atom-wise representations, `(total_atoms, teacher_dim)` |

The baseline does not need a representation file. REPA loads only the selected
teacher's file. Teacher dimensions are 128 for the three DPA models, 64 for the
two SevenNet models, and 256 for MACE, ORB and MatterSim. Representations must be
finite and follow the ordering of `pos.npy` and `atomic_numbers.npy`.

The feature cache contains representations of clean structures, computed offline.
Teacher inference and feature extraction are not part of the training entrypoint.
For formation-energy probing, each split additionally needs
`formation_energy_per_atom.json` with a `values` array indexed by structure.

## Checkpoints

```text
checkpoints/mattergen/
├── baseline/
│   ├── config.yaml
│   └── checkpoints/epoch=1899.ckpt
├── dpa_3_1_3m/
│   ├── config.yaml
│   └── checkpoints/epoch=1899.ckpt
└── ...
```

`experiments/mattergen/models.json` lists each experiment's configuration,
checkpoint location and SHA256 checksum. All eleven models correspond to epoch
1899. The checkpoint contains model parameters, optimizer/scheduler states and
training progress. Training-log checkpoint-selection histories are not included.

Sampling needs only the model directory; it does not read training caches.
For training or continuation, provide `CRYSTALREPA_DATA_ROOT` and choose a new
output directory as described in [MatterGen usage](mattergen.md).
