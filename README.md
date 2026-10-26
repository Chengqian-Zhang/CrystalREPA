# CrystalREPA

Official implementation of **CrystalREPA: Transferring Physical Priors from
Universal MLIPs to Crystal Generative Models**.

**Accepted at NeurIPS 2026.**

[Paper](https://arxiv.org/abs/2605.08960) · [Full text](https://arxiv.org/html/2605.08960v1)

CrystalREPA aligns the atom-wise hidden states of a crystal generative model with
representations from a frozen machine-learning interatomic potential (MLIP).
Its element-aware contrastive objective, EA-NCE, excludes same-element negative pairs while retaining atom-wise positive pairs.

This repository releases the MatterGen implementation of CrystalREPA, including
pretrained checkpoints and training code.

## Installation

The MatterGen implementation uses Python 3.10 and PyTorch 2.2.1 with CUDA 11.8.
From the repository root:

```bash
conda create -n crystalrepa python=3.10 pip
conda activate crystalrepa

python -m pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
python -m pip install -e ./mattergen
```

The scripts in `scripts/` select the implementation in this repository.

## Generate crystals with a pretrained model

Download a pretrained MatterGen model from
[CrystalREPA-Models](https://huggingface.co/CyrilPKU/CrystalREPA-Models)
and generate 1,024 crystal structures. Run from the repository root with the
Hugging Face `hf` CLI installed:

```bash
hf download CyrilPKU/CrystalREPA-Models \
  --repo-type model --include "mattergen/dpa_3_1_3m/*" --local-dir checkpoints

python scripts/mattergen.py generate \
  --model_path=checkpoints/mattergen/dpa_3_1_3m --checkpoint_epoch=1899 \
  --output_path=outputs/generated/dpa_3_1_3m --batch_size=64 --num_batches=16 \
  --record_trajectories=False
```

Generated structures are saved under `outputs/generated/dpa_3_1_3m/` as EXTXYZ
and a ZIP archive of CIF files. Sampling requires no training data or teacher
representations. The command generates 16 batches of 64 structures. If GPU
memory is limited, reduce `--batch_size` and increase `--num_batches` to keep
the total at 1,024 (for example, 32 × 32). For a quick trial, use
`--num_batches=1` to generate 64 structures.

The sampling configuration uses 1,000 diffusion steps and the `ALEX_MP_20`
atom-count distribution. See [MatterGen usage](docs/mattergen.md#sampling)
for sampling and evaluation options.

Evaluation uses the [MatterGen evaluation pipeline](https://github.com/microsoft/mattergen).
Relaxation and reference-based metrics require the corresponding potential and
reference datasets.

### Available pretrained models

The release includes a baseline and ten MLIP teachers. All checkpoints use
**epoch 1899**, corresponding to 1,900 training epochs. To use another model,
replace `dpa_3_1_3m` in the download and generation commands with its experiment
name below.

| Experiment | Teacher |
| --- | --- |
| `baseline` | No alignment |
| `dpa_3_1_3m` | DPA-3.1-3M |
| `dpa_3_1_3m_ft` | DPA-3.1-3M-FT |
| `dpa_3_1_mptrj` | DPA-3.1-MPtrj |
| `mace_mpa_0` | MACE-MPA-0 |
| `mace_mp_0` | MACE-MP-0 |
| `orb_v3` | ORB v3 |
| `orb_v2` | ORB v2 |
| `sevennet_omni_i12` | SevenNet-Omni-i12 |
| `sevennet_l3i5` | SevenNet-l3i5 |
| `mattersim_v1_5m` | MatterSim-v1-5M |

## Training

Download MP-20 structures and precomputed teacher representations from
[CrystalREPA-Data](https://huggingface.co/datasets/CyrilPKU/CrystalREPA-Data),
then train with a paper configuration:

```bash
hf download CyrilPKU/CrystalREPA-Data \
  --repo-type dataset --include "mattergen/datasets/cache/mp_20/*" --local-dir .
export CRYSTALREPA_DATA_ROOT="$PWD/mattergen/datasets/cache"

python scripts/mattergen.py train --experiment dpa_3_1_3m
```

Use `--experiment baseline` to train without representation alignment. All
eleven configurations share the same MP-20 cache and load only the selected
teacher's representations. Training from this cache does not require installing
the teacher models themselves.

If the data are already stored elsewhere, set `CRYSTALREPA_DATA_ROOT` to the
parent of `mp_20/`. See [Data and checkpoints](docs/artifacts.md) for file formats,
fixed download revisions and pretrained preprocessing details.

Experiment configurations are in `experiments/mattergen/`. Output is written to
`outputs/mattergen/<experiment>`. Use `--output-root` for independent runs;
existing runs in the selected output directory are resumed automatically.
See [MatterGen usage](docs/mattergen.md) for optional training checks,
continuation training, configuration overrides and formation-energy probing.

## Citation

The paper has been accepted at NeurIPS 2026. The link below points to the
arXiv preprint; the camera-ready version is forthcoming.

```bibtex
@inproceedings{zhang2026crystalrepa,
  title={CrystalREPA: Transferring Physical Priors from Universal MLIPs to Crystal Generative Models},
  author={Zhang, Chengqian and Jin, Yucheng and Zhang, Duo and Li, Tiejun and Wang, Han},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  note={Accepted at NeurIPS 2026; camera-ready version forthcoming},
  url={https://arxiv.org/abs/2605.08960}
}
```

## Acknowledgments and license

CrystalREPA is inspired by [REPA](https://github.com/sihyun-yu/REPA)
(*Representation Alignment for Generation: Training Diffusion Transformers Is
Easier Than You Think*). We thank the authors for their work and open-source
implementation.

The MatterGen implementation builds on [Microsoft MatterGen](https://github.com/microsoft/mattergen).
Its MIT license, copyright notices and third-party notices are retained in
`mattergen/LICENSE` and `mattergen/NOTICE`. See [LICENSE](LICENSE).
