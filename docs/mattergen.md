# MatterGen usage

Run commands from the repository root after installing the package.

## Training settings

Each configuration in `experiments/mattergen/` defines the model, data cache,
alignment objective and training settings. The paper configurations run for
1,900 epochs and evaluate the zero-based epoch-1899 checkpoint.

```bash
export CRYSTALREPA_DATA_ROOT=/path/to/datasets/cache
python scripts/mattergen.py train --experiment orb_v3 --output-root outputs/run1
```

To continue from pretrained weights while retaining optimizer state:

```bash
python scripts/mattergen.py train --experiment dpa_3_1_3m \
  --output-root outputs/continued auto_resume=false trainer.max_epochs=2000 \
  checkpoint_path="$PWD/checkpoints/mattergen/dpa_3_1_3m/checkpoints/epoch=1899.ckpt"
```

OmegaConf overrides are supported, for example `data_module.num_workers.train=0`.

### Optional training checks

After setting `CRYSTALREPA_DATA_ROOT`, these commands can help diagnose data or
environment problems before a full training run. They are not required for
sampling with a pretrained model.

```bash
# Validate data and instantiate the model.
python scripts/mattergen.py check --experiment dpa_3_1_3m

# Run one training and validation batch.
python scripts/mattergen.py train --experiment dpa_3_1_3m --smoke
```

`--smoke` uses small batches and disables checkpoint writes.
`--cpu-threads` optionally limits PyTorch's CPU thread count.

## Sampling

```bash
python scripts/mattergen.py generate \
  --model_path=checkpoints/mattergen/orb_v3 --checkpoint_epoch=1899 \
  --output_path=outputs/generated/orb_v3 --batch_size=64 --num_batches=16 \
  --record_trajectories=False
```

The output includes EXTXYZ structures and a ZIP archive of CIF files. Use
`python scripts/mattergen.py generate --help` for sampling options and
`python scripts/mattergen.py evaluate --help` for evaluation options.

## Formation-energy probing

```bash
python scripts/mattergen.py probe \
  --runs-root checkpoints/mattergen --data-root /path/to/datasets/cache/mp_20 \
  --models baseline dpa_3_1_3m --output-dir outputs/probing --device cuda
```

The probe extracts atom-wise hidden states on clean structures, mean-pools them
per crystal and fits a standardized ridge regressor. The checkpoint is selected
explicitly at epoch 1899. All models use the shared `mp_20` cache;
pass the desired aliases with `--models`. See [Data and checkpoints](artifacts.md)
for pretrained checkpoint preprocessing provenance.
