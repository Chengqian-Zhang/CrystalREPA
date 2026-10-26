#!/usr/bin/env python
"""Run the vendored backend without changing the environment's editable install."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mattergen"))
os.environ.setdefault("CRYSTALREPA_OUTPUT_ROOT", str(ROOT / "outputs/mattergen"))


def load_config(experiment, overrides=()):
    import mattergen.common.utils.globals  # registers resolvers and PROJECT_ROOT
    from omegaconf import OmegaConf
    config = OmegaConf.load(ROOT / "experiments/mattergen" / f"{experiment}.yaml")
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    return config


def main():
    modules = {"generate": "generate", "evaluate": "evaluate", "probe": "linear_probe_formation_energy"}
    if len(sys.argv) == 3 and sys.argv[1] in modules and sys.argv[2] in ("--help", "-h"):
        command = sys.argv[1]
        sys.argv = [command, "--help"]
        runpy.run_module(f"mattergen.scripts.{modules[command]}", run_name="__main__")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "check", *modules])
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--data-root", help="Directory containing the shared mp_20 cache")
    parser.add_argument("--output-root")
    parser.add_argument("--cpu-threads", type=int, help="Optional PyTorch CPU thread limit")
    parser.add_argument("--smoke", action="store_true", help="One training and validation batch")
    args, extra = parser.parse_known_args()
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error("--cpu-threads must be positive")
        import torch
        torch.set_num_threads(args.cpu_threads)
    if args.data_root:
        os.environ["CRYSTALREPA_DATA_ROOT"] = str(Path(args.data_root).resolve())
    if args.output_root:
        os.environ["CRYSTALREPA_OUTPUT_ROOT"] = str(Path(args.output_root).resolve())
    if args.command in modules:
        if args.command == "probe" and args.data_root:
            extra.extend(["--data-root", args.data_root])
        sys.argv = [args.command, *extra]
        runpy.run_module(f"mattergen.scripts.{modules[args.command]}", run_name="__main__")
        return
    if "CRYSTALREPA_DATA_ROOT" not in os.environ:
        parser.error("Set CRYSTALREPA_DATA_ROOT or pass --data-root")
    config = load_config(args.experiment, extra)
    if args.command == "check":
        from hydra.utils import instantiate
        from mattergen.common.data.collate import collate
        module = instantiate(config.data_module)
        for split in ("train", "val", "test"):
            dataset = getattr(module, f"{split}_dataset")
            batch = collate([dataset[0], dataset[1]])
            print(json.dumps({"experiment": args.experiment, "split": split,
                              "structures": len(dataset), "batch_atoms": int(batch.num_nodes),
                              "features": {k: list(v.shape) for k, v in dataset.representations.items()}}))
        instantiate(config.lightning_module)
        print("Model configuration OK")
        return
    if args.smoke:
        import torch
        torch.set_num_threads(4)
        config.auto_resume = False
        config.checkpoint_path = None
        config.trainer.default_root_dir = str(ROOT / "outputs/smoke" / args.experiment)
        config.trainer.fast_dev_run = 1
        config.trainer.strategy = "auto"
        config.trainer.devices = 1
        config.trainer.accumulate_grad_batches = 1
        config.trainer.logger = False
        config.trainer.enable_checkpointing = False
        config.trainer.callbacks = []
        for split in ("train", "val", "test"):
            config.data_module.batch_size[split] = 2
            config.data_module.num_workers[split] = 0
    from mattergen.diffusion.run import main as train
    import torch
    # Match the original mattergen.scripts.run training entry point.
    torch.set_float32_matmul_precision("high")
    train(config, save_config=not args.smoke, seed=42 if args.smoke else None)


if __name__ == "__main__":
    main()
