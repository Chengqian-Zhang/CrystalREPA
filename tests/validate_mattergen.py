"""Validate paper configurations, cached features and pretrained checkpoints."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("launcher", ROOT / "scripts/mattergen.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

import torch
from hydra.utils import instantiate
from mattergen.common.data.collate import collate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints/mattergen")
    parser.add_argument("--experiment", default="all")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/validation.json")
    args = parser.parse_args()
    os.environ["CRYSTALREPA_DATA_ROOT"] = str(args.data_root.resolve())
    catalog = json.loads((ROOT / "experiments/mattergen/models.json").read_text())
    if args.experiment != "all" and args.experiment not in catalog:
        parser.error(f"Unknown experiment: {args.experiment}")
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for alias, entry in catalog.items():
        if args.experiment not in ("all", alias):
            continue
        cfg = launcher.load_config(alias)
        counts = {}
        for split in ("train", "val", "test"):
            data = instantiate(cfg.data_module[split + "_dataset"])
            counts[split] = len(data)
            if split == "train":
                subset = data.subset([1, 0])
                for key in data.representations:
                    assert torch.equal(subset[0][key], data[1][key])
                    assert torch.equal(subset.repeat(2)[2][key], subset[0][key])
                batch = collate([data[0], data[1]]).to(device)
            del data
        model = instantiate(cfg.lightning_module).to(device)
        path = args.checkpoint_root / alias / "checkpoints/epoch=1899.ckpt"
        checkpoint = torch.load(path, map_location="cpu")
        assert checkpoint["epoch"] == entry["epoch"] == 1899
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
        torch.manual_seed(42)
        loss, metrics = model.diffusion_module.calc_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if alias != "baseline":
            gradients = [p.grad for name, p in model.named_parameters() if "projection_heads" in name]
            assert gradients and all(g is not None for g in gradients)
            assert any(torch.count_nonzero(g) for g in gradients)
        result = {"experiment": alias, "epoch": 1899, "structures": counts,
                  "strict_load": True, "finite_loss_and_gradients": True}
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, batch, loss, metrics
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"device": device, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
