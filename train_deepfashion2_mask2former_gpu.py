"""
GPU-optimized launcher for Mask2Former training (Swin backbone).

This is adapted from `train_deepfashion2_mask2former.py` with GPU defaults,
AMP enabled by flag, and a short-benchmark mode for measuring throughput.
"""

import sys
import os

# Ensure the Mask2Former package directory is on sys.path. Put the package
# root first so that imports like `Mask2Former.*` and `mask2former` resolve to
# the same module objects (prevents double-loading under different paths).
_mask2former_root = "/content/Deepfashion2-finetune/Mask2Former"
if _mask2former_root not in sys.path:
    sys.path.insert(0, _mask2former_root)

print("sys.path:", sys.path)
print(
    "------------------------------------------------------------------------------------------"
)

import argparse
import json
import math
import shutil

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Mask2Former on DeepFashion2 (COCO JSONs)"
    )
    p.add_argument("--train-json", required=False, default="")
    p.add_argument("--train-imgs", required=False, default="")
    p.add_argument("--val-json", required=False, default="")
    p.add_argument("--val-imgs", required=False, default="")
    p.add_argument("--config-file", required=True)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--amp", action="store_true", help="Enable automatic mixed precision (FP16)"
    )
    p.add_argument(
        "--weights", default="", help="Path to backbone/pretrained weights (.pth)"
    )
    p.add_argument(
        "--benchmark-iters",
        type=int,
        default=0,
        help="If >0, run this many training iterations then exit",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # unregister if present
    for name in ("deepfashion2_train", "deepfashion2_val"):
        try:
            if name in DatasetCatalog.list():
                DatasetCatalog.remove(name)
        except Exception:
            pass

    if args.train_json and args.train_imgs:
        register_coco_instances(
            "deepfashion2_train", {}, args.train_json, args.train_imgs
        )
    if args.val_json and args.val_imgs:
        register_coco_instances("deepfashion2_val", {}, args.val_json, args.val_imgs)

    cfg = get_cfg()

    # Try to register Mask2Former-specific cfg additions if available
    add_maskformer_cfg = None
    # Temporarily wrap DatasetCatalog.register to ignore duplicate registration
    # errors raised by some Mask2Former dataset registration code. Keep this
    # wrapper active through Mask2Former imports, then restore afterwards.
    _orig_register = None
    _dcat = None
    # Also wrap fvcore Registry duplicate checks to avoid double-registering
    # components like D2SwinTransformer when imports happen repeatedly.
    _orig_do_register = None
    _fvreg = None
    try:
        import detectron2.data.catalog as _dcat

        _orig_register = _dcat.DatasetCatalog.register

        def _safe_register(*args, **kwargs):
            try:
                return _orig_register(*args, **kwargs)
            except AssertionError as exc:
                if "already registered" in str(exc):
                    # Ignore duplicate registration to avoid crashes when code
                    # registers ADE20K multiple times.
                    return None
                raise

        _dcat.DatasetCatalog.register = _safe_register
    except Exception:
        _orig_register = None

    try:
        import fvcore.common.registry as _fvreg

        _orig_do_register = _fvreg.Registry._do_register

        def _safe_do_register(self, name, obj):
            if name in self._obj_map:
                # Skip duplicates instead of asserting
                return
            return _orig_do_register(self, name, obj)

        _fvreg.Registry._do_register = _safe_do_register
    except Exception:
        _orig_do_register = None

    try:
        from Mask2Former.mask2former import add_maskformer2_config as _add

        add_maskformer_cfg = _add
    except Exception:
        try:
            from Mask2Former.mask2former.config import add_maskformer2_config as _add

            add_maskformer_cfg = _add
        except Exception as e:
            raise e

    print(f"add_maskformer_cfg: {add_maskformer_cfg}")

    if add_maskformer_cfg is not None:
        try:
            add_maskformer_cfg(cfg)
        except Exception as e:
            print("Warning: failed to apply Mask2Former cfg registration:", e)
            raise e

    # Merge user config
    try:
        cfg.set_new_allowed(True)
        cfg.merge_from_file(args.config_file)
    except Exception as e:
        print(
            "Error merging config file. If this references Mask2Former-specific keys, ensure Mask2Former is installed and 'add_maskformer2_config' is importable.\n",
            e,
        )
        raise

    # Prefer GPU when available
    try:
        import torch

        if args.device is None:
            if torch.cuda.is_available():
                args.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                args.device = "mps"
            else:
                args.device = "cpu"
    except Exception:
        args.device = args.device or "cpu"

    cfg.MODEL.DEVICE = args.device

    if args.train_json:
        cfg.DATASETS.TRAIN = ("deepfashion2_train",)
    else:
        cfg.DATASETS.TRAIN = ()
    if args.val_json:
        cfg.DATASETS.TEST = ("deepfashion2_val",)
    else:
        cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size

    # set AMP
    cfg.SOLVER.AMP.ENABLED = bool(args.amp)

    # estimate iterations
    try:
        if args.train_json:
            with open(args.train_json) as f:
                num_images = len(json.load(f).get("images", []))
        else:
            num_images = 0
    except Exception:
        num_images = 0

    if num_images:
        iters_per_epoch = max(1, math.ceil(num_images / float(max(1, args.batch_size))))
        cfg.SOLVER.MAX_ITER = iters_per_epoch * max(1, args.epochs)
        cfg.TEST.EVAL_PERIOD = iters_per_epoch
        cfg.SOLVER.CHECKPOINT_PERIOD = iters_per_epoch
    else:
        # if benchmarking small number of iters, we'll override later
        cfg.SOLVER.MAX_ITER = cfg.SOLVER.MAX_ITER

    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    try:
        shutil.copyfile(
            args.config_file, os.path.join(cfg.OUTPUT_DIR, "used_config.yaml")
        )
    except Exception:
        pass

    # set weights if provided
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    # If benchmarking requested, override iterations to a small number
    if args.benchmark_iters and args.benchmark_iters > 0:
        print(f"Benchmark mode: running {args.benchmark_iters} training iterations")
        cfg.SOLVER.MAX_ITER = int(args.benchmark_iters)
        cfg.TEST.EVAL_PERIOD = int(args.benchmark_iters)
        cfg.SOLVER.CHECKPOINT_PERIOD = int(args.benchmark_iters)

    # Disable Detectron2 AMP if not requested (we handle AMP via config)
    if not cfg.SOLVER.AMP.ENABLED:
        cfg.SOLVER.AMP.ENABLED = False

    try:
        from Mask2Former.train_net import Trainer as MaskTrainer

        trainer = MaskTrainer(cfg)

        trainer.resume_or_load(resume=args.resume)
        trainer.train()
    finally:
        # Restore original DatasetCatalog.register to avoid side-effects on
        # subsequent library users.
        try:
            if _orig_register is not None and _dcat is not None:
                _dcat.DatasetCatalog.register = _orig_register
        except Exception:
            pass
        try:
            if _orig_do_register is not None and _fvreg is not None:
                _fvreg.Registry._do_register = _orig_do_register
        except Exception:
            pass


if __name__ == "__main__":
    main()
