**Overview**

This document describes a streamlined, up-to-date workflow to prepare the DeepFashion2 dataset and run Mask2Former training on a GPU. Use the provided Python helper scripts in this folder to download weights, prepare the dataset, convert annotations, and start training.

**Prerequisites**

- A Linux or macOS machine with an NVIDIA GPU and CUDA 13 support (or a remote instance with CUDA 13 installed).
- Sufficient disk space for the dataset and checkpoints (DeepFashion2 full images ~10s of GBs). NVMe recommended.
- Python 3.8+ (3.10/3.11 recommended).

1. Create and activate a Python virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install PyTorch (CUDA 13)

Go to https://pytorch.org/get-started/locally/ and select the appropriate options (Linux/macOS, pip, CUDA 13) to get the exact pip wheel command for your Python version. Example (replace with the command from the site):

```bash
# Example placeholder — replace with the exact command from pytorch.org
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu13x
```

3. Install other Python dependencies

Install the project's Python requirements (we recommend installing detectron2 with the correct CUDA build as described below):

```bash
python -m pip install -r requirements.txt
```

4. Install Detectron2 (CUDA 13)

Detectron2 wheels are published per-CUDA and PyTorch version. See https://github.com/facebookresearch/detectron2 for wheel links. Example:

```bash
# Replace {url} with the wheel URL that matches your torch+cuda version ex: git+https://github.com/facebookresearch/detectron2.git
python -m pip install {detectron2_wheel_url}
```

If a wheel for CUDA 13 is not available for your torch version, build from source per detectron2 docs:

```bash
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

5. Install Mask2Former library in editable mode

From the repository root run:

```bash
python -m pip install -e Mask2Former
```

6. Prepare dataset and weights (recommended — automated)

This repo includes a `prepare_deepfashion2.py` helper that automates the common preparation steps:

- Download the public DeepFashion2 zip files from Google Drive (you will be asked for the two share links when the script runs, or edit the script to set them).
- Unzip the archives (with password handling if needed).
- Convert per-image DeepFashion2 annotations into COCO-style JSON files.
- Download the Swin backbone weights into `deepfashion2-finetune/weights/`.

Run the prepare script from the `deepfashion2-finetune` directory:

```bash
cd deepfashion2-finetune
python3 prepare_deepfashion2.py
```

The script will place prepared files under `datasets/deepfashion2/` (COCO JSONs will be under `datasets/coco_format/`). If you prefer to run steps manually, the `deepfashion2_to_coco.py` converter is available.

9. Run a short benchmark (recommended to estimate images/sec)

You can run a short benchmark using the training script included here. Example (adjust config and paths as needed):

```bash
cd deepfashion2-finetune
python3 train_deepfashion2_mask2former_gpu.py \
  --config-file configs/maskformer2_swin_base_384_bs16_50ep.yaml \
  --train-json datasets/coco_format/deepfashion2_train.json \
  --train-imgs datasets/deepfashion2/train/image/ \
  --val-json datasets/coco_format/deepfashion2_val.json \
  --val-imgs datasets/deepfashion2/validation/image/ \
  --output-dir output/maskformer2_swin_base_384_bs16 \
  --device cuda \
  --amp \
  --epochs 1 \
  --batch-size 1 \
  --weights weights/swin_base_patch4_window12_384.pth \
  --num-workers 4 | tee training_log.txt
```

10. Run full training

Use the same script for full training; increase `--epochs`, tune `--batch-size`, and adjust `--config-file` for the model/backbone you want. Example:

```bash
cd deepfashion2-finetune
python3 train_deepfashion2_mask2former_gpu.py \
  --config-file configs/maskformer2_swin_base_384_bs16_50ep.yaml \
  --train-json datasets/coco_format/deepfashion2_train.json \
  --train-imgs datasets/deepfashion2/train/image/ \
  --val-json datasets/coco_format/deepfashion2_val.json \
  --val-imgs datasets/deepfashion2/validation/image/ \
  --output-dir output/maskformer2_swin_base_384_bs16 \
  --device cuda \
  --amp \
  --epochs 50 \
  --batch-size 1 \
  --weights weights/swin_base_patch4_window12_384.pth \
  --num-workers 8 | tee training_log.txt
```

Notes & tuning

- Use `--batch-size 1` for 24GB GPUs; increase if you have more memory.
- Use mixed precision (AMP) where supported to reduce memory and speed up training.
- Increase `--num-workers` to improve data pipeline throughput.
- Use gradient accumulation if you need a larger effective batch size.

Troubleshooting

- If you see OOM: reduce input resolution, reduce augmentations, or use gradient checkpointing.
- If detectron2 import fails: ensure you installed the wheel matching your torch+cuda.

Contact
If you want, I can also generate a small script to run a measured 100-iteration benchmark on your rented server and estimate total time automatically.
