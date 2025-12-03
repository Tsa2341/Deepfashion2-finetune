import argparse
import json
import os
import glob
from PIL import Image
import numpy as np


def build_base_dataset():
    dataset = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    cat_names = [
        "short_sleeved_shirt",
        "long_sleeved_shirt",
        "short_sleeved_outwear",
        "long_sleeved_outwear",
        "vest",
        "sling",
        "shorts",
        "trousers",
        "skirt",
        "short_sleeved_dress",
        "long_sleeved_dress",
        "vest_dress",
        "sling_dress",
    ]

    for idx, name in enumerate(cat_names, start=1):
        dataset["categories"].append(
            {
                "id": idx,
                "name": name,
                "supercategory": "clothes",
                "keypoints": [str(i + 1) for i in range(294)],
                "skeleton": [],
            }
        )

    return dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert per-image DeepFashion2 JSONs to a single COCO-style JSON"
    )
    p.add_argument(
        "--annos-dir",
        required=True,
        help="Directory containing per-image annotation JSONs (e.g. 000001.json)",
    )
    p.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing image files (e.g. 000001.jpg)",
    )
    p.add_argument("--output", required=True, help="Output COCO JSON path")
    p.add_argument(
        "--ext", default=".json", help="Annotation file extension (default: .json)"
    )
    return p.parse_args()


def extract_numeric_id(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    digits = "".join([c for c in base if c.isdigit()])
    return int(digits) if digits else None


def fill_keypoints(points, landmarks, cat):
    points_x = landmarks[0::3]
    points_y = landmarks[1::3]
    points_v = landmarks[2::3]

    mapping = {
        1: (0, 25),
        2: (25, 33),
        3: (58, 31),
        4: (89, 39),
        5: (128, 15),
        6: (143, 15),
        7: (158, 10),
        8: (168, 14),
        9: (182, 8),
        10: (190, 29),
        11: (219, 37),
        12: (256, 19),
        13: (275, 19),
    }

    if cat not in mapping:
        return points, 0

    start, length = mapping[cat]
    px = np.array(points_x)
    py = np.array(points_y)
    pv = np.array(points_v)
    count = min(length, len(pv))
    for i in range(count):
        n = start + i
        points[3 * n] = px[i]
        points[3 * n + 1] = py[i]
        points[3 * n + 2] = pv[i]

    num_keypoints = int((pv > 0).sum())
    return points, num_keypoints


def convert_deepfashion2_to_coco(
    annos_dir: str, images_dir: str, out_path: str, ext: str = ".json"
):
    """Convert a folder of per-image DeepFashion2 annotation JSONs into a
    single COCO-style JSON file.

    Returns the constructed dataset dict.
    """
    if not os.path.isdir(annos_dir):
        raise SystemExit(f"Annotations directory not found: {annos_dir}")
    if not os.path.isdir(images_dir):
        raise SystemExit(f"Images directory not found: {images_dir}")

    dataset = build_base_dataset()
    sub_index = 0

    pattern = os.path.join(annos_dir, "*" + ext)
    annos = sorted(glob.glob(pattern))
    if not annos:
        raise SystemExit(f"No annotation files found with pattern: {pattern}")

    for anno_file in annos:
        img_id = extract_numeric_id(anno_file)
        if img_id is None:
            print(f"Skipping file with no numeric id: {anno_file}")
            continue

        img_base = os.path.basename(os.path.splitext(anno_file)[0])
        img_name = os.path.join(images_dir, img_base + ".jpg")
        if not os.path.exists(img_name):
            found = False
            for alt_ext in (".jpg", ".jpeg", ".png"):
                alt = os.path.join(images_dir, img_base + alt_ext)
                if os.path.exists(alt):
                    img_name = alt
                    found = True
                    break
            if not found:
                print(f"Image not found for annotation {anno_file}, skipping")
                continue

        with Image.open(img_name) as imag:
            width, height = imag.size

        with open(anno_file, "r") as f:
            temp = json.load(f)

        pair_id = temp.get("pair_id", None)

        dataset["images"].append(
            {
                "coco_url": "",
                "date_captured": "",
                "file_name": os.path.basename(img_name),
                "flickr_url": "",
                "id": img_id,
                "license": 0,
                "width": width,
                "height": height,
            }
        )

        for k, v in temp.items():
            if k in ("source", "pair_id"):
                continue
            if not isinstance(v, dict):
                continue
            if "landmarks" not in v:
                continue

            points = np.zeros(294 * 3)
            sub_index += 1
            box = v.get("bounding_box", [0, 0, 0, 0])
            w = float(box[2]) - float(box[0])
            h = float(box[3]) - float(box[1])
            x_1 = float(box[0])
            y_1 = float(box[1])
            bbox = [x_1, y_1, w, h]
            cat = int(v.get("category_id", 0))
            style = v.get("style", None)
            seg = v.get("segmentation", [])
            landmarks = v.get("landmarks", [])

            points, num_points = fill_keypoints(points, landmarks, cat)

            dataset["annotations"].append(
                {
                    "area": w * h,
                    "bbox": bbox,
                    "category_id": cat,
                    "id": sub_index,
                    "pair_id": pair_id,
                    "image_id": img_id,
                    "iscrowd": 0,
                    "style": style,
                    "num_keypoints": num_points,
                    "keypoints": points.tolist(),
                    "segmentation": seg,
                }
            )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dataset, f)

    print(
        f"Wrote COCO-style JSON with {len(dataset['images'])} images and {len(dataset['annotations'])} annotations to {out_path}"
    )

    return dataset


if __name__ == "__main__":
    args = parse_args()
    convert_deepfashion2_to_coco(
        args.annos_dir, args.images_dir, args.output, ext=args.ext
    )
