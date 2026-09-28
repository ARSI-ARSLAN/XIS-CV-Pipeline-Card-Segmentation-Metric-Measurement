"""
Splits a COCO-format annotations.json into train/val/test (default 70/20/10).
"""
import json
import random
import os
import argparse


def split_coco(ann_path, out_dir, splits=(0.7, 0.2, 0.1), seed=0):
    with open(ann_path) as f:
        coco = json.load(f)

    images = coco["images"][:]
    random.Random(seed).shuffle(images)
    n = len(images)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    parts = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    class_counts = {}
    for split, imgs in parts.items():
        anns = [a for im in imgs for a in anns_by_img.get(im["id"], [])]
        with open(os.path.join(out_dir, f"{split}.json"), "w") as f:
            json.dump({"images": imgs, "annotations": anns,
                       "categories": coco["categories"]}, f)
        summary[split] = {"images": len(imgs), "annotations": len(anns)}
        for a in anns:
            class_counts.setdefault(split, {}).setdefault(a["category_id"], 0)
            class_counts[split][a["category_id"]] += 1

    print(json.dumps(summary, indent=2))
    with open(os.path.join(out_dir, "split_summary.json"), "w") as f:
        json.dump({"counts": summary, "class_distribution": class_counts}, f, indent=2)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="data/dataset/annotations.json")
    ap.add_argument("--out", default="data/dataset/splits")
    args = ap.parse_args()
    split_coco(args.ann, args.out)
