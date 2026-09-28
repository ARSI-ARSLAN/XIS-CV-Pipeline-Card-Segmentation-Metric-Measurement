"""
Imports Roboflow COCO export from `coco/` into `data/dataset/`.
1. Copies all images from coco/train, coco/valid, coco/test into data/dataset/images_undistorted/
2. Saves train.json, val.json, test.json into data/dataset/splits/
3. Combines them into data/dataset/annotations.json
"""
import os
import shutil
import json

def import_roboflow(coco_root="coco", dest_root="data/dataset"):
    splits_dir = os.path.join(dest_root, "splits")
    images_dest = os.path.join(dest_root, "images_undistorted")
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(images_dest, exist_ok=True)

    split_map = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }

    all_images = []
    all_annotations = []
    categories = None
    summary = {}
    class_counts = {}

    image_id_offset = 0
    ann_id_offset = 0

    for roboflow_split, pipeline_split in split_map.items():
        split_path = os.path.join(coco_root, roboflow_split)
        ann_path = os.path.join(split_path, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            print(f"Warning: {ann_path} not found.")
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if categories is None:
            # Normalize categories: ensure category id 1 is the card
            # Roboflow sometimes has id 0 or id 1
            categories = data.get("categories", [])

        # Copy images to images_dest
        for img_info in data.get("images", []):
            fname = img_info["file_name"]
            src_img = os.path.join(split_path, fname)
            dst_img = os.path.join(images_dest, fname)
            if os.path.exists(src_img) and not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)

        # Write this split json
        out_split_json = os.path.join(splits_dir, f"{pipeline_split}.json")
        with open(out_split_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        n_imgs = len(data.get("images", []))
        n_anns = len(data.get("annotations", []))
        summary[pipeline_split] = {"images": n_imgs, "annotations": n_anns}

        for a in data.get("annotations", []):
            cat_id = a.get("category_id", 1)
            class_counts.setdefault(pipeline_split, {}).setdefault(cat_id, 0)
            class_counts[pipeline_split][cat_id] += 1

        print(f"[{pipeline_split}] Imported {n_imgs} images, {n_anns} annotations.")

        # Re-id for combined annotations.json
        local_to_global_img_id = {}
        for im in data.get("images", []):
            old_id = im["id"]
            new_id = old_id + image_id_offset
            local_to_global_img_id[old_id] = new_id
            im_copy = dict(im)
            im_copy["id"] = new_id
            all_images.append(im_copy)

        for an in data.get("annotations", []):
            an_copy = dict(an)
            an_copy["id"] = an["id"] + ann_id_offset
            an_copy["image_id"] = local_to_global_img_id[an["image_id"]]
            all_annotations.append(an_copy)

        image_id_offset += len(data.get("images", [])) + 1000
        ann_id_offset += len(data.get("annotations", [])) + 1000

    # Write combined annotations.json
    combined = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": categories
    }
    comb_path = os.path.join(dest_root, "annotations.json")
    with open(comb_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved combined annotations to {comb_path} ({len(all_images)} images, {len(all_annotations)} annotations).")

    # Write split_summary.json
    with open(os.path.join(splits_dir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"counts": summary, "class_distribution": class_counts}, f, indent=2)
    print(f"Saved split summary to {splits_dir}/split_summary.json")

if __name__ == "__main__":
    import_roboflow()
