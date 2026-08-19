from pathlib import Path

import yaml

import shutil
from pathlib import Path

from hacod.datasets.label_mapping import (
    local_to_global,
    global_to_task,
)

def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

#  --------------------------------------------------
def clear_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_image_by_stem(img_dir, stem):
    img_dir = Path(img_dir)
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

#  --------------------------------------------------
def copy_image_label_with_kd(
        img_path,
        src_label_dir,
        dst_img_dir,
        dst_label_dir,
        label_map=None,
        active_classes=None,
        pseudo_label_dir=None,
        is_current=False, force_global=False
):
    img_path = Path(img_path)
    src_label_dir = Path(src_label_dir)

    real_label = src_label_dir / f"{img_path.stem}.txt"

    make_dir(dst_img_dir)
    make_dir(dst_label_dir)

    shutil.copy2(img_path, Path(dst_img_dir) / img_path.name)

    lines = []

    # =========================
    # 1. REAL LABEL
    # =========================
    if real_label.exists():
        with open(real_label, "r") as f:
            raw_lines = f.readlines()

        # 🔥 CURRENT TASK: local -> global
        if is_current and label_map is not None:
            raw_lines = local_to_global(raw_lines, label_map)

        lines.extend(raw_lines)

    # =========================
    # 2. PSEUDO LABEL (ALREADY GLOBAL)
    # =========================
    if pseudo_label_dir is not None:
        pseudo_label = Path(pseudo_label_dir) / f"{img_path.stem}.txt"

        if pseudo_label.exists():
            with open(pseudo_label, "r") as f:
                pseudo_lines = f.readlines()

            lines.extend(pseudo_lines)

    # =========================
    # 3. GLOBAL -> CURRENT TASK FORMAT
    # =========================
    # 🔥 NEW RULE
    if force_global:
        final_lines = lines
    else:
        final_lines = global_to_task(lines, active_classes)
    # final_lines = global_to_task(lines, active_classes)

    with open(Path(dst_label_dir) / f"{img_path.stem}.txt", "w") as f:
        f.writelines(final_lines)




def build_train_dataset(current_task_root, memory_root, output_root, label_map, pseudo_label_dir=None, active_class = None):
    output_root = Path(output_root)

    out_img = output_root / "images/train"
    out_lab = output_root / "labels/train"

    clear_dir(out_img)
    clear_dir(out_lab)

    sources = [
        Path(current_task_root),
        Path(memory_root)
    ]

    for src in sources:
        img_dir = src / "images/train"
        lab_dir = src / "labels/train"

        if not img_dir.exists():
            continue
        is_current = (src == Path(current_task_root))

        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG"]:
            for img_path in sorted(img_dir.glob(ext)):
                use_pseudo = pseudo_label_dir if src == Path(current_task_root) else None
                copy_image_label_with_kd(
                    img_path=img_path,
                    src_label_dir=lab_dir,
                    dst_img_dir=out_img,
                    dst_label_dir=out_lab,
                    label_map=label_map,
                    active_classes=active_class,
                    pseudo_label_dir=use_pseudo,
                    is_current=is_current   # ⭐ thêm

                )

#  --------------------------------------------------
# def copy_val_dataset(current_task_root, output_root):
def copy_val_dataset(
        current_task_root,
        output_root,
        label_map,
        active_classes
):
    current_task_root = Path(current_task_root)
    output_root = Path(output_root)

    src_img = current_task_root / "images/test"
    src_lab = current_task_root / "labels/test"

    dst_img = output_root / "images/test"
    dst_lab = output_root / "labels/test"

    clear_dir(dst_img)
    clear_dir(dst_lab)
    is_current = True

    if src_img.exists():
        for ext in ["*.jpg","*.JPG", "*.jpeg", "*.png", "*.bmp"]:
            for img_path in src_img.glob(ext):

                copy_image_label_with_kd(
                    img_path,
                    src_lab,
                    dst_img,
                    dst_lab,
                    label_map=label_map,
                    active_classes=active_classes,
                    is_current=True,
                )


def write_yaml(
    yaml_path,
    dataset_root,
    active_classes
):

    names = []

    inv = {
        v:k for k,v in GLOBAL_LABELS.items()
    }

    for cls_id in active_classes:
        names.append(inv[cls_id])

    nc = len(active_classes)

    names_text = "[" + ", ".join(
        [f"'{x}'" for x in names]
    ) + "]"

    content = f"""
path: {Path(dataset_root).resolve()}

train: images/train
val: images/test

nc: {nc}
names: {names_text}
"""

    with open(yaml_path, "w") as f:
        f.write(content.strip())
