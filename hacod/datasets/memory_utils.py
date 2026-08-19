from pathlib import Path

import yaml

import shutil
from pathlib import Path

from pathlib import Path
from collections import defaultdict



def read_classes_from_label(label_file):
    classes = set()

    if not Path(label_file).exists():
        return classes

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                classes.add(int(parts[0]))

    return classes

def collect_images_by_class(img_dir, label_dir):
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)

    class_to_images = {}

    for label_file in label_dir.glob("*.txt"):
        img_path = find_image_by_stem(img_dir, label_file.stem)
        if img_path is None:
            continue

        classes = read_classes_from_label(label_file)

        for cls in classes:
            class_to_images.setdefault(cls, set()).add(img_path)

    # return {k: list(v) for k, v in class_to_images.items()}
    return {
        k:sorted(v)
        for k,v in class_to_images.items()
    }

def collect_current_task_images_by_global_class(task_root, task_info):
    task_root = Path(task_root)

    img_dir = task_root / "images/train"
    lab_dir = task_root / "labels/train"

    class_to_images = {}

    for label_file in sorted(lab_dir.glob("*.txt")):

        img_path = find_image_by_stem(
            img_dir,
            label_file.stem
        )

        if img_path is None:
            continue

        local_classes = read_classes_from_label(label_file)

        for local_cls in local_classes:

            if local_cls not in task_info["label_map"]:
                continue

            global_cls = task_info["label_map"][local_cls]

            class_to_images.setdefault(
                global_cls,
                []
            ).append(str(img_path))

    return class_to_images

def find_task_root_for_image(img_path, seen_tasks):
    img_path = Path(img_path)

    for task_root in seen_tasks:
        task_root = Path(task_root)
        img_dir = task_root / "images/train"

        candidate = img_dir / img_path.name
        if candidate.exists():
            return task_root

    return None


def restore_global_memory_from_disk(memory_root):

    global GLOBAL_MEMORY

    GLOBAL_MEMORY = {}

    memory_root = Path(memory_root)

    img_dir = memory_root / "images/train"
    lab_dir = memory_root / "labels/train"

    if not img_dir.exists():
        return

    for label_file in sorted(lab_dir.glob("*.txt")):

        img_path = find_image_by_stem(
            img_dir,
            label_file.stem
        )

        if img_path is None:
            continue

        classes = read_classes_from_label(
            label_file
        )

        for cls_id in classes:

            GLOBAL_MEMORY.setdefault(
                cls_id,
                set()
            ).add(str(img_path))

    print(
        f"Restored GLOBAL_MEMORY: "
        f"{len(GLOBAL_MEMORY)} classes"
    )

    total = sum(
        len(v)
        for v in GLOBAL_MEMORY.values()
    )

    print(
        f"Restored exemplars: {total}"
    )



def print_replay_statistics(memory_root):

    memory_root = Path(memory_root)

    label_dir = memory_root / "labels/train"

    class_count = defaultdict(int)

    image_count = 0

    for label_file in label_dir.glob("*.txt"):

        image_count += 1

        appeared = set()

        with open(label_file, "r") as f:
            for line in f:

                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])

                # tránh đếm 2 lần cùng class trong 1 ảnh
                appeared.add(cls_id)

        for cls_id in appeared:
            class_count[cls_id] += 1

    print("\n" + "="*60)
    print("REPLAY MEMORY STATISTICS")
    print("="*60)

    inv = {v:k for k,v in GLOBAL_LABELS.items()}

    total = 0

    for cls_id in sorted(class_count.keys()):

        cls_name = inv.get(cls_id, f"class_{cls_id}")

        n = class_count[cls_id]

        total += n

        print(f"[{cls_id:02d}] {cls_name:<40} : {n}")

    print("-"*60)
    print(f"Total replay images: {image_count}")
    print(f"Total class instances: {total}")
    print("="*60)
