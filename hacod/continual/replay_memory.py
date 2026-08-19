from pathlib import Path

import yaml

import shutil
from pathlib import Path

from pathlib import Path
from collections import defaultdict


from ultralytics import YOLO

GLOBAL_MEMORY = {}

def rebuild_memory_from_all_tasks(
    seen_tasks,
    memory_root,
    memory_budget,
    label_map,
    active_class,
    model_p
):

    global GLOBAL_MEMORY

    memory_root = Path(memory_root)

    dst_img = memory_root / "images/train"
    dst_lab = memory_root / "labels/train"

    make_dir(dst_img)
    make_dir(dst_lab)

    # =====================================================
    # INIT
    # =====================================================
    if "GLOBAL_MEMORY" not in globals():
        GLOBAL_MEMORY = {}

    current_task_root = Path(seen_tasks[-1])

    current_task_info = next(
        t for t in TASKS
        if t["root"] == str(current_task_root)
    )

    # =====================================================
    # ALL SEEN CLASSES
    # =====================================================
    seen_classes = set()

    for task_root in seen_tasks:

        task_root = Path(task_root)

        lab_dir = task_root / "labels/train"

        task_info = next(
            t for t in TASKS
            if t["root"] == str(task_root)
        )

        for lf in lab_dir.glob("*.txt"):

            local_classes = read_classes_from_label(lf)

            for c in local_classes:

                if c in task_info["label_map"]:
                    seen_classes.add(
                        task_info["label_map"][c]
                    )

    seen_classes = sorted(list(seen_classes))

    if len(seen_classes) == 0:
        return

    # =====================================================
    # ICARL QUOTA
    # =====================================================
    quota = max(
        1,
        memory_budget // len(seen_classes)
    )

    print(
        f"\n[iCaRL] classes={len(seen_classes)} "
        f" quota={quota}"
    )

    # =====================================================
    # SHRINK OLD MEMORY
    # =====================================================
    for cls_id in sorted(GLOBAL_MEMORY.keys()):

        imgs = list(GLOBAL_MEMORY[cls_id])

        if len(imgs) > quota:

            GLOBAL_MEMORY[cls_id] = sorted(
                imgs[:quota]
            )

    # =====================================================
    # CURRENT TASK DATA
    # =====================================================
    class_to_images = \
        collect_current_task_images_by_global_class(
            current_task_root,
            current_task_info
        )

    # =====================================================
    # OLD PROTOTYPES
    # =====================================================
    old_prototypes = compute_class_prototypes_from_memory(
        memory_root=memory_root,
        feature_model_path=model_p
    )

    # =====================================================
    # UPDATE ONLY AFFECTED CLASSES
    # =====================================================
    for cls_id, new_images in class_to_images.items():

        new_images = sorted(set(new_images))

        if len(new_images) == 0:
            continue

        # --------------------------------------------
        # NEW CLASS
        # --------------------------------------------
        if cls_id not in GLOBAL_MEMORY:

            print(
                f"[NEW CLASS] {cls_id}"
            )

            # def select_sota_replay_exemplars(
            #         task_root,
            #         memory_per_class,
            #         feature_model_path="yolov8l.pt",
            #         old_prototypes=None,
            #         alpha=0.4,  # current rep
            #         beta=0.2,  # old stability
            #         gamma=0.2,  # diversity
            #         delta=0.2,  # uncertainty
            #         candidate_size=50,
            #         imgsz=640
            # )

            selected = select_sota_replay_exemplars_for_class(
                image_paths=new_images,
                class_id=cls_id,
                feature_model_path=model_p,
                old_prototype=old_prototypes.get(
                    cls_id,
                    None
                ),
                memory_per_class=quota,
                alpha=0.4,
                beta=0.2,
                gamma=0.2,
                delta=0.2
            )

            GLOBAL_MEMORY[cls_id] = sorted(selected)

        # --------------------------------------------
        # OLD CLASS APPEARS AGAIN
        # --------------------------------------------
        else:

            print(
                f"[UPDATE CLASS] {cls_id}"
            )

            old_imgs = list(
                GLOBAL_MEMORY[cls_id]
            )

            candidates = sorted(
                set(old_imgs + new_images)
            )

            selected = select_sota_replay_exemplars_for_class(
                image_paths=candidates,
                class_id=cls_id,
                feature_model_path=model_p,
                old_prototype=old_prototypes.get(
                    cls_id,
                    None
                ),
                memory_per_class=quota,
                alpha=0.4,
                beta=0.2,
                gamma=0.2,
                delta=0.2
            )

            GLOBAL_MEMORY[cls_id] = sorted(selected)

    # =====================================================
    # FLATTEN MEMORY
    # =====================================================
    final_images = set()

    for cls_id, imgs in GLOBAL_MEMORY.items():
        final_images.update(imgs)

    final_images = sorted(list(final_images))

    # =====================================================
    # WRITE MEMORY TO DISK
    # =====================================================
    clear_dir(dst_img)
    clear_dir(dst_lab)

    for img_path in final_images:

        img_path = Path(img_path)

        task_root = find_task_root_for_image(
            img_path,
            seen_tasks
        )

        if task_root is None:
            continue

        task_root = Path(task_root)

        task_info = next(
            t for t in TASKS
            if t["root"] == str(task_root)
        )

        label_path = str(img_path).replace(
            "images/train",
            "labels/train"
        )

        label_path = str(
            Path(label_path).with_suffix(".txt")
        )

        if not Path(label_path).exists():
            continue

        shutil.copy2(
            img_path,
            dst_img / img_path.name
        )

        with open(label_path, "r") as f:
            raw_lines = f.readlines()

        new_lines = []

        for line in raw_lines:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            local_cls = int(parts[0])

            if local_cls not in task_info["label_map"]:
                continue

            global_cls = task_info["label_map"][
                local_cls
            ]

            parts[0] = str(global_cls)

            new_lines.append(
                " ".join(parts) + "\n"
            )

        with open(
            dst_lab / f"{img_path.stem}.txt",
            "w"
        ) as f:
            f.writelines(new_lines)

    # =====================================================
    # DEBUG
    # =====================================================
    print("\n===== MEMORY =====")

    total = 0

    for cls_id in sorted(GLOBAL_MEMORY.keys()):

        n = len(GLOBAL_MEMORY[cls_id])

        total += n

        print(
            f"class {cls_id}: {n}"
        )

    print(
        f"TOTAL EXEMPLARS = {total}"
    )


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