from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_global_classes():

    path = (
        PROJECT_ROOT
        / "mappings"
        / "global_classes.yaml"
    )

    cfg = load_yaml(path)

    classes = cfg["global_classes"]

    return {
        int(k): v
        for k, v in classes.items()
    }


def load_dataset_mappings():

    path = (
        PROJECT_ROOT
        / "mappings"
        / "dataset_class_mapping.yaml"
    )

    return load_yaml(path)


def load_semantic_hierarchy():

    path = (
        PROJECT_ROOT
        / "mappings"
        / "semantic_hierarchy.yaml"
    )

    return load_yaml(path)

def local_to_global(lines, label_map):
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        local_cls = int(parts[0])
        if local_cls not in label_map:
            continue

        global_cls = label_map[local_cls]
        parts[0] = str(global_cls)
        new_lines.append(" ".join(parts) + "\n")

    return new_lines


def global_to_task(lines, active_classes):
    global_to_local = {g:i for i,g in enumerate(active_classes)}

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        global_cls = int(parts[0])
        if global_cls not in global_to_local:
            continue

        parts[0] = str(global_to_local[global_cls])
        new_lines.append(" ".join(parts) + "\n")

    return new_lines

def remap_label_lines(lines, active_classes):
    """
    active_classes: list global class ids của task
    => map global id -> local id (0..nc-1)
    """

    global_to_local = {
        cls_id: i for i, cls_id in enumerate(active_classes)
    }

    new_lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        global_cls = int(parts[0])

        if global_cls not in global_to_local:
            continue

        parts[0] = str(global_to_local[global_cls])
        new_lines.append(" ".join(parts) + "\n")

    return new_lines