import os
import random

import numpy as np
import torch
from pathlib import Path

from ultralytics import YOLO

from hacod.datasets.dataset_builder import (
    copy_val_dataset,
    write_yaml,
)



def evaluate_on_seen_tasks(model_path, seen_tasks, seen_task_names, task_info, current_output):
    model = YOLO(model_path)

    print("\n===== Evaluation on all seen tasks =====")

    for task_root, task_name in zip(seen_tasks, seen_task_names):
        task_info = next(
            t for t in TASKS if t["name"] == task_name
        )

        active_classes = task_info["active_classes"]
        # eval_root = Path(task_root)
        eval_root = Path(current_output) / f"eval_{task_name}"

        copy_val_dataset(
            current_task_root=task_root,
            output_root=eval_root,
            label_map=task_info["label_map"],
            active_classes=task_info["active_classes"]
        )

        yaml_path = eval_root / "data_eval.yaml"

        write_yaml(
            yaml_path=yaml_path,
            dataset_root=eval_root,
            active_classes=active_classes
        )

        print(f"\nEvaluating on {task_name}")
        model.val(
            data=str(yaml_path),
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE
        )
