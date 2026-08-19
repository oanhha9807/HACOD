from pathlib import Path

import yaml

import shutil
from pathlib import Path


import torch
import torch.nn.functional as F
from ultralytics import YOLO
from .feature_distillation import (
    FeatureHook,
    kd_loss,
)

from .relation_preservation import (
    relation_preserve_loss,
)

from .semantic_hierarchy import (
    semantic_consistency_loss,
)

def attach_hacod_loss(
    yolo_obj,
    teacher_model_path,
    device,
    lambda_kd=0.5,
    lambda_mem=LAMBDA_MEM,
    lambda_rel=LAMBDA_REL,
    layer_idx=DISTILL_LAYER,
    # active_class=active_classes
):

    student = yolo_obj.model


    # =====================================================
    # LOAD TEACHER
    # =====================================================
    teacher = YOLO(teacher_model_path).model

    teacher.to(device)
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad = False

    # =====================================================
    # FEATURE HOOK
    # =====================================================
    student_layer = student.model[layer_idx]
    teacher_layer = teacher.model[layer_idx]

    student_hook = FeatureHook(student_layer)
    teacher_hook = FeatureHook(teacher_layer)

    print("🔥 STRONG MEMORY DISTILLATION READY")

    # =====================================================
    # PATCH LOSS
    # =====================================================
    def patch_loss(trainer):

        det_model = trainer.model

        old_loss_fn = det_model.loss
        detect_hook = FeatureHook(det_model.model[-1])

        def new_loss(batch, preds=None):
            # print('here: ')

            # =========================================
            # 1. STUDENT FORWARD
            # =========================================
            # if preds is None:
            #     preds = det_model(batch["img"])
            student_preds = preds
            # student_preds = det_model(batch["img"])
            # active_class = yolo_obj.active_class

            det_model = trainer.model

            student_layer = det_model.model[layer_idx]

            teacher_layer = teacher.model[layer_idx]

            student_hook = FeatureHook(student_layer)
            teacher_hook = FeatureHook(teacher_layer)

            # =========================================
            # 2. ORIGINAL YOLO LOSS
            # =========================================
            base = old_loss_fn(batch, student_preds)

            if isinstance(base, tuple):
                base_loss, loss_items = base
            else:
                base_loss = base
                loss_items = None

            # =========================================
            # TASK 1 -> NORMAL TRAIN
            # =========================================
            # if not hasattr(det_model, "task_id") \
            #         or det_model.task_id <= 1:
            #
            #     return (
            #         (base_loss, loss_items)
            #         if loss_items is not None
            #         else base_loss
            #     )

            # =========================================
            # 3. TEACHER FORWARD
            # =========================================
            with torch.no_grad():
                teacher_preds = teacher(batch["img"])

            # =========================================
            # 4. LOGIT KD
            # =========================================
            kd_loss_value = kd_loss(
                student_hook.feature,
                teacher_hook.feature
            )
            rel_loss = relation_preserve_loss(
                student_hook.feature,
                teacher_hook.feature
            )

            semantic_loss_value = semantic_consistency_loss(
                    scores=detect_hook.feature["scores"],
                    active_classes=active_class,
                    parent_global=SEMANTIC_PARENT_GLOBAL,
                    # conf_thres=SEM_CONF_THRES
                )
            total_loss = (
                base_loss
                + 0.2 * kd_loss_value
                + 0.5 * rel_loss
                + 0.2 * semantic_loss_value

                # + 0.1*boundary_loss_value
            )




            if loss_items is not None:
                return total_loss, loss_items

            return total_loss

        det_model.loss = new_loss

        print("✅ STRONG MEMORY KD PATCHED")

    yolo_obj.add_callback(
        "on_train_start",
        # "on_train_start",
        # "on_train_start",
        patch_loss
    )

    return yolo_obj