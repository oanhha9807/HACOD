from pathlib import Path

import yaml

import shutil
from pathlib import Path


import torch
import torch.nn.functional as F

def semantic_consistency_loss(
    scores,
    active_classes,
    parent_global,
    temperature=2.0
):
    """
    Semantic hierarchy consistency.

    scores:
        [B, nc, N] from detect_hook.feature["scores"]

    active_classes:
        current active global class ids

    parent_global:
        {
            child_global: parent_global
        }

    Example:
        {
            5: 2,
            6: 2,
            7: 3
        }
    """

    if scores is None:
        return torch.tensor(0.0, device="cuda")

    if active_classes is None:
        return torch.tensor(0.0, device=scores.device)

    if len(active_classes) <= 1:
        return torch.tensor(0.0, device=scores.device)

    # --------------------------------------------------
    # [B,nc,N] -> [B,N,nc]
    # --------------------------------------------------
    if scores.ndim != 3:
        return torch.tensor(0.0, device=scores.device)

    scores = scores.permute(0, 2, 1).contiguous()

    B, N, nc = scores.shape

    # --------------------------------------------------
    # mapping
    # --------------------------------------------------
    global_to_local = {
        g: i
        for i, g in enumerate(active_classes)
    }

    # --------------------------------------------------
    # teacher target
    # --------------------------------------------------
    target_scores = scores.detach().clone()

    valid_pairs = 0

    for child_global, parent_id in parent_global.items():

        if child_global not in global_to_local:
            continue

        if parent_id not in global_to_local:
            continue

        child_l = global_to_local[child_global]
        parent_l = global_to_local[parent_id]

        valid_pairs += 1

        # semantic merge
        target_scores[..., parent_l] += \
            0.5 * target_scores[..., child_l]

    if valid_pairs == 0:
        return torch.tensor(
            0.0,
            device=scores.device
        )

    # --------------------------------------------------
    # KD
    # --------------------------------------------------
    student_log_prob = F.log_softmax(
        scores / temperature,
        dim=-1
    )

    teacher_prob = F.softmax(
        target_scores / temperature,
        dim=-1
    )

    loss = F.kl_div(
        student_log_prob,
        teacher_prob.detach(),
        reduction="none"
    ).sum(-1)

    loss = loss.mean() * (temperature ** 2)

    return loss