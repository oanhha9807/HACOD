from pathlib import Path

import yaml

import shutil
from pathlib import Path


import torch
import torch.nn.functional as F


def pooled_feature(x):

    x = F.adaptive_avg_pool2d(
        x,
        (1, 1)
    )

    x = x.flatten(1)

    return F.normalize(
        x,
        dim=1
    )


def relation_preserve_loss(
    student_features,
    teacher_features
):

    fs = student_features
    ft = teacher_features

    if fs is None or ft is None:
        return torch.tensor(
            0.0,
            device=(
                fs.device
                if fs is not None
                else ft.device
            )
        )

    if fs.shape[-2:] != ft.shape[-2:]:

        ft = F.interpolate(
            ft,
            size=fs.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    if fs.shape[1] != ft.shape[1]:

        c = min(
            fs.shape[1],
            ft.shape[1]
        )

        fs = fs[:, :c]
        ft = ft[:, :c]

    fs = pooled_feature(fs)
    ft = pooled_feature(ft)

    relation_student = (
        fs @ fs.T
    )

    relation_teacher = (
        ft @ ft.T
    )

    return F.mse_loss(
        relation_student,
        relation_teacher.detach()
    )