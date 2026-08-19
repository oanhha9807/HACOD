from pathlib import Path

import yaml

import shutil
from pathlib import Path


class FeatureHook:
    def __init__(self, module):
        self.feature = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inp, out):
        if isinstance(out, (list, tuple)):
            out = out[0]
        self.feature = out

    @property
    def features(self):
        return self.feature

    def close(self):
        self.hook.remove()


def fd_loss(fs, ft):
    if fs is None or ft is None:
        return torch.tensor(0.0, device=ft.device if ft is not None else 'cpu')

    if fs.shape[-2:] != ft.shape[-2:]:
        ft = F.interpolate(ft, size=fs.shape[-2:], mode='bilinear', align_corners=False)

    if fs.shape[1] != ft.shape[1]:
        c = min(fs.shape[1], ft.shape[1])
        fs = fs[:, :c]
        ft = ft[:, :c]

    return F.mse_loss(fs, ft.detach())


def kd_loss(
        student_feat,
        teacher_feat
):
    """
    Feature distillation.

    student_feat:
        B,C,H,W

    teacher_feat:
        B,C,H,W
    """

    if student_feat is None or teacher_feat is None:

        return torch.tensor(
            0.0,
            device=student_feat.device
            if student_feat is not None
            else teacher_feat.device
        )

    # -----------------------------
    # spatial align
    # -----------------------------
    if student_feat.shape[-2:] != teacher_feat.shape[-2:]:

        teacher_feat = F.interpolate(
            teacher_feat,
            size=student_feat.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    # -----------------------------
    # channel align
    # -----------------------------
    if student_feat.shape[1] != teacher_feat.shape[1]:

        c = min(
            student_feat.shape[1],
            teacher_feat.shape[1]
        )

        student_feat = student_feat[:, :c]
        teacher_feat = teacher_feat[:, :c]

    # -----------------------------
    # normalize
    # -----------------------------
    student_feat = F.normalize(
        student_feat.flatten(2),
        dim=1
    )

    teacher_feat = F.normalize(
        teacher_feat.flatten(2),
        dim=1
    )

    # return F.mse_loss(
    #     student_feat,
    #     teacher_feat.detach()
    # )
    fs = F.adaptive_avg_pool2d(student_feat, 1).flatten(1)
    ft = F.adaptive_avg_pool2d(teacher_feat, 1).flatten(1)

    fs = F.normalize(fs, dim=1)
    ft = F.normalize(ft, dim=1)

    return (
            1 -
            F.cosine_similarity(
                fs,
                ft.detach(),
                dim=1
            ).mean()
    )

def pooled_feature(x):
    """
    GAP feature vector
    B,C,H,W -> B,C
    """
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.flatten(1)
    x = F.normalize(x, dim=1)
    return x