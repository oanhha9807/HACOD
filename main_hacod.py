import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import shutil
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
from collections import defaultdict
from pathlib import Path
import random
import numpy as np
import torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


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
S = "active_classes"

random.seed(42)


#  --------------------------------------------------
# CONFIG
#  --------------------------------------------------
DATASET_ROOT = "/data/oanh/PPG_topic/Yolo/plad/ultralytics/"

TASK_NAMES = [
    "InsPLAD",
    "CPLID",
    "MPID",
    "STN_PLAD"
]

GLOBAL_LABELS = {

    "yoke": 0,
    "yoke suspension": 1,
    "spacer": 2,
    "stockbridge damper": 3,
    "lightning rod shackle": 4,
    "lightning rod suspension": 5,

    "polymer insulator": 6,
    "glass insulator": 7,

    "plate": 8,
    "vari-grip": 9,

    "polymer insulator lower shackle": 10,
    "polymer insulator upper shackle": 11,
    "polymer insulator tower shackle": 12,

    "glass insulator big shackle": 13,
    "glass insulator small shackle": 14,
    "glass insulator tower shackle": 15,

    "spiral damper": 16,
    "sphere": 17,

    # NEW
    "defect": 18,
    "porcelain": 19,
    "tower": 20
}
TASKS = [

    {
        "name": "InsPLAD",
        "root": "/data/oanh/PPG_topic/Yolo/plad/ultralytics/task1_InsPLAD",

        "label_map": {
            0:0,
            1:1,
            2:2,
            3:3,
            4:4,
            5:5,
            6:6,
            7:7,
            8:8,
            9:9,
            10:10,
            11:11,
            12:12,
            13:13,
            14:14,
            15:15,
            16:16,
            17:17
        },

        "active_classes": list(range(18))
    },

    {
        "name": "CPLID",
        "root": "/data/oanh/PPG_topic/Yolo/plad/ultralytics/task2_CPLID",

        "label_map": {
            0:6,
            1:18
        },

        "active_classes": [
            0,1,2,3,4,5,
            6,7,8,9,
            10,11,12,
            13,14,15,
            16,17,18
        ]
    },

    {
        "name": "MPID",
        "root": "/data/oanh/PPG_topic/Yolo/plad/ultralytics/task3_MPID",

        "label_map": {
            0:7,
            1:6,
            2:19
        },

        "active_classes": list(range(20))
    },

    {
        "name": "STN",
        "root": "/data/oanh/PPG_topic/Yolo/plad/ultralytics/task4_STN_PLAD",

        "label_map": {
            0:20,
            1:6,
            2:2,
            3:8,
            4:3
        },

        "active_classes": list(range(21))
    }
]


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


def compute_class_prototypes_from_memory(
    memory_root,
    feature_model_path,
    layer_idx=9,
    imgsz=640
):

    memory_root = Path(memory_root)

    img_dir = memory_root / "images/train"
    lab_dir = memory_root / "labels/train"

    if not img_dir.exists() or not lab_dir.exists():
        return {}

    class_to_images = collect_images_by_class(
        img_dir,
        lab_dir
    )

    extractor = YOLOEmbeddingExtractor(
        feature_model_path,
        layer_idx=layer_idx,
        device=device
    )

    prototypes = {}

    for cls_id, images in class_to_images.items():

        embeddings = []

        for img_path in images:

            try:

                feat = extractor.extract(
                    img_path,
                    imgsz
                )

                if np.isnan(feat).any():
                    continue

                embeddings.append(
                    feat
                )

            except Exception:
                continue

        if len(embeddings) == 0:
            continue

        embeddings = np.stack(
            embeddings
        )

        prototype = np.mean(
            embeddings,
            axis=0
        )

        prototype = (
            prototype /
            (
                np.linalg.norm(
                    prototype
                ) + 1e-8
            )
        )

        prototypes[cls_id] = prototype

    extractor.close()

    return prototypes


def embedding_uncertainty(
    feat,
    prototype
):

    sim = np.dot(
        feat,
        prototype
    )

    return 1.0 - sim
MEMORY_ROOT = "/data/oanh/PPG_topic/Yolo/plad/ultralytics/workspace_icarl_shift_doamin_change_l9/replay_memory"
WORK_ROOT = "/data/oanh/PPG_topic/Yolo/plad/ultralytics/workspace_icarl_shift_doamin_change_l9/"
RUN_PROJECT = "/data/oanh/PPG_topic/Yolo/plad/ultralytics/workspace_icarl_shift_doamin_change_l9/runs_icarl_like_yolov8l"

MEMORY_BUDGET = 500  # tong so anh replay toi da
EPOCHS = 50
IMGSZ = 640
BATCH = 16
DEVICE = 0
FREEZE_LAYERS_AFTER_TASK1 = 2
LR0_TASK1 = 0.01
LR0_INCREMENTAL = 0.005 # cai nay co the tang len xiu nhu 0.002 hoac 0.003 tuy em nhe
device = "cuda" if torch.cuda.is_available() else "cpu"
#  --------------------------------------------------
# BASIC UTILS
#  --------------------------------------------------



from ultralytics.nn.tasks import DetectionModel

# =========================
# SAVE ORIGINAL LOSS ONCE
# =========================
if not hasattr(DetectionModel, "_old_loss"):
    DetectionModel._old_loss = DetectionModel.loss

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

#  --------------------------------------------------
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


#  --------------------------------------------------
# YOLO DATASET BUILDING
#  --------------------------------------------------

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

#  --------------------------------------------------

#  --------------------------------------------------

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

import cv2
import torch
import numpy as np
import torch.nn.functional as F

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO


class YOLOEmbeddingExtractor:

    def __init__(
        self,
        model_path,
        layer_idx=9,
        device="cuda"
    ):

        self.device = device

        self.model = YOLO(model_path).model
        self.model.to(device)
        self.model.eval()

        self.feature = None

        self.hook = self.model.model[layer_idx].register_forward_hook(
            self._hook_fn
        )

    def _hook_fn(self, module, inp, out):

        if isinstance(out, (tuple, list)):
            out = out[0]

        self.feature = out

    def preprocess(
        self,
        img_path,
        imgsz=640
    ):

        img = cv2.imread(str(img_path))

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (imgsz, imgsz)
        )

        img = img.astype(np.float32) / 255.0

        img = torch.from_numpy(img)

        img = img.permute(2, 0, 1)

        img = img.unsqueeze(0)

        return img.to(self.device)

    @torch.no_grad()
    def extract(
        self,
        img_path,
        imgsz=640
    ):

        x = self.preprocess(
            img_path,
            imgsz
        )

        _ = self.model(x)

        feat = self.feature

        feat = F.adaptive_avg_pool2d(
            feat,
            (1, 1)
        )

        feat = feat.flatten(1)

        feat = F.normalize(
            feat,
            dim=1
        )

        return feat.squeeze(0).cpu().numpy()

    def close(self):
        self.hook.remove()


from pathlib import Path
import numpy as np
import random
from ultralytics import YOLO


def select_sota_replay_exemplars_for_class(
    image_paths,
    class_id,
    memory_per_class,
    feature_model_path,
    old_prototype=None,
    alpha=0.45,
    beta=0.25,
    gamma=0.20,
    delta=0.10,
    candidate_size=50,
    imgsz=640,
    layer_idx=9
):

    if len(image_paths) <= memory_per_class:
        return list(image_paths)

    extractor = YOLOEmbeddingExtractor(
        feature_model_path,
        layer_idx=layer_idx,
        device=device
    )
    print('*****************Using YOLOEmbeddingExtractor **************')

    features = []
    valid_images = []

    # =====================================
    # EMBEDDING EXTRACTION
    # =====================================
    for img_path in image_paths:

        try:

            feat = extractor.extract(
                img_path,
                imgsz
            )

            if np.isnan(feat).any():
                continue

            features.append(feat)

            valid_images.append(img_path)

        except Exception:
            continue

    extractor.close()

    if len(valid_images) <= memory_per_class:
        return valid_images

    features = np.stack(features)

    features = (
        features /
        (
            np.linalg.norm(
                features,
                axis=1,
                keepdims=True
            ) + 1e-8
        )
    )

    # =====================================
    # CURRENT PROTOTYPE
    # =====================================
    current_proto = np.mean(
        features,
        axis=0
    )

    current_proto = (
        current_proto /
        (
            np.linalg.norm(
                current_proto
            ) + 1e-8
        )
    )

    # =====================================
    # OLD PROTOTYPE
    # =====================================
    if old_prototype is not None:

        old_proto = (
            old_prototype /
            (
                np.linalg.norm(
                    old_prototype
                ) + 1e-8
            )
        )

    else:
        old_proto = None

    # =====================================
    # UNCERTAINTY
    # distance from prototype
    # =====================================
    uncertainties = []

    for feat in features:

        unc = np.linalg.norm(
            feat - current_proto
        )

        uncertainties.append(
            unc
        )

    uncertainties = np.array(
        uncertainties
    )

    uncertainties = (
        uncertainties /
        (
            uncertainties.max()
            + 1e-8
        )
    )

    # =====================================
    # INITIAL EXEMPLAR
    # =====================================
    scores = features @ current_proto

    init_idx = int(
        np.argmax(scores)
    )

    chosen = [init_idx]

    remaining = list(
        range(len(features))
    )

    remaining.remove(
        init_idx
    )

    # =====================================
    # GREEDY SELECTION
    # =====================================
    while len(chosen) < memory_per_class:

        best_idx = None

        best_score = -1e9

        if len(remaining) > candidate_size:

            rng = random.Random(42)

            candidates = rng.sample(
                remaining,
                candidate_size
            )

        else:

            candidates = remaining

        for idx in candidates:

            feat = features[idx]

            # -------------------------
            # representative
            # -------------------------
            rep = np.dot(
                feat,
                current_proto
            )

            # -------------------------
            # stability
            # -------------------------
            if old_proto is not None:

                align = np.dot(
                    feat,
                    old_proto
                )

            else:

                align = 0.0

            # -------------------------
            # diversity
            # -------------------------
            div = min(
                np.linalg.norm(
                    feat - features[j]
                )
                for j in chosen
            )

            # -------------------------
            # uncertainty
            # -------------------------
            unc = uncertainties[idx]

            # -------------------------
            # final score
            # -------------------------
            score = (
                alpha * rep
                + beta * align
                + gamma * div
                + delta * unc
            )
            # score = (
            #         0.5 * rep +
            #         0.2 * align +
            #         0.2 * div +
            #         0.2 * unc
            # )#best resnet




            if score > best_score:

                best_score = score

                best_idx = idx

        if best_idx is None:
            break

        chosen.append(
            best_idx
        )

        remaining.remove(
            best_idx
        )

    return [
        valid_images[i]
        for i in chosen
    ]
#  --------------------------------------------------
def generate_kd_pseudo_labels(
    old_model_path,
    image_dir,
    pseudo_label_dir,
    conf_thres=0.45,
    iou_thres=0.6
):
    """
    Use previous YOLO model as teacher to generate pseudo labels.
    This is teacher-guided distillation for YOLO training.
    """
    image_dir = Path(image_dir)
    pseudo_label_dir = Path(pseudo_label_dir)
    make_dir(pseudo_label_dir)

    teacher = YOLO(old_model_path)

    image_paths = []
    for ext in ["*.jpg","*.JPG", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(list(image_dir.glob(ext)))

    for img_path in image_paths:
        results = teacher.predict(
            source=str(img_path),
            conf=0.6, #default: 0.25
            iou=iou_thres, #default: 0.7
            verbose=False
        )

        pseudo_file = pseudo_label_dir / f"{img_path.stem}.txt"

        with open(pseudo_file, "w", encoding="utf-8") as f:
            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:

                    cls_id = int(box.cls.item())
                    if cls_id not in GLOBAL_LABELS:
                        continue
                    xywhn = box.xywhn[0].tolist()

                    x, y, w, h = xywhn
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


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

#  --------------------------------------------------
def find_task_root_for_image(img_path, seen_tasks):
    img_path = Path(img_path)

    for task_root in seen_tasks:
        task_root = Path(task_root)
        img_dir = task_root / "images/train"

        candidate = img_dir / img_path.name
        if candidate.exists():
            return task_root

    return None


#  --------------------------------------------------
# EVALUATION ON ALL SEEN TASKS
#  --------------------------------------------------

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




# ==================================================
# TRUE FEATURE-DISTILLATION LOSS FOR YOLO TRAINING
# L_total = L_yolo + lambda_fd * ||F_student - F_teacher||_2^2
# ==================================================
LAMBDA_FD = 0.05
LAMBDA_KD = 0.10
LAMBDA_ALIGN = 0.05
LAMBDA_SEM = 0.001
KD_TEMPERATURE = 2.0
SEM_CONF_THRES = 0.50
DISTILL_LAYER = 9

SEMANTIC_PARENT_GLOBAL = {
    10: 6,
    11: 6,
    12: 6,
    # 13: 7,
    # 14: 7,
    # 15: 7,
}


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



def _first_tensor_from_preds(preds):
    """Return the largest prediction tensor from YOLO outputs."""
    tensors = []

    def collect(x):
        if torch.is_tensor(x):
            tensors.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                collect(y)
        elif isinstance(x, dict):
            for y in x.values():
                collect(y)

    collect(preds)
    if not tensors:
        return None
    return max(tensors, key=lambda z: z.numel())


def _prediction_to_3d(preds):
    """
    Convert YOLO prediction tensor to [B, A, C] approximately.
    Works with common YOLOv8 shapes: [B, C, A] or [B, A, C].
    """
    x = _first_tensor_from_preds(preds)
    if x is None:
        return None
    if x.ndim == 4:
        x = x.flatten(2).permute(0, 2, 1).contiguous()
    elif x.ndim == 3:
        if x.shape[1] < x.shape[2]:
            x = x.permute(0, 2, 1).contiguous()
    return x



def build_prototype_tensor(old_prototypes, device):
    """Convert prototype dict {class_id: np.array} to tensor dict."""
    proto = {}
    if old_prototypes is None:
        return proto
    for k, v in old_prototypes.items():
        proto[int(k)] = torch.as_tensor(v, dtype=torch.float32, device=device)
    return proto


def align_loss(student_features, batch, prototype_tensor):
    """
    Cross-dataset semantic feature alignment.
    Align pooled student features to replay prototypes of the same class.
    """
    if student_features is None or not prototype_tensor:
        return torch.tensor(0.0, device=student_features.device if student_features is not None else 'cpu')

    feats = F.adaptive_avg_pool2d(student_features, (1, 1)).flatten(1)
    device = feats.device

    if 'batch_idx' not in batch or 'cls' not in batch:
        return torch.tensor(0.0, device=device)

    batch_idx = batch['batch_idx'].long().view(-1).to(device)
    # cls = batch['cls'].long().view(-1).to(device)

    cls_local = batch['cls'].long().view(-1).to(device)

    active_classes = batch["active_classes"]

    cls = torch.tensor(
        [active_classes[int(x)] for x in cls_local],
        device=device
    )

    losses = []
    for img_i in range(feats.shape[0]):
        labels_i = cls[batch_idx == img_i].unique()
        for c in labels_i:
            c_int = int(c.item())
            if c_int not in prototype_tensor:
                continue
            proto = prototype_tensor[c_int]
            dim = min(feats.shape[1], proto.numel())
            losses.append(F.mse_loss(feats[img_i, :dim], proto[:dim].detach()))

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def feature_align_loss(sf, tf):

    sf = pooled_feature(sf)
    tf = pooled_feature(tf)

    return F.mse_loss(
        sf,
        tf.detach()
    )

#
# def semantic_consistency_loss(student_preds, active_classes=None, parent_global=SEMANTIC_PARENT_GLOBAL, conf_thres=SEM_CONF_THRES):
#     """
#     Semantic consistency for evolving labels.
#     Low-confidence fine-grained predictions are relaxed to their parent class if active.
#     """
#     p = _prediction_to_3d(student_preds)
#     if p is None:
#         return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
#
#     nc = len(active_classes) if active_classes is not None else 0
#     if nc <= 1 or p.shape[-1] < nc:
#         return torch.tensor(0.0, device=p.device)
#
#     logits = p[..., -nc:]
#     probs = torch.softmax(logits.detach(), dim=-1)
#     conf, pred_local = probs.max(dim=-1)
#
#     local_to_global = {i: g for i, g in enumerate(active_classes)}
#     global_to_local = {g: i for i, g in enumerate(active_classes)}
#
#     target = pred_local.clone()
#     low_conf = conf < conf_thres
#
#     for local_id, global_id in local_to_global.items():
#         parent_global_id = parent_global.get(global_id, None)
#         if parent_global_id is None or parent_global_id not in global_to_local:
#             continue
#         parent_local = global_to_local[parent_global_id]
#         mask = (pred_local == local_id) & low_conf
#         target[mask] = parent_local
#
#     return F.cross_entropy(logits.reshape(-1, nc), target.reshape(-1))
def semantic_consistency_loss(
    student_preds,
    active_classes=None,
    parent_global=SEMANTIC_PARENT_GLOBAL,
    conf_thres=SEM_CONF_THRES
):
    """
    Semantic consistency for YOLO incremental learning.
    Works on decoded prediction tensor safely.
    """

    p = _prediction_to_3d(student_preds)
    if p is None:
        return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')

    if active_classes is None or len(active_classes) <= 1:
        return torch.tensor(0.0, device=p.device)

    nc = len(active_classes)

    # ================================
    # class logits
    # ================================
    if p.shape[-1] < nc:
        return torch.tensor(0.0, device=p.device)

    logits = p[..., -nc:]  # [B, A, nc]

    probs = torch.softmax(logits, dim=-1)

    conf, pred_local = probs.max(dim=-1)  # [B, A]

    # ================================
    # mapping
    # ================================
    global_to_local = {g: i for i, g in enumerate(active_classes)}

    # ================================
    # build soft targets (IMPORTANT FIX)
    # ================================
    target_probs = probs.detach().clone()

    low_conf_mask = conf < conf_thres

    for child_global, parent_global in parent_global.items():

        if child_global not in global_to_local:
            continue
        if parent_global not in global_to_local:
            continue

        child_l = global_to_local[child_global]
        parent_l = global_to_local[parent_global]

        mask = low_conf_mask & (pred_local == child_l)

        # 👉 soft merge instead of hard label change
        target_probs[..., parent_l][mask] += target_probs[..., child_l][mask]
        target_probs[..., child_l][mask] = 0.0

    # normalize again
    target_probs = target_probs / (target_probs.sum(dim=-1, keepdim=True) + 1e-8)

    # ================================
    # KL divergence (CORRECT LOSS)
    # ================================
    loss = F.kl_div(
        torch.log(probs + 1e-8),
        target_probs,
        reduction="batchmean"
    )

    return loss

LAMBDA_MEM = 0.15
LAMBDA_REL = 0.05
DISTILL_LAYER = 9


class FeatureHook:
    def __init__(self, module):
        self.feature = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inp, out):

        if isinstance(out, (list, tuple)):
            out = out[0]

        self.feature = out

    def close(self):
        self.hook.remove()

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



def relation_preserve_loss(fs, ft):
    """
    Preserve inter-sample relations.
    MUCH stronger memory retention.
    Similar to PODNet / relational KD.
    """

    if fs is None or ft is None:
        return torch.tensor(
            0.0,
            device=fs.device if fs is not None else ft.device
        )

    if fs.shape[-2:] != ft.shape[-2:]:
        ft = F.interpolate(
            ft,
            size=fs.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    if fs.shape[1] != ft.shape[1]:
        c = min(fs.shape[1], ft.shape[1])
        fs = fs[:, :c]
        ft = ft[:, :c]

    fs = pooled_feature(fs)
    ft = pooled_feature(ft)

    # relation matrix
    Rs = torch.matmul(fs, fs.t())
    Rt = torch.matmul(ft, ft.t())

    return F.mse_loss(Rs, Rt.detach())





def semantic_consistency_loss_new(
        scores,
        active_classes,
        parent_global,
        temperature=2.0,
        alpha=0.1,
        conf_thres=0.5,
):
    """
    Semantic hierarchy KD.

    scores:
        [B, nc, N]
        detect_hook.feature["scores"]

    active_classes:
        global class ids currently active.

    parent_global:
        {
            child_global: parent_global
        }

    Constraint:
        Increase parent probability softly.

    Example:
        10->6
        11->6
        12->6
    """

    # =====================================================
    # safety
    # =====================================================
    if scores is None:
        return torch.tensor(
            0.0,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    if active_classes is None:
        return torch.tensor(
            0.0,
            device=scores.device
        )

    if len(active_classes) <= 1:
        return torch.tensor(
            0.0,
            device=scores.device
        )

    if scores.ndim != 3:
        return torch.tensor(
            0.0,
            device=scores.device
        )

    # =====================================================
    # [B,nc,N] -> [B,N,nc]
    # =====================================================
    scores = scores.permute(
        0,
        2,
        1
    ).contiguous()

    B, N, nc = scores.shape

    # =====================================================
    # mapping
    # =====================================================
    global_to_local = {
        g: i
        for i, g in enumerate(active_classes)
    }

    # =====================================================
    # confidence mask
    # =====================================================
    probs = F.softmax(
        scores.detach(),
        dim=-1
    )

    conf, pred = probs.max(dim=-1)

    valid_mask = conf > conf_thres

    if valid_mask.sum() == 0:
        return torch.tensor(
            0.0,
            device=scores.device
        )

    # =====================================================
    # build teacher target
    # =====================================================
    target_scores = scores.detach().clone()

    valid_pairs = 0

    for child_global, parent_id in parent_global.items():

        # child not active
        if child_global not in global_to_local:
            continue

        # parent not active
        if parent_id not in global_to_local:
            continue

        child_l = global_to_local[child_global]
        parent_l = global_to_local[parent_id]

        # only anchors having
        # confident prediction
        mask = valid_mask

        if mask.sum() == 0:
            continue

        target_scores[..., parent_l][mask] += (
                alpha *
                target_scores[..., child_l][mask]
        )

        valid_pairs += 1

    if valid_pairs == 0:
        return torch.tensor(
            0.0,
            device=scores.device
        )

    # =====================================================
    # center logits
    # prevent extremely large KL
    # =====================================================
    target_scores = (
            target_scores
            - target_scores.mean(
        dim=-1,
        keepdim=True
    )
    )

    # =====================================================
    # KD
    # =====================================================
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
        reduction="batchmean"
    )

    loss = loss * (temperature ** 2)

    return loss
def semantic_consistency_loss_rank(
    scores,
    active_classes,
    parent_global,
    conf_thres=0.5
):

    if scores is None:
        return torch.tensor(0.0, device="cuda")

    scores = scores.permute(0, 2, 1)
    probs = F.softmax(scores, dim=-1)

    obj_conf = probs.max(dim=-1).values
    valid_mask = obj_conf > conf_thres

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=scores.device)

    global_to_local = {g:i for i,g in enumerate(active_classes)}

    loss = 0.0
    count = 0

    for c, p in parent_global.items():

        if c not in global_to_local or p not in global_to_local:
            continue

        cl = global_to_local[c]
        pl = global_to_local[p]

        child = probs[..., cl][valid_mask]
        parent = probs[..., pl][valid_mask]

        loss += F.relu(
            torch.log(child + 1e-6) -
            torch.log(parent + 1e-6)
        ).mean()

        count += 1

    return loss / max(count, 1)
# =========================================================
# STRONG MEMORY KD ATTACH
# Replace attach_feature_distillation_loss(...)
# =========================================================
def prototype_loss(
    student_feat,
    teacher_feat,
    labels=None
):
    """
    Class-wise prototype preservation
    """

    if student_feat is None or teacher_feat is None:
        return torch.tensor(0.0, device=student_feat.device)

    # align spatial + channel
    if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
        teacher_feat = F.interpolate(
            teacher_feat,
            size=student_feat.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    c = min(student_feat.shape[1], teacher_feat.shape[1])
    student_feat = student_feat[:, :c]
    teacher_feat = teacher_feat[:, :c]

    # embedding
    sf = F.adaptive_avg_pool2d(student_feat, 1).flatten(1)
    tf = F.adaptive_avg_pool2d(teacher_feat, 1).flatten(1)

    # -------------------------------------------------
    # CLASS-WISE PROTOTYPE (FIX)
    # -------------------------------------------------
    if labels is None:
        # fallback: batch prototype (weak version)
        proto_s = sf.mean(dim=0)
        proto_t = tf.mean(dim=0)
    else:
        loss = 0.0
        valid = 0

        for cls in labels.unique():
            mask = (labels == cls)

            if mask.sum() < 2:
                continue

            ps = sf[mask].mean(dim=0)
            pt = tf[mask].mean(dim=0)

            ps = F.normalize(ps, dim=0)
            pt = F.normalize(pt, dim=0)

            loss += (1 - F.cosine_similarity(
                ps.unsqueeze(0),
                pt.unsqueeze(0).detach(),
                dim=1
            ))
            valid += 1

        return loss / max(valid, 1)

    # fallback loss
    proto_s = F.normalize(proto_s, dim=0)
    proto_t = F.normalize(proto_t, dim=0)

    return 1 - F.cosine_similarity(
        proto_s.unsqueeze(0),
        proto_t.unsqueeze(0).detach(),
        dim=1
    )

def build_class_prototypes(
        feat,
        labels,
        batch_idx
):
    """
    feat:
        [B,C,H,W]

    labels:
        [num_obj]

    batch_idx:
        [num_obj]
    """

    B = feat.shape[0]

    feat = F.adaptive_avg_pool2d(
        feat,
        1
    ).flatten(1)

    feat = F.normalize(
        feat,
        dim=1
    )

    proto = {}

    for cls in labels.unique():

        cls = int(cls.item())

        mask = labels == cls

        imgs = batch_idx[mask].unique()

        if len(imgs) == 0:
            continue

        f = feat[imgs.long()]

        p = f.mean(0)

        p = F.normalize(
            p,
            dim=0
        )

        proto[cls] = p

    return proto

def boundary_preserve_loss(
        student_feat,
        teacher_feat,
        labels,
        batch_idx,
        important_classes=[6, 7, 19]
):
    """
    Preserve class geometry of MPID.
    """

    if student_feat is None \
            or teacher_feat is None:

        return torch.tensor(
            0.0,
            device=student_feat.device
        )

    # spatial align
    if student_feat.shape[-2:] != teacher_feat.shape[-2:]:

        teacher_feat = F.interpolate(
            teacher_feat,
            size=student_feat.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    c = min(
        student_feat.shape[1],
        teacher_feat.shape[1]
    )

    student_feat = student_feat[:, :c]
    teacher_feat = teacher_feat[:, :c]

    s_proto = build_class_prototypes(
        student_feat,
        labels,
        batch_idx
    )

    t_proto = build_class_prototypes(
        teacher_feat,
        labels,
        batch_idx
    )

    cls_exist = []

    for c in important_classes:
        if c in s_proto and c in t_proto:
            cls_exist.append(c)

    if len(cls_exist) < 2:
        return torch.tensor(
            0.0,
            device=student_feat.device
        )

    loss = 0.
    n = 0

    for i in range(len(cls_exist)):
        for j in range(i + 1,
                       len(cls_exist)):

            c1 = cls_exist[i]
            c2 = cls_exist[j]

            sim_s = F.cosine_similarity(
                s_proto[c1].unsqueeze(0),
                s_proto[c2].unsqueeze(0)
            )

            sim_t = F.cosine_similarity(
                t_proto[c1].unsqueeze(0),
                t_proto[c2].unsqueeze(0)
            )

            loss += F.mse_loss(
                sim_s,
                sim_t.detach()
            )

            n += 1

    return loss / max(n, 1)
import torch
import torch.nn.functional as F


def semantic_consistency_loss123(
    student_preds,
    active_classes=None,
    parent_global=None,
    conf_thres=0.7,
):
    """
    Confidence-Aware Semantic Relaxation Loss (CASR)

    Low-confidence predictions are softly relaxed toward
    their semantic parent instead of hard relabeling.

    Args
    ----
    student_preds :
        YOLO predictions.

    active_classes :
        current active global ids.

    parent_global :
        {
            child_global : parent_global
        }

    conf_thres :
        confidence threshold.

    Returns
    -------
    scalar loss
    """

    p = _prediction_to_3d(student_preds)

    if p is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        return torch.tensor(0.0, device=device)

    if active_classes is None:
        return torch.tensor(
            0.0,
            device=p.device
        )

    nc = len(active_classes)

    if nc <= 1:
        return torch.tensor(
            0.0,
            device=p.device
        )

    if p.shape[-1] < nc:
        return torch.tensor(
            0.0,
            device=p.device
        )

    if parent_global is None:
        return torch.tensor(
            0.0,
            device=p.device
        )

    # --------------------------------------------------
    # class logits
    # [B,A,C]
    # --------------------------------------------------
    logits = p[..., -nc:]

    probs = torch.softmax(
        logits.detach(),
        dim=-1
    )

    conf, pred_local = probs.max(
        dim=-1
    )

    # --------------------------------------------------
    # build mapping
    # --------------------------------------------------
    local_to_global = {
        i: g
        for i, g
        in enumerate(active_classes)
    }

    global_to_local = {
        g: i
        for i, g
        in enumerate(active_classes)
    }

    # --------------------------------------------------
    # initialize target
    # --------------------------------------------------
    target_probs = probs.clone()

    # --------------------------------------------------
    # semantic relaxation
    # --------------------------------------------------
    for local_id, global_id in \
            local_to_global.items():

        parent_global_id = parent_global.get(
            global_id,
            None
        )

        if parent_global_id is None:
            continue

        if parent_global_id \
                not in global_to_local:
            continue

        parent_local = \
            global_to_local[
                parent_global_id
            ]

        # current anchors predicted
        # as this child class
        mask = (
            pred_local == local_id
        )

        if mask.sum() == 0:
            continue

        child_conf = conf[mask]

        # --------------------------------------------------
        # confidence-aware weight
        #
        # conf=1.0 -> alpha=0
        # conf=0.2 -> alpha=0.8
        # --------------------------------------------------
        alpha = (
            1.0 - child_conf
        ).clamp(
            min=0.0,
            max=1.0
        )

        # only relax low-confidence anchors
        low_mask = (
            child_conf < conf_thres
        )

        if low_mask.sum() == 0:
            continue

        idx = torch.where(mask)

        anchor_idx = (
            idx[0][low_mask],
            idx[1][low_mask]
        )

        alpha = alpha[
            low_mask
        ]

        # --------------------------------------------------
        # redistribute probability
        # --------------------------------------------------
        child_prob = \
            target_probs[
                anchor_idx[0],
                anchor_idx[1],
                local_id
            ]

        move_prob = (
            alpha * child_prob
        )

        # parent += moved prob
        target_probs[
            anchor_idx[0],
            anchor_idx[1],
            parent_local
        ] += move_prob

        # child keeps remaining prob
        target_probs[
            anchor_idx[0],
            anchor_idx[1],
            local_id
        ] -= move_prob

    # --------------------------------------------------
    # normalize
    # --------------------------------------------------
    target_probs = (
        target_probs
        /
        (
            target_probs.sum(
                dim=-1,
                keepdim=True
            )
            + 1e-8
        )
    )

    # --------------------------------------------------
    # KL loss
    # --------------------------------------------------
    student_log_prob = F.log_softmax(
        logits,
        dim=-1
    )

    loss = F.kl_div(
        student_log_prob,
        target_probs.detach(),
        reduction="none"
    ).sum(-1).mean()

    return loss

def attach_feature_distillation_loss(
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

            # active_class = yolo_obj.active_class
            #
            # semantic_loss_value = semantic_consistency_loss123(
            #     student_preds=detect_hook.feature["scores"],
            #     active_classes=active_class,
            #     parent_global=SEMANTIC_PARENT_GLOBAL,
            #     conf_thres=0.5
            # )
            # print('semantic_loss_value: ', semantic_loss_value)

            # semantic_loss_value = semantic_consistency_loss_rank(
            #     scores=detect_hook.feature["scores"],
            #     active_classes=active_class,
            #     parent_global=SEMANTIC_PARENT_GLOBAL,
            #     conf_thres=0.5
            # )
            # prototype_loss_value = prototype_loss(
            #     student_hook.feature,
            #     teacher_hook.feature
            # )

            # =========================================
            # 5. FEATURE MEMORY LOSS
            # =========================================


            # =========================================
            # 7. TOTAL LOSS
            # =========================================
            # print('boundary_loss_value: ', boundary_loss_value)
            total_loss = (
                base_loss
                + 0.2 * kd_loss_value
                + 0.5 * rel_loss

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

def train_icarl_like_yolov8l():

    # =====================================================
    # RESUME CONFIG
    # =====================================================
    name_space = ''
    START_TASK = 1

    if START_TASK == 2:
        name_space = 'task_1_InsPLAD'
        EPOCHS = 30
        FREEZE_LAYERS_AFTER_TASK1 = 4
    elif START_TASK==3:
        name_space = 'task_2_CPLID'
        FREEZE_LAYERS_AFTER_TASK1 =4
        EPOCHS = 30
    elif START_TASK==4:
        name_space = 'task_3_MPID'
        FREEZE_LAYERS_AFTER_TASK1 = 4
        EPOCHS = 50


    PRETRAINED_MODEL = (f"/data/oanh/PPG_topic/Yolo/plad/ultralytics/workspace_icarl_shift_doamin_change_l9/"
                        f"runs_icarl_like_yolov8l/{name_space}/weights/best.pt")#
    PRETRAINED_MODEL_all = ("/data/oanh/PPG_topic/Yolo/plad/ultralytics/workspace_icarl_shift_doamin_change_l9/runs_icarl_like_yolov8l/task1_InsPLAD/weights/best.pt")  #

    # task_1_InsPLAD
    # task_2_CPLID
    # task_3_MPID
    # task_4_STN

    # =====================================================
    # INITIAL MODEL
    # =====================================================
    if START_TASK == 1:
        model_path = "yolov8l.pt"
    else:
        model_path = PRETRAINED_MODEL

    # =====================================================
    # RESTORE PREVIOUS TASK HISTORY
    # =====================================================
    seen_tasks = []
    seen_task_names = []

    if START_TASK > 1:

        for i in range(START_TASK - 1):

            seen_tasks.append(TASKS[i]["root"])
            seen_task_names.append(TASKS[i]["name"])


    print("\n" + "=" * 70)
    print("RESUME TRAINING")
    print("=" * 70)
    print("START TASK:", START_TASK)
    print("PRETRAINED:", model_path)
    print("SEEN TASKS:", seen_task_names)
    print("=" * 70)

    # =====================================================
    # CHECK REPLAY MEMORY
    # =====================================================
    replay_img_dir = Path(MEMORY_ROOT) / "images/train"
    replay_lab_dir = Path(MEMORY_ROOT) / "labels/train"

    if START_TASK > 1:

        if (not replay_img_dir.exists()) or (not replay_lab_dir.exists()):
            raise ValueError(
                "❌ Replay memory does not exist.\n"
                "You must restore replay memory generated after previous tasks."
            )

        n_imgs = len(list(replay_img_dir.glob("*")))

        print(f"✅ Replay memory found: {n_imgs} images")

        restore_global_memory_from_disk(
            MEMORY_ROOT
        )

    # =====================================================
    # MAIN LOOP
    # =====================================================
    for task_id, task_info in enumerate(TASKS, start=1):

        # -----------------------------------------
        # SKIP PREVIOUS TASKS
        # -----------------------------------------
        if task_id < START_TASK:
            continue


        if task_id == 1:
            print("Clearing old replay memory...")

            clear_dir(Path(MEMORY_ROOT) / "images/train")
            clear_dir(Path(MEMORY_ROOT) / "labels/train")

            GLOBAL_MEMORY.clear()


        task_root = task_info["root"]
        task_name = task_info["name"]

        print(f"\n========== TASK {task_id}: {task_name} ==========")

        # add CURRENT task
        seen_tasks.append(task_root)
        seen_task_names.append(task_name)

        current_output = f"{WORK_ROOT}/task_{task_id}_{task_name}"

        # =====================================================
        # PSEUDO LABEL GENERATION
        # =====================================================
        pseudo_label_dir = None

        # =====================================================
        # ACTIVE CLASSES
        # =====================================================
        active_classes = task_info["active_classes"]

        print("ACTIVE_CLASSES:", active_classes)

        # =====================================================
        # BUILD TRAIN DATASET
        # =====================================================
        build_train_dataset(
            current_task_root=task_root,
            memory_root=MEMORY_ROOT,
            output_root=current_output,
            label_map=task_info["label_map"],
            pseudo_label_dir=pseudo_label_dir,
            active_class=active_classes,
        )

        n_current = len(
            list((Path(task_root) / "images/train").glob("*")))

        n_memory = len(
            list((Path(MEMORY_ROOT) / "images/train").glob("*")))

        n_final = len(
            list((Path(current_output) / "images/train").glob("*")))

        print(
            f"-------------Current={n_current}, "
            f"Memory={n_memory}, "
            f"Final={n_final}")

        # =====================================================
        # BUILD VAL DATASET
        # =====================================================
        copy_val_dataset(
            current_task_root=task_root,
            output_root=current_output,
            label_map=task_info["label_map"],
            active_classes=active_classes,
        )

        # =====================================================
        # YAML
        # =====================================================
        yaml_path = f"{current_output}/data.yaml"

        write_yaml(
            yaml_path=yaml_path,
            dataset_root=current_output,
            active_classes=active_classes
        )

        # =====================================================
        # LOAD MODEL
        # =====================================================
        teacher_model_path = model_path

        print(f"Loading model: {model_path}")

        model = YOLO(model_path)

        # =====================================================
        # TRAIN CONFIG
        # =====================================================
        if task_id > 1:
            print("🔥 Attaching iCaRL KD loss (task > 1)")
            teacher_model_path = model_path

            model = attach_feature_distillation_loss(
                yolo_obj=model,
                teacher_model_path=teacher_model_path,
                device=DEVICE,
                lambda_kd=0.5,
                # active_class=active_classes
            )

            # IMPORTANT: mark task > 1 so KD is enabled
            model.task_id = task_id
            model.active_class = active_classes

            print("✅ KD loss attached (iCaRL mode)")
            print("LOSS AFTER PATCH:", model.model.loss)

        if task_id == 1:
            freeze_layers = 0
            lr0 = LR0_TASK1
            EPOCHS = 30
        elif task_id ==2:
            freeze_layers  = 2
            lr0 = LR0_INCREMENTAL
            EPOCHS = 30

        # elif:
        elif task_id==3:
            # name_space = 'task_2_CPLID'
            freeze_layers  = 2
            EPOCHS = 30
            lr0 = LR0_INCREMENTAL

        elif task_id==4:
            # name_space = 'task_3_MPID'
            freeze_layers = 2
            EPOCHS = 50
            lr0 = LR0_INCREMENTAL


        epochs = EPOCHS

        print("\n" + "-" * 60)
        print("TRAIN CONFIG")
        print("-" * 60)
        print("freeze_layers:", freeze_layers)
        print("lr0:", lr0)
        print("epochs:", epochs)
        print("seen task:", seen_tasks)
        print("-" * 60)


        # =====================================================
        # TRAIN
        # =====================================================

        model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            project=RUN_PROJECT,
            name=f"task_{task_id}_{task_name}",
            exist_ok=True,
            freeze=freeze_layers,
            lr0=lr0,
            seed=42, deterministic=True,
            # workers=0,
            amp=False, mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            # workers=0
        )

        # =====================================================
        # UPDATE MODEL PATH
        # =====================================================
        model_path = (
            f"{RUN_PROJECT}/"
            f"task_{task_id}_{task_name}/"
            f"weights/best.pt"
        )

        print(f"\n✅ New model saved: {model_path}")

        # =====================================================
        # UPDATE REPLAY MEMORY
        # =====================================================
        print("\nUpdating replay memory...")

        rebuild_memory_from_all_tasks(
            # seen_tasks=[current_output],
            seen_tasks=seen_tasks,
            memory_root=MEMORY_ROOT,
            memory_budget=MEMORY_BUDGET,
            label_map=task_info["label_map"],
            active_class=active_classes,
            model_p=model_path
        )

        print_replay_statistics(MEMORY_ROOT)

        print(f"✅ Replay memory updated after {task_name}")

        # =====================================================
        # EVALUATE
        # =====================================================
        evaluate_on_seen_tasks(
            model_path=model_path,
            seen_tasks=seen_tasks,
            seen_task_names=seen_task_names,
            task_info=task_info,
            current_output=current_output
        )

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

if __name__ == "__main__":
    train_icarl_like_yolov8l()