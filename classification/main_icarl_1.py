import torch
import torch.nn as nn

from torch.optim import SGD

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from avalanche.training.supervised import ICaRL_ori

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    forgetting_metrics
)

from avalanche.logging import (
    InteractiveLogger,
    TextLogger
)

from avalanche.training.plugins import EvaluationPlugin

from benchmark import build_benchmark
import os

import random
import numpy as np
import torch
from torchvision import transforms

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# =========================================================
# MODEL
# =========================================================
class ResNet18Net(nn.Module):

    def __init__(self, n_classes=20):

        super().__init__()

        # -------------------------------------------------
        # FEATURE EXTRACTOR
        # -------------------------------------------------
        self.feature_extractor = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        in_features = self.feature_extractor.fc.in_features

        self.feature_extractor.fc = nn.Identity()

        # -------------------------------------------------
        # CLASSIFIER
        # -------------------------------------------------
        self.classifier = nn.Linear(
            in_features,
            n_classes
        )

    def forward(self, x):

        feats = self.feature_extractor(x)

        logits = self.classifier(feats)

        return logits


# =========================================================
# DEVICE
# =========================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")

# =========================================================
# BENCHMARK
# =========================================================
benchmark = build_benchmark()

# IMPORTANT:
# only NEW classes per experience
benchmark.n_classes_per_exp = [

    17,  # Exp0 -> InsPLAD

    1,   # Exp1 -> defect

    1,   # Exp2 -> porcelain

    1    # Exp3 -> tower
]

benchmark.classes_order = list(range(20))

# =========================================================
# LOGGER
# =========================================================
interactive_logger = InteractiveLogger()

log_file = open(
    "icarl_log.txt",
    "w"
)

text_logger = TextLogger(log_file)

# =========================================================
# EVALUATION
# =========================================================
eval_plugin = EvaluationPlugin(

    accuracy_metrics(
        epoch=True,
        experience=True,
        stream=True
    ),

    loss_metrics(
        epoch=True,
        experience=True,
        stream=True
    ),

    forgetting_metrics(
        experience=True,
        stream=True
    ),

    loggers=[
        interactive_logger,
        text_logger
    ]
)

# =========================================================
# MODEL
# =========================================================
model = ResNet18Net(
    n_classes=20
).to(device)

# =========================================================
# OPTIMIZER
# =========================================================
optimizer = SGD(

    model.parameters(),

    lr=0.01,

    momentum=0.9,

    weight_decay=5e-4
)


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

buffer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=mean, std=std)
])
# =========================================================
# ICARL
# =========================================================
strategy = ICaRL_ori(

    feature_extractor=model.feature_extractor,

    classifier=model.classifier,

    optimizer=optimizer,

    memory_size=500,

    buffer_transform=buffer_transform,

    fixed_memory=True,

    train_mb_size=32,

    train_epochs=30,

    eval_mb_size=64,

    device=device,

    evaluator=eval_plugin
)

# =========================================================
# PHASE NAMES
# =========================================================
phase_names = {

    0: "InsPLAD",

    1: "CPLID",

    2: "MPID",

    3: "STN"
}

# =========================================================
# TRAINING
# =========================================================
print("\n" + "#" * 70)
print("START CONTINUAL LEARNING")
print("#" * 70)

for experience in benchmark.train_stream:

    exp_id = experience.current_experience

    print("\n" + "=" * 70)

    print(f"EXPERIENCE {exp_id}")
    print(f"Dataset: {phase_names[exp_id]}")

    # -----------------------------------------------------
    # CLASSES
    # -----------------------------------------------------
    if hasattr(experience, "classes_in_this_experience"):

        print(
            "Classes:",
            sorted(experience.classes_in_this_experience)
        )

    # -----------------------------------------------------
    # SAMPLES
    # -----------------------------------------------------
    print(
        "Samples:",
        len(experience.dataset)
    )

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------
    print("\n>>> TRAINING <<<")

    strategy.train(experience)

    # -----------------------------------------------------
    # MEMORY BUFFER
    # -----------------------------------------------------
    print("\n>>> MEMORY BUFFER <<<")

    try:

        if hasattr(strategy, "exemplar_sets"):

            total_buffer = 0

            for cls_id, exemplar_set in enumerate(strategy.exemplar_sets):

                if exemplar_set is not None:

                    n = len(exemplar_set)

                    total_buffer += n

                    print(
                        f"Class {cls_id}: {n} samples"
                    )

            print(
                f"\nTotal buffer size: {total_buffer}"
            )

    except Exception as e:

        print(
            f"Cannot inspect buffer: {e}"
        )

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------
    print("\n>>> EVALUATION <<<")

    results = strategy.eval(
        benchmark.test_stream
    )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------
    print("\n>>> RESULTS <<<")

    for k, v in results.items():

        print(f"{k}: {v}")

    print("=" * 70)

# =========================================================
# CLOSE LOG
# =========================================================
log_file.close()

print("\nTraining finished!")