# HACOD
# Hierarchical Continual Object Detection

> A Continual Object Detection framework with Feature Distillation, Relational Preservation, and Confidence-Aware Semantic Relaxation.

---

## Overview

This repository implements a continual object detection framework built upon **YOLOv8** for sequential learning of multiple insulator datasets without catastrophic forgetting.

Unlike conventional continual detection methods that only preserve features, our framework jointly maintains

- Feature consistency
- Sample relation consistency
- Semantic hierarchy consistency

to improve knowledge retention while allowing adaptation to newly introduced categories.

---

## Motivation

Continual object detection suffers from catastrophic forgetting when new datasets are sequentially introduced.

Previous methods mainly rely on

- feature distillation
- replay memory
- classifier regularization

However,

- feature representations drift after each task,
- semantic relationships between categories are ignored,
- and uncertain predictions are over-penalized.

Our objective is therefore to preserve both

- geometric knowledge,
- and semantic knowledge

during continual learning.

---

## Method

Our total training loss consists of four components

\[
L =
L_{det}
+
\lambda_{kd}L_{KD}
+
\lambda_{rel}L_{Relation}
+
\lambda_{sem}L_{Semantic}
\]

where

### 1. Detection Loss

Standard YOLOv8 detection loss

- Box regression
- Classification
- Distribution Focal Loss

---

### 2. Feature Distillation

Preserves feature representations extracted from the neck.

\[
L_{KD}
=
1-
\cos(f_s,f_t)
\]

This reduces feature drift after learning new tasks.

---

### 3. Relation Preservation Loss

Instead of preserving individual features only, we additionally preserve inter-sample relations.

For pooled features

\[
R=f f^T
\]

the relation loss becomes

\[
L_{Relation}
=
\|R_s-R_t\|_2^2
\]

This stabilizes the feature manifold throughout continual learning.

---

### 4. Confidence-Aware Semantic Relaxation (CASR)

Instead of hard semantic relabeling, we propose a soft semantic relaxation mechanism.

For low-confidence predictions

\[
\alpha=1-c
\]

where

- \(c\) is the prediction confidence.

The supervision target becomes

\[
y_{sem}
=
(1-\alpha)y_{child}
+
\alpha y_{parent}
\]

Finally,

\[
L_{Semantic}
=
KL(p_s,y_{sem})
\]

Unlike previous hard semantic constraints, CASR softly transfers supervision toward semantic ancestors while preserving discriminative information.

---

## Framework

```
Task 1
      │
      ▼
Teacher
      │
      ▼
Feature Distillation
      │
      ├────────► Relation Preservation
      │
      ├────────► Confidence-aware Semantic Relaxation
      │
      ▼
YOLOv8 Student
      │
      ▼
Task t+1
```

---

## Continual Learning Pipeline

```
Task1
   │
   ▼
Teacher

   │
   ▼

Task2
   │
   ▼

Teacher

   │
   ▼

Task3
   │
   ▼

Teacher

   │
   ▼

Task4
```

Teacher weights are frozen after each task and are used to regularize the current student.

---

## Datasets

Sequential continual learning is performed on four public insulator datasets

| Task | Dataset |
|-------|----------|
| Task 1 | InsPLAD |
| Task 2 | CPLID |
| Task 3 | MPID |
| Task 4 | STN |

Each task introduces new categories while preserving previous knowledge.

---

## Training

```
python train.py
```

Important hyperparameters

```python
lambda_kd = 0.2
lambda_relation = 0.5
lambda_semantic = 0.1
```

---

## Main Contributions

- Feature-level knowledge distillation
- Relation-preserving feature regularization
- Confidence-Aware Semantic Relaxation
- Plug-and-play continual learning framework for YOLOv8
- Applicable to sequential object detection without modifying detector architecture

---

## Citation

If you find this repository useful, please consider citing our paper.

```
Coming soon.
```

---

## License

MIT License
