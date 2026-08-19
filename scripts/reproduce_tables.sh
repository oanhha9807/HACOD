#!/bin/bash

set -e


echo "Reproducing HACOD detection results"

python eval_yolov8l.py \
    --config configs/yolov8l_hacod.yaml \
    --checkpoint checkpoints/hacod_task4.pt


echo "Reproducing YOLOv8l-iCaRL results"

python eval_yolov8l.py \
    --config configs/yolov8l_icarl.yaml \
    --checkpoint checkpoints/icarl_task4.pt