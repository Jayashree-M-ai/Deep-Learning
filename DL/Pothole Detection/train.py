# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:59:18 2026

@author: Jayashree M
"""

from ultralytics import YOLO

model=YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
    )

metrics=model.val()

print(metrics)