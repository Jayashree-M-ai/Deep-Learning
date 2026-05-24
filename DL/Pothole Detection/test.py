# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:08:31 2026

@author: Jayashree M
"""

from ultralytics import YOLO
from tkinter import Tk, filedialog
import cv2
import os

# Load model
model = YOLO("runs/detect/train/weights/best.pt")

# File picker
Tk().withdraw()
file_path = filedialog.askopenfilename()

# Get file extension
ext = os.path.splitext(file_path)[1].lower()

# Image extensions
image_exts = ['.jpg', '.jpeg', '.png']

# Video extensions
video_exts = ['.mp4', '.avi', '.mov']

# IMAGE
if ext in image_exts:

    results = model(file_path)

    annotated_frame = results[0].plot()

    resized_frame = cv2.resize(annotated_frame, (900, 600))

    cv2.imshow("Pothole Detection", resized_frame)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# VIDEO
elif ext in video_exts:

    cap = cv2.VideoCapture(file_path)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("Pothole Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Unsupported file format")