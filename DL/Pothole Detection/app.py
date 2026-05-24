# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:50:45 2026

@author: Jayashree M
"""

import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
import os

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# ---------------- FUNCTION ---------------- #

def upload_file():

    file_path = filedialog.askopenfilename(
        title="Select Image or Video",
        filetypes=[
            ("Image and Video Files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov")
        ]
    )

    if not file_path:
        return

    ext = os.path.splitext(file_path)[1].lower()

    image_exts = ['.jpg', '.jpeg', '.png']
    video_exts = ['.mp4', '.avi', '.mov']

    # ---------- IMAGE DETECTION ---------- #

    if ext in image_exts:

        results = model(file_path)

        annotated_frame = results[0].plot()

        resized_frame = cv2.resize(annotated_frame, (900, 600))

        cv2.imshow("Pothole Detection", resized_frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ---------- VIDEO DETECTION ---------- #

    elif ext in video_exts:

        cap = cv2.VideoCapture(file_path)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame)

            annotated_frame = results[0].plot()

            cv2.imshow("Pothole Detection", annotated_frame)

            # Press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        print("Unsupported file format")


# ---------------- GUI WINDOW ---------------- #

root = tk.Tk()

root.title("Pothole Detection System")

root.geometry("700x450")

root.configure(bg="#f0f0f0")

# ---------- TITLE ---------- #

title_label = tk.Label(
    root,
    text="Real-Time Pothole Detection System",
    font=("Arial", 22, "bold"),
    bg="#f0f0f0",
    fg="darkblue"
)

title_label.pack(pady=20)

# ---------- SUBTITLE ---------- #

subtitle_label = tk.Label(
    root,
    text="Deep Learning based Road Damage Detection using YOLOv8",
    font=("Arial", 13),
    bg="#f0f0f0",
    fg="black"
)

subtitle_label.pack(pady=10)

# ---------- DESCRIPTION ---------- #

description_label = tk.Label(
    root,
    text="Upload a road image or video to detect potholes automatically.",
    font=("Arial", 11),
    bg="#f0f0f0",
    fg="gray"
)

description_label.pack(pady=10)

# ---------- BUTTON ---------- #

upload_btn = tk.Button(
    root,
    text="Upload Image / Video",
    font=("Arial", 14, "bold"),
    bg="darkblue",
    fg="white",
    padx=20,
    pady=10,
    command=upload_file
)

upload_btn.pack(pady=50)

# ---------- FOOTER ---------- #

footer_label = tk.Label(
    root,
    text="Developed using Python, OpenCV and YOLOv8",
    font=("Arial", 10),
    bg="#f0f0f0",
    fg="gray"
)

footer_label.pack(side="bottom", pady=20)

# ---------- RUN APP ---------- #

root.mainloop()