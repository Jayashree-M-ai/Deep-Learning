# -*- coding: utf-8 -*-
"""
Created on Sun May 24 11:41:37 2026

@author: Jayashree M
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read YOLO results
df = pd.read_csv("runs/detect/train/results.csv")

# Plot mAP50
plt.plot(df['metrics/mAP50(B)'])

plt.title("mAP50 Graph")

plt.xlabel("Epoch")

plt.ylabel("mAP50")

plt.show()