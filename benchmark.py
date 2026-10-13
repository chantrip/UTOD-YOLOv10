import time
import torch
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "/home/dsai-st125287/project/UTOD-YOLOv10/runs/detect/ruod_bs_v2/weights/best.pt"
IMGSZ = 640
DEVICE = "cuda:0"

WARMUP = 20
ITER = 100
# ----------------------------------------

# Load model
model = YOLO(MODEL_PATH)
model.model.eval().to(DEVICE)

# Optional FP16 (comment out if not needed)
# model.model.half()

# Dummy input (batch=1, RGB, 640x640)
dummy_input = torch.randn(1, 3, IMGSZ, IMGSZ).to(DEVICE)

# If FP16
# dummy_input = dummy_input.half()

# ---------------- WARMUP ----------------
with torch.no_grad():
    for _ in range(WARMUP):
        _ = model.model(dummy_input)

# ---------------- LATENCY ----------------
latencies = []

with torch.no_grad():
    for _ in range(ITER):
        torch.cuda.synchronize()
        start = time.time()

        _ = model.model(dummy_input)

        torch.cuda.synchronize()
        end = time.time()

        latencies.append((end - start) * 1000)  # ms

latencies = np.array(latencies)

mean_latency = latencies.mean()
std_latency = latencies.std()
p95_latency = np.percentile(latencies, 95)

# ---------------- FPS ----------------
total_time = latencies.sum() / 1000  # seconds
fps = ITER / total_time

print("Benchmark scope: synthetic input, batch=1, GPU model-forward only; excludes image loading, decoding, preprocessing, output handling and robot integration.")
print(f"Image size: {IMGSZ}x{IMGSZ}; device: {DEVICE}; warmup: {WARMUP}; iterations: {ITER}.")
print(f"Latency (ms): mean={mean_latency:.2f}, std={std_latency:.2f}, p95={p95_latency:.2f}")
print(f"FPS: {fps:.2f}")
