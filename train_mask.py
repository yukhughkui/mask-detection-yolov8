"""训练 YOLOv8n 口罩检测模型（4070 GPU 约 10~20 分钟，CPU 会慢很多）"""
import os
from pathlib import Path
import torch
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE, "datasets", "mask", "mask.yaml")
PROJECT = os.path.join(BASE, "runs")

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  # 首次运行会自动下载 6MB 预训练权重
    results = model.train(
        data=DATA_YAML,
        epochs=40,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",  # 有 GPU 用 GPU，没有自动退回 CPU
        project=PROJECT,
        name="mask_train",
        exist_ok=True,
        patience=10,        # 10 轮不提升就早停
        workers=0,          # Windows 下多进程读图容易崩，用单进程
    )
    print("\n训练完成。最优权重:", Path(results.save_dir) / "weights" / "best.pt")
