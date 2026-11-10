# 口罩实时检测（YOLOv8n）

摄像头实时口罩检测：**戴了口罩画绿框 "Mask OK"，没戴画红框 "NO Mask!"**，左上角实时显示 FPS，按 `q` 退出。

在 [archie9211/Mask-Detection-Dataset](https://github.com/archie9211/Mask-Detection-Dataset)
（8214 张已标注图片）上训练的 YOLOv8n 模型，40 轮训练后的验证集表现：

| 指标 | 数值 |
|---|---|
| Precision | 88.8% |
| Recall | 70.2% |
| mAP50 | 76.2% |

nano 级模型 + 消费级 GPU（RTX 4070 Laptop）就能跑到实时帧率，无 GPU 用 CPU 也能跑，只是帧率下降。

## 快速开始（30 秒体验）

仓库自带训练好的权重 `weights/best.pt`，clone 后即可直接开摄像头：

```bash
git clone https://github.com/yukhughkui/mask-detection-yolov8.git
cd mask-detection-yolov8
pip install -r requirements.txt
python mask_webcam.py
```

## 从头训练

```bash
# 1. 下载数据集（约 570MB，克隆到本目录）
git clone https://github.com/archie9211/Mask-Detection-Dataset.git mask_repo

# 2. 整理成 YOLO 训练格式（生成 datasets/mask/）
python prepare_data.py

# 3. 训练（有 GPU 自动用 GPU，没有退回 CPU）
python train_mask.py

# 4. 训练完用 runs/ 里的新权重再玩
python mask_webcam.py
```

## 文件说明

```
├── mask_webcam.py     # 摄像头实时检测入口
├── prepare_data.py    # 数据集 → YOLO 格式（含标注合法性校验）
├── train_mask.py      # 训练脚本
├── weights/best.pt    # 训练好的权重（可直接推理）
└── requirements.txt
```

## 说明

- `mask_webcam.py` 优先加载 `weights/best.pt`；若你本地重新训练过，会自动改用 `runs/mask_train/weights/best.pt` 的最新结果
- 检测置信度阈值 0.4；若觉得框时有时无，可把 `model(frame, conf=0.4)` 的阈值调低到 0.3
- 数据集与标注出自 [archie9211](https://github.com/archie9211/Mask-Detection-Dataset)，仅作学习用途
