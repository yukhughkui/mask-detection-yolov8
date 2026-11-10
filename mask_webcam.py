"""摄像头实时口罩检测：绿框=戴了口罩，红框=没戴
运行后弹出窗口，按 q 退出。"""
import os
import time
import cv2
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
# 优先用仓库自带的训练好的权重；本地重新训练过则用 runs/ 里的最新结果
WEIGHTS = os.path.join(BASE, "weights", "best.pt")
if not os.path.exists(WEIGHTS):
    WEIGHTS = os.path.join(BASE, "runs", "mask_train", "weights", "best.pt")

# 类别名 → (颜色BGR, 中文标签)
COLORS = {
    "mask": ((80, 220, 80), "Mask OK"),        # 绿
    "nomask": ((60, 60, 255), "NO Mask!"),     # 红
    # 兼容 kaggle 三分类命名的模型
    "with_mask": ((80, 220, 80), "Mask OK"),
    "without_mask": ((60, 60, 255), "NO Mask!"),
    "mask_weared_incorrect": ((30, 170, 255), "Bad Mask!"),  # 橙
}
FALLBACK = ((30, 170, 255), "?")


def main():
    model = YOLO(WEIGHTS)
    print("模型加载完成，打开摄像头……")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows 下 DirectShow 打开更快
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("打不开摄像头！检查是否有其他程序占用了摄像头。")
        return

    fps, t_prev = 0.0, time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取画面失败")
            break

        results = model(frame, verbose=False, conf=0.4)
        for r in results:
            names = r.names  # {0: 'mask', 1: 'nomask'}
            for box in r.boxes:
                cls_name = names[int(box.cls)]
                color, label = COLORS.get(cls_name, FALLBACK)
                conf = float(box.conf)
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

                # 框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # 标签牌
                text = f"{label} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, text, (x1 + 3, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FPS
        t_now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, t_now - t_prev))
        t_prev = t_now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow("Mask Detection  (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("已退出。")


if __name__ == "__main__":
    main()
