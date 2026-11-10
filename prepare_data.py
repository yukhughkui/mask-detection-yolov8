"""把 archie9211/Mask-Detection-Dataset 仓库整理成 ultralytics YOLO 训练格式"""
import os
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE / "mask_repo"    # 先 clone 数据集仓库到本目录（见 README）
OUT = BASE / "datasets" / "mask"

CLASS_NAMES = ["mask", "nomask"]  # 仓库 obj.names：0=mask 1=nomask


def copy_split(list_file: Path, split: str):
    img_out = OUT / "images" / split
    lbl_out = OUT / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n_img = n_lbl = 0
    for line in list_file.read_text().splitlines():
        rel = line.strip().replace("\\", "/")
        if not rel:
            continue
        img = REPO / rel
        if not img.exists():
            img = REPO / "imgs" / Path(rel).name
        if not img.exists():
            continue
        lbl = img.with_suffix(".txt")
        shutil.copy2(img, img_out / img.name)
        n_img += 1
        if lbl.exists():
            # 检查标注合法性：5 个数字一行，类别号只能是 0/1
            ok = True
            for ln in lbl.read_text().splitlines():
                parts = ln.split()
                if len(parts) != 5 or int(parts[0]) not in (0, 1):
                    ok = False
                    break
            if ok:
                shutil.copy2(lbl, lbl_out / lbl.name)
                n_lbl += 1
            else:
                (img_out / img.name).unlink()  # 标注坏的整对丢弃
                n_img -= 1
    return n_img, n_lbl


if __name__ == "__main__":
    for split, list_name in [("train", "train.txt"), ("val", "test.txt")]:
        lf = REPO / list_name
        if lf.exists():
            ni, nl = copy_split(lf, split)
            print(f"{split}: {ni} 张图, {nl} 个标注")
        else:
            print(f"找不到 {lf}")

    # 写 dataset yaml
    yaml_path = OUT / "mask.yaml"
    yaml_path.write_text(
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n  0: {CLASS_NAMES[0]}\n  1: {CLASS_NAMES[1]}\n",
        encoding="utf-8",
    )
    print("yaml 已写入:", yaml_path)
