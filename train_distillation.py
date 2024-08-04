import os
from ultralytics import YOLO
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    model_t = YOLO('runs/detect/yolov8s/weights/best.pt')  # the teacher model
    model_s = YOLO('runs/detect/prune/weights/best.pt')  # the student model
    """
    Attributes:
        Distillation: the distillation model
        loss_type: mgd, cwd
        amp: Automatic Mixed Precision
    """
    model_s.train(data="data.yaml", Distillation=model_t.model, loss_type='mgd', amp=False, imgsz=640, epochs=100,
                  batch=20, device=0, workers=0, lr0=0.001)


if __name__ == '__main__':
    main()
