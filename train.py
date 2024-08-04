import os
from ultralytics import YOLO
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    # model = YOLO(r'ultralytics/cfg/models/v8/yolov8s.yaml').load('runs/detect/yolov8s/weights/best.pt')
    model_s = YOLO("./runs/detect/prune/weights/prune.pt")
    model_s.train(data="data.yaml", Distillation = None, loss_type='None', amp=False, imgsz=640, epochs=50, batch=20, device=0, workers=0)


if __name__ == '__main__':
    main()
