from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    # model = YOLO('runs/detect/yolov8s/weights/best.pt')  # load a custom model
    model = YOLO('runs/detect/distillation/weights/best.pt')
    # Validate the model
    metrics = model.val(workers=0)  # no arguments needed, dataset and settings remembered
    var1 = metrics.box.map  # map50-95
    var2 = metrics.box.map50  # map50
    var3 = metrics.box.map75  # map75
    var4 = metrics.box.maps  # a list contains map50-95 of each category


if __name__ == '__main__':
    main()
