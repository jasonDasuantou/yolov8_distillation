# YOLOv8-distillation yolov8剪枝+蒸馏

`YOLOv8` 轻量化并且提升精度 !

# Prepare the environment

1. Install `CUDA` follow [`CUDA official website`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#download-the-nvidia-cuda-toolkit).

   🚀 RECOMMENDED `CUDA` >= 11.4

2. Install `TensorRT` follow [`TensorRT official website`](https://developer.nvidia.com/nvidia-tensorrt-8x-download).

   🚀 RECOMMENDED `TensorRT` >= 8.4

2. Install python requirements.

   ``` shell
   pip install -r requirements.txt
   ```

3. Install [`ultralytics`](https://github.com/ultralytics/ultralytics) package for ONNX export or TensorRT API building.

   ``` shell
   pip install ultralytics
   ```

5. Prepare your own PyTorch weight such as `yolov8s.pt`.

***NOTICE:***

Please use the latest `CUDA` and `TensorRT`, so that you can achieve the fastest speed !

If you have to use a lower version of `CUDA` and `TensorRT`, please read the relevant issues carefully !

# Normal Usage

``` shell
python train_distillation.py
```

# 提示
1.准备好配置环境

2.准备好（训练好的）老师模型和（训练好的）学生模型

3.更改写在类Distillation_loss的 channels_s和channels_t，将通道数改成自己模型的通道数

csdn上有详细教程，链接：https://blog.csdn.net/W_extend/article/details/140902235?spm=1001.2014.3001.5502
