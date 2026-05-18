# Deep Learning Homework 2

学生：张之蔚  
学号：25210980169

本仓库按任务分类保存期中作业代码，报告和模型权重在本地分别整理为独立目录。

## 目录结构

```text
task1_flower_classification/
  Flowers102 分类实验代码：ResNet18/ResNet34 微调、SE/CBAM 注意力、消融实验。

task2_detection_tracking/
  Road Vehicle Images Dataset 上的 YOLOv8 检测训练、学校路口视频 tracking、越线计数、遮挡分析代码。

task3_unet_segmentation/
  Stanford Background Dataset 上从零实现 U-Net、Dice Loss、CE/Dice/CE+Dice 对比实验代码。

requirements_task2_3.txt
  任务2和任务3使用的主要依赖。
```

## 环境

本次实验使用：

- Python 3.12.9
- PyTorch 2.6.0 + CUDA 12.4
- torchvision 0.21.0
- Ultralytics YOLO
- OpenCV
- pandas / matplotlib

建议在项目根目录安装依赖：

```bash
python -m pip install -r task1_flower_classification/requirements.txt
python -m pip install -r requirements_task2_3.txt
```

## 运行方式概览

任务1：

```bash
cd task1_flower_classification
python scripts/prepare_data.py --data-dir data
python scripts/run_experiments.py --plan configs/experiment_plan.yaml --data-dir data --output-dir runs --tracker wandb --wandb-mode offline --num-workers 4 --amp --skip-existing --continue-on-error
python scripts/summarize_results.py --runs-dir runs --out-dir results
```

任务2：

```bash
cd task2_detection_tracking
python scripts/prepare_road_vehicle_dataset.py --kaggle --out-dir data/road_vehicle_yolo
python scripts/run_experiments.py --plan configs/task2_yolo_plan.yaml --data data/road_vehicle_yolo/data.yaml --project runs/detect --device 0 --workers 4 --skip-existing --continue-on-error
python scripts/summarize_results.py --runs-dir runs/detect --out-dir results
python scripts/track_video.py --weights runs/detect/04_yolov8s_baseline/weights/best.pt --source video/068ebedc8785bfbf87d869391b11177a.mp4 --out-dir results/tracking_school_crossing --tracker bytetrack.yaml --line 180 760 700 585
```

任务3：

```bash
cd task3_unet_segmentation
python scripts/prepare_stanford_background.py --kaggle --out-dir data/stanford_background
python scripts/run_experiments.py --plan configs/task3_unet_plan.yaml --data-dir data/stanford_background --runs-dir runs --device cuda --num-workers 4 --tracker wandb --wandb-mode offline --amp --skip-existing --continue-on-error
python scripts/summarize_results.py --runs-dir runs --out-dir results
```

## 报告和权重

本地合并版 LaTeX 报告素材位于：

```text
/home/zhangzhiwei/deeplearning/homework2/docs/1_2_3
```

最终推荐模型权重位于：

```text
/home/zhangzhiwei/deeplearning/homework2/weights
```

权重文件体积较大，不提交 GitHub，请单独上传网盘后在报告中填写下载链接。
