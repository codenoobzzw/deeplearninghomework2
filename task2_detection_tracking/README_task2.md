# 任务 2：场景目标检测与视频多目标跟踪

## 1. 本任务需要完成什么

本任务对应作业“场景目标检测与视频多目标跟踪”。你需要：

1. 使用 Road Vehicle Images Dataset。
2. 微调 YOLOv8 或同类现代单阶段检测模型。
3. 使用训练好的检测模型对 10-30 秒测试视频逐帧检测和跟踪。
4. 输出 Bounding Box、类别和 Tracking ID。
5. 对遮挡或密集交汇片段截取连续 3-4 帧并分析 ID 是否保持、丢失或跳变。
6. 设置一条虚拟线，统计跨越该线的物体总数。

## 2. 目录结构

```text
task2_detection_tracking/
├── configs/task2_yolo_plan.yaml
├── scripts/
│   ├── check_env.py
│   ├── prepare_road_vehicle_dataset.py
│   ├── train_yolo.py
│   ├── run_experiments.py
│   ├── summarize_results.py
│   ├── log_yolo_results_to_wandb.py
│   ├── make_demo_video_from_images.py
│   ├── track_video.py
│   └── analyze_occlusion.py
├── docs/report_template_task2.md
└── README_task2.md
```

## 3. 数据准备

### 3.1 Kaggle 自动下载

```bash
python scripts/prepare_road_vehicle_dataset.py --kaggle --out-dir data/road_vehicle_yolo
```

### 3.2 使用本地上传的数据

```bash
python scripts/prepare_road_vehicle_dataset.py \
  --source-dir /path/to/road-vehicle-images-dataset \
  --out-dir data/road_vehicle_yolo
```

准备完成后会生成：

```text
data/road_vehicle_yolo/data.yaml
```

这是 YOLOv8 训练需要的数据配置文件。

## 4. 训练

冒烟测试：

```bash
python scripts/train_yolo.py \
  --data data/road_vehicle_yolo/data.yaml \
  --model yolov8n.pt \
  --epochs 1 \
  --batch 4 \
  --imgsz 416 \
  --project runs/detect \
  --name smoke_yolov8n \
  --device 0
```

正式实验：

```bash
python scripts/run_experiments.py \
  --plan configs/task2_yolo_plan.yaml \
  --data data/road_vehicle_yolo/data.yaml \
  --project runs/detect \
  --device 0 \
  --skip-existing \
  --continue-on-error
```

汇总：

```bash
python scripts/summarize_results.py --runs-dir runs/detect --out-dir results
```

## 5. 视频跟踪与越线计数

正式报告必须使用 10-30 秒测试视频。推荐你用手机拍摄校园道路、停车场入口或路口，画面中最好有车辆交汇或遮挡。

```bash
python scripts/track_video.py \
  --weights runs/detect/01_yolov8n_baseline/weights/best.pt \
  --source /path/to/video.mp4 \
  --out-dir outputs/tracking_demo \
  --line 100 360 1180 360 \
  --tracker bytetrack.yaml \
  --conf 0.25 \
  --device 0
```

输出：

```text
outputs/tracking_demo/tracked.mp4
outputs/tracking_demo/tracks.csv
outputs/tracking_demo/crossing_events.csv
outputs/tracking_demo/tracking_summary.json
```

## 6. 遮挡/ID 跳变分析

```bash
python scripts/analyze_occlusion.py \
  --track-csv outputs/tracking_demo/tracks.csv \
  --annotated-video outputs/tracking_demo/tracked.mp4 \
  --out-dir outputs/occlusion_analysis \
  --num-frames 4
```

脚本会自动查找 bbox 重叠较多的候选帧，并保存连续帧。你需要人工观看这些帧和 `tracked.mp4`，判断 ID 是否稳定。

## 7. wandb / swanlab

Ultralytics 会生成本地 `results.csv`、`results.png` 等文件。若需要补 wandb 曲线，可在训练后执行：

```bash
python scripts/log_yolo_results_to_wandb.py \
  --run-dir runs/detect/01_yolov8n_baseline \
  --project dl-hw2-task2 \
  --mode offline
```

然后可使用 `wandb sync` 上传离线日志。
