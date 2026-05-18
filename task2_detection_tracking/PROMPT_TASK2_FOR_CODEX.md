# 给 Codex 的任务 2 专用提示词

请在当前 `task2_detection_tracking` 项目中完成 Road Vehicle Images Dataset + YOLOv8 + 视频多目标跟踪实验。先阅读 `README_task2.md`、`configs/task2_yolo_plan.yaml` 和 `scripts/*.py`。

必须完成：

1. 检查 GPU 和环境：`python scripts/check_env.py`。
2. 准备 Road Vehicle Images Dataset：优先 `python scripts/prepare_road_vehicle_dataset.py --kaggle --out-dir data/road_vehicle_yolo`；如果 Kaggle 不可用，使用用户上传的数据目录并运行 `--source-dir`。
3. 先跑 1 epoch 冒烟测试，确认 `runs/detect/smoke_yolov8n/weights/best.pt` 和 `results.csv` 生成。
4. 按 `configs/task2_yolo_plan.yaml` 跑正式 YOLOv8 实验：`python scripts/run_experiments.py --data data/road_vehicle_yolo/data.yaml --project runs/detect --device 0 --skip-existing --continue-on-error`。
5. 汇总结果：`python scripts/summarize_results.py --runs-dir runs/detect --out-dir results`。
6. 使用最佳 `best.pt` 对用户准备的 10-30 秒真实道路/校园视频运行 tracking：`python scripts/track_video.py ...`。
7. 调整 `--line x1 y1 x2 y2`，使虚拟线位于车辆会经过的位置。
8. 运行 `python scripts/analyze_occlusion.py` 截取连续 3-4 帧遮挡或密集交汇候选片段。
9. 最后给出：最佳检测模型、mAP50/mAP50-95、tracking 视频路径、越线计数、遮挡截帧路径、报告还缺哪些人工信息。

不要编造任何结果；如果缺少真实视频，请说明 tracking 正式实验尚未完成，不能用 demo 视频冒充正式视频。
