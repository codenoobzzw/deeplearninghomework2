# 任务 2 报告草稿模板：场景目标检测与视频多目标跟踪

> 请把“待填写”替换为真实信息。不要编造结果，所有数值以 `results/summary.csv` 和 tracking 输出文件为准。

## 1. 任务简介

本任务使用 Road Vehicle Images Dataset 微调 YOLOv8 单阶段目标检测模型，得到车辆检测模型。随后使用训练好的模型对一段 10-30 秒道路/校园视频逐帧推理，并结合 YOLOv8 内置 tracking，为同一目标分配 Tracking ID。同时，基于检测框中心点和虚拟线的位置关系实现越线计数，并对遮挡或密集交汇片段进行 ID 稳定性分析。

## 2. 数据集

- 数据集：Road Vehicle Images Dataset
- 训练集数量：待填写
- 验证集数量：待填写
- 测试集数量：待填写
- 类别名称：待填写
- 标注格式：YOLO txt 格式，每行包括 class、x_center、y_center、width、height，坐标为归一化值。

## 3. 模型与方法

- 检测模型：YOLOv8n / YOLOv8s
- 初始化方式：使用 YOLOv8 COCO 预训练权重进行微调
- 输入分辨率：640
- 优化器与训练策略：Ultralytics 默认优化器和增强策略
- 评价指标：Precision、Recall、mAP@0.50、mAP@0.50:0.95

## 4. 实验设置

| 实验名 | 模型 | epoch | batch | imgsz | lr0 | lrf | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| 01_yolov8n_baseline | YOLOv8n | 80 | 16 | 640 | 0.01 | 0.01 | Baseline |
| 02_yolov8n_low_lr | YOLOv8n | 80 | 16 | 640 | 0.003 | 0.01 | 学习率对比 |
| 03_yolov8n_more_epochs | YOLOv8n | 120 | 16 | 640 | 0.01 | 0.01 | 训练轮数对比 |
| 04_yolov8s_baseline | YOLOv8s | 80 | 8 | 640 | 0.01 | 0.01 | 更大模型对比 |

## 5. 检测实验结果

> 从 `results/summary.md` 或 `results/summary.csv` 复制结果表。

| 实验名 | best epoch | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 | 权重路径 |
|---|---:|---:|---:|---:|---:|---|
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

## 6. 训练曲线

插入以下图：

- `results/map50_curves.png`
- `results/map5095_curves.png`
- `results/loss_curves.png`
- `results/summary_bar.png`

## 7. 视频多目标跟踪

- 测试视频路径：待填写
- 视频时长：待填写，需为 10-30 秒
- 跟踪算法：ByteTrack / BoT-SORT
- 检测置信度阈值：待填写
- 输出视频：`outputs/tracking_demo/tracked.mp4`
- Tracking ID 输出：`outputs/tracking_demo/tracks.csv`

插入视频关键帧截图，展示 Bounding Box、类别与 Tracking ID。

## 8. 越线计数

虚拟线设置：

```text
(x1, y1) = 待填写
(x2, y2) = 待填写
```

统计逻辑：对每个 Tracking ID 保存上一帧检测框中心点相对虚拟线的符号，如果当前帧符号与上一帧相反，则认为该目标跨越虚拟线。每个 Tracking ID 只计数一次，避免来回抖动重复计数。

- 跨线物体总数：待填写
- 事件文件：`outputs/tracking_demo/crossing_events.csv`
- 汇总文件：`outputs/tracking_demo/tracking_summary.json`

## 9. 遮挡与 ID 跳变分析

插入 `outputs/occlusion_analysis/frame_*.jpg` 中连续 3-4 帧。

请填写：

1. 遮挡/密集交汇发生在哪些帧：待填写
2. 涉及的 Tracking ID：待填写
3. 遮挡前后 ID 是否保持：待填写
4. 是否发生目标丢失或 ID switch：待填写
5. 原因分析：待填写

可参考分析：多目标跟踪通常将检测框与已有轨迹进行关联。ByteTrack 会利用高置信度和低置信度检测框进行两阶段关联，在部分遮挡导致置信度下降时，仍有机会维持轨迹；但如果遮挡过长、目标外观或位置变化过大、框重叠严重，关联会失败，导致 ID 丢失或重新分配。

## 10. 结论

待填写：总结哪个 YOLO 实验最佳、tracking 是否稳定、越线计数是否合理、遮挡场景中是否发生 ID 跳变。

## 11. 提交信息

- GitHub repo 链接：待填写
- 模型权重网盘链接：待填写
- 小组成员与分工：待填写
