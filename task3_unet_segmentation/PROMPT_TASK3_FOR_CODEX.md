# 给 Codex 的任务 3 专用提示词

请在当前 `task3_unet_segmentation` 项目中完成 Stanford Background Dataset + 从零手写 U-Net + CE/Dice/CE+Dice 对比实验。先阅读 `README_task3.md`、`configs/task3_unet_plan.yaml` 和 `scripts/*.py`。

必须完成：

1. 检查 GPU 和环境：`python scripts/check_env.py`。
2. 准备 Stanford Background Dataset：优先 `python scripts/prepare_stanford_background.py --kaggle --out-dir data/stanford_background`；如果 Kaggle 不可用，使用用户上传的数据目录并运行 `--source-dir`。
3. 确认 `data/stanford_background/dataset_info.json`、`splits/train.txt`、`splits/val.txt`、`processed/masks/*.png` 存在。
4. 先跑 1 epoch 冒烟测试：`python scripts/train.py --data-dir data/stanford_background --loss ce --epochs 1 --batch-size 2 --image-size 256 --base-channels 16 --output-dir runs/smoke_ce --amp --tracker none`。
5. 按 `configs/task3_unet_plan.yaml` 跑三组正式实验：CE、Dice、CE+Dice。命令：`python scripts/run_experiments.py --data-dir data/stanford_background --runs-dir runs --tracker wandb --wandb-mode offline --amp --skip-existing --continue-on-error`。
6. 汇总结果：`python scripts/summarize_results.py --runs-dir runs --out-dir results`。
7. 使用验证集 mIoU 最高的 `best.pt` 生成预测可视化：`python scripts/visualize_predictions.py --checkpoint runs/<best-exp>/best.pt --data-dir data/stanford_background --split val --out-dir results/predictions --num-samples 8`。
8. 最后给出：三种 loss 的 best val mIoU、pixel accuracy、最佳权重路径、曲线和预测图路径、报告还缺哪些人工信息。

任务 3 严禁使用任何预训练权重；Dice Loss 必须使用 `scripts/losses.py` 中手动实现的版本。
