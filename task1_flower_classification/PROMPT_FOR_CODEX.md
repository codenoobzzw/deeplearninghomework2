# 给服务器上 Codex 的执行提示词

你现在位于一个深度学习课程期中作业任务 1 项目中。请你作为实验执行助手，完整完成 102 Category Flower Dataset 图像分类微调实验：数据准备、GPU 环境检查、训练实验、日志记录、结果汇总、报告素材整理。请优先阅读 `README.md`、`configs/experiment_plan.yaml` 和所有 `scripts/*.py`。

## 一、任务背景与必须完成的实验

作业任务 1 的核心要求是：在 102 Category Flower Dataset 上，用 ImageNet 预训练的 ResNet-18 或 ResNet-34 作为 Baseline，修改输出层为 102 类分类器；新输出层从零训练，backbone 用较小学习率微调；观察不同训练步数、学习率和组合的影响；做预训练消融实验，即随机初始化训练并与预训练对比；在 Baseline 基础上加入注意力模块，例如 SE-block 或 CBAM，并比较 Accuracy。

本项目已经实现：

1. `resnet18`、`resnet34` baseline。
2. `se_resnet18`、`se_resnet34`：手动插入 SE-block。
3. `cbam_resnet18`、`cbam_resnet34`：手动插入 CBAM。
4. `scripts/train.py`：单实验训练、验证、测试、保存 best checkpoint。
5. `scripts/run_experiments.py`：按 YAML 顺序跑实验计划。
6. `scripts/summarize_results.py`：汇总 CSV、Markdown 和曲线图。
7. 支持 `wandb` / `swanlab`，也支持本地日志。

请不要把任务改成宠物数据集。作业标题中有“宠物识别”字样，但正文明确要求使用 `102 Category Flower Dataset`，因此本项目按 Flowers102 执行。

## 二、首先检查环境

在服务器终端执行：

```bash
pwd
ls -la
python --version
nvidia-smi
```

创建或激活环境。如果没有环境，执行：

```bash
conda create -n dl_hw2_task1 python=3.10 -y
conda activate dl_hw2_task1
python -m pip install --upgrade pip
```

根据服务器 CUDA 版本安装 PyTorch。若服务器已有可用 PyTorch，可跳过。然后执行：

```bash
pip install -r requirements.txt
python scripts/check_env.py
```

必须确认：

- `torch cuda available: True`。
- 能看到 GPU 名称。
- `torchvision`、`scipy`、`pandas`、`matplotlib` 可 import。

如果 `torch cuda available: False`，请根据 `nvidia-smi` 显示的 CUDA / driver 情况重新安装匹配版本 PyTorch，然后再次运行 `python scripts/check_env.py`。

## 三、准备数据

执行：

```bash
python scripts/prepare_data.py --data-dir data
```

确认输出中有 train / val / test 三个 split，并且写入：

```text
data/flowers102_info.json
```

如果服务器无法下载数据，请提示我上传数据，或者尝试手动下载 Oxford Flowers102 数据并放到 `data/` 目录。数据准备成功后再继续训练。

## 四、先做冒烟测试

为了避免一次性跑很久才发现错误，请先运行 1 个 epoch 小 batch 测试：

```bash
python scripts/run_experiments.py \
  --epochs-override 1 \
  --batch-size-override 8 \
  --tracker none \
  --amp
```

冒烟测试通过的标准：

- 每个实验能进入 train 和 eval。
- `runs/<exp-name>/` 下出现 `config.json`、`history.csv`、`metrics_best.json`、`best.pt` 或 `last.pt`。
- 没有 CUDA OOM。

如果 CUDA OOM，请把 batch size 降到 16、8 或 4。请不要直接放弃。

## 五、正式实验计划

正式实验使用 `configs/experiment_plan.yaml`。默认包括：

1. `01_baseline_resnet18_pretrained_lr1e3_1e4`：Baseline，ImageNet 预训练，head lr=1e-3，backbone lr=1e-4。
2. `02_hparam_resnet18_pretrained_lr3e4_3e5`：学习率较小。
3. `03_hparam_resnet18_pretrained_lr3e3_3e4`：学习率较大。
4. `04_hparam_resnet18_pretrained_50ep`：训练 epoch 更多。
5. `05_ablation_resnet18_random_init`：随机初始化消融。
6. `06_attention_se_resnet18_pretrained`：SE 注意力。
7. `07_attention_cbam_resnet18_pretrained`：CBAM 注意力。
8. `08_optional_resnet34_pretrained`：可选 ResNet-34。

如果时间有限，至少完成 1、2、3、5、6、7。第 4 和第 8 个实验可作为补充。

## 六、正式运行命令

如果使用 wandb 离线模式：

```bash
python scripts/run_experiments.py \
  --plan configs/experiment_plan.yaml \
  --data-dir data \
  --output-dir runs \
  --tracker wandb \
  --wandb-mode offline \
  --num-workers 4 \
  --amp \
  --skip-existing \
  --continue-on-error
```

如果我已经告诉你要使用 swanlab，则改成：

```bash
python scripts/run_experiments.py \
  --plan configs/experiment_plan.yaml \
  --data-dir data \
  --output-dir runs \
  --tracker swanlab \
  --num-workers 4 \
  --amp \
  --skip-existing \
  --continue-on-error
```

如果显存不足，统一降 batch size：

```bash
python scripts/run_experiments.py \
  --plan configs/experiment_plan.yaml \
  --data-dir data \
  --output-dir runs \
  --tracker wandb \
  --wandb-mode offline \
  --num-workers 4 \
  --batch-size-override 16 \
  --amp \
  --skip-existing \
  --continue-on-error
```

如果服务器有多张 GPU，但只想使用第 0 张：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py --tracker wandb --wandb-mode offline --amp --skip-existing
```

## 七、实验过程中你要记录的东西

请维护一个简单运行记录文件，例如 `results/run_notes.md`，每跑完一个实验写下：

```text
实验名：
命令：
开始/结束时间：
是否成功：
best_val_acc1：
test_acc1：
best checkpoint：
异常或备注：
```

如果出现报错，请先分析并修复。例如：

- CUDA OOM：降低 batch size，继续。
- 数据集下载失败：检查网络，或提示上传数据。
- wandb 初始化失败：改用 `--wandb-mode offline` 或 `--tracker none`，本地结果仍可汇总。
- 某一个实验失败：不要影响其他实验，使用 `--continue-on-error` 继续跑，最后在记录里说明。

## 八、结果汇总

所有实验完成后执行：

```bash
python scripts/summarize_results.py --runs-dir runs --out-dir results
```

请检查以下文件是否生成：

```text
results/summary.csv
results/summary.md
results/summary_bar.png
results/val_accuracy_curves.png
results/train_loss_curves.png
results/val_loss_curves.png
```

然后阅读 `results/summary.md`，找出最佳模型，并写一个简短结论，至少包括：

1. Baseline 的 best val acc 和 test acc。
2. 学习率变化对性能的影响。
3. 预训练 vs 随机初始化的差异。
4. SE / CBAM 注意力是否比 Baseline 更好。
5. 最终推荐上传哪个 `best.pt`。

## 九、报告素材整理

请为实验报告准备这些素材：

1. `results/summary.csv` 或 Markdown 表格。
2. `results/val_accuracy_curves.png`。
3. `results/train_loss_curves.png`。
4. `results/val_loss_curves.png`。
5. wandb 或 swanlab 中每个关键实验的训练/验证 loss 和 val accuracy 截图。
6. 最佳模型的 `best.pt` 路径。
7. GitHub repo 链接占位符：等我上传后填写。
8. 模型权重网盘链接占位符：等我上传后填写。

请使用 `report_template.md` 作为报告草稿模板，把运行出来的数据填进去。不要编造实验结果，必须以 `runs/*/metrics_best.json` 和 `results/summary.csv` 为准。

## 十、最终交付给我的内容

请你最后给我一个清单，包含：

- 环境是否配置成功。
- 数据集是否准备成功。
- 跑完了哪些实验，哪些失败了，失败原因是什么。
- `results/summary.md` 的主要结论。
- 最佳模型权重路径，例如 `runs/xx/best.pt`。
- 需要我上传 GitHub / 网盘的文件。
- 报告还缺哪些人工信息，例如姓名、学号、组员分工、GitHub 链接、网盘链接。

重要原则：

- 不要凭空填写结果。
- 不要把 train/val/test split 搞混。
- 不要把任务改成宠物数据集。
- 如果发生 OOM，优先降低 batch size。
- 如果没有 wandb/swanlab 截图，至少保留本地曲线图，并说明需要补截图。
