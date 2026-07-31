# 从零实现大语言模型：学习复现与实验记录

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPT%20from%20scratch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

这是一个基于 Sebastian Raschka 著作
[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
及其[配套代码](https://github.com/rasbt/LLMs-from-scratch)整理的个人学习复现仓库。
代码主体沿用原书章节结构，本仓库用于逐章运行、理解和复盘 GPT 从数据处理、
注意力机制、预训练到分类/指令微调的完整流程。

> 本仓库是学习复现，不是对原书代码的独立原创实现。原始代码、实验设计和英文
> Notebook 版权归原作者及相应贡献者所有；本仓库的个人工作主要是学习路径整理、
> 中文说明、环境固化和运行结果归档。

## 我的学习产出

- 将 Ch 2—Ch 7 的主线整理为可顺序执行的学习路径。
- 使用顶层 `ch02_complete.py`—`ch07_complete.py` 串联各章核心实现，便于脱离
  Notebook 阅读完整代码。
- 重点复盘 BPE/数据加载、因果多头注意力、GPT 主干、预训练损失、分类头微调、
  指令数据集与 LoRA。
- 保留 Notebook 的中间张量、损失、准确率和文本生成输出，便于对照代码定位每个
  阶段的输入输出。
- 通过 README 明确区分上游内容与个人学习整理，避免将原书成果误标为个人原创。

## 学习路线

| 阶段 | 目录 | 核心内容 |
|---|---|---|
| 文本处理 | `ch02/` | 分词、滑动窗口、Embedding、DataLoader |
| 注意力机制 | `ch03/` | Self-Attention、因果掩码、多头注意力 |
| GPT 主干 | `ch04/` | LayerNorm、GELU、残差连接、Transformer Block |
| 预训练 | `ch05/` | 自回归训练、损失评估、采样与权重加载 |
| 分类微调 | `ch06/` | 垃圾短信数据集、分类头、准确率评估 |
| 指令微调 | `ch07/` | 指令数据模板、监督微调、响应评估 |
| 扩展内容 | `appendix-*` | PyTorch 基础、训练技巧、LoRA |

## 仓库中可核验的运行结果

以下数据直接来自仓库已保存的 Notebook 输出，用于说明代码链路和结果文件是完整的；
它们依赖原书实验配置，不作为独立 benchmark 或排行榜成绩。

| 实验 | 已保存输出 | 证据位置 |
|---|---|---|
| 垃圾短信分类微调 | Train 97.21%、Validation 97.32%、Test 95.67% | `ch06/01_main-chapter-code/ch06.ipynb` |
| 指令微调初始评估 | Training loss 3.826、Validation loss 3.762 | `ch07/01_main-chapter-code/ch07.ipynb` |
| 指令微调过程 | 验证损失由约 2.626 下降至 0.6—0.7 区间，并保存生成回答示例 | `ch07/01_main-chapter-code/ch07.ipynb` |

重新运行时，结果会随 PyTorch 版本、随机种子、设备和预训练权重而变化。若用于严谨
对比，应同时记录硬件、依赖版本、随机种子和完整训练参数。

## 快速开始

建议使用 Python 3.10 或 3.11，并在独立虚拟环境中安装依赖：

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

运行各阶段脚本：

```bash
# GPT 主干结构
python ch04/01_main-chapter-code/gpt.py

# 预训练与文本生成
python ch05/01_main-chapter-code/gpt_train.py
python ch05/01_main-chapter-code/gpt_generate.py

# 分类微调
python ch06/01_main-chapter-code/gpt_class_finetune.py

# 指令微调
python ch07/01_main-chapter-code/gpt_instruction_finetuning.py
```

部分脚本需要下载 GPT-2 权重或额外数据集，具体依赖和运行说明请查看对应章节的
README。

## 项目结构

```text
LLMs-from-scratch/
├── ch02/ ... ch07/       # 原书主线章节与扩展实验
├── appendix-A/           # PyTorch 基础
├── appendix-D/           # 训练技巧
├── appendix-E/           # LoRA
├── ch02_complete.py      # 各章完整代码入口
├── ...
├── ch07_complete.py
├── setup/                # Python、依赖与容器环境说明
├── train.csv             # 分类微调训练集
├── validation.csv        # 分类微调验证集
└── test.csv              # 分类微调测试集
```

## 来源与许可

- 原书作者：Sebastian Raschka
- [原书配套仓库](https://github.com/rasbt/LLMs-from-scratch)
- [书籍官网](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- ISBN：9781633437166

本仓库遵循 [Apache License 2.0](LICENSE)。引用或复用代码时，请同时遵守上游
项目的版权和许可要求。
