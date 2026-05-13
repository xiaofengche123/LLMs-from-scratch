# 从零实现大语言模型 - 学习实践

本项目记录了我**从零开发、预训练和微调 GPT 模型**的个人学习过程，基于 Sebastian Raschka 的著作 [Build a Large Language Model (From Scratch)](http://mng.bz/orYv)。



<br>
<br>

<a href="https://github.com/xiaofengche123/LLMs-from-scratch"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="250px"></a>

<br>

在这个学习项目中，我**一步步构建了一个完整的 GPT 模型**，从分词器到预训练再到微调。目标不仅是跑通代码，而是**理解每一行代码的含义** —— 包括多头注意力、因果掩码、残差连接、层归一化和 LoRA 微调等。

本仓库包含我添加注释的代码、训练日志和生成的文本示例。

- 我的学习仓库：[xiaofengche123/LLMs-from-scratch](https://github.com/xiaofengche123/LLMs-from-scratch)
- 原书代码库：[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [书籍官网](http://mng.bz/orYv)
- ISBN 9781633437166

<br>

## 我学习和实现的内容

| 章节 | 内容 | 我的理解 |
|------|------|----------|
| Ch 2 | 文本数据处理 | BPE 分词器的实现原理 |
| Ch 3 | 编码注意力机制 | 因果多头注意力的代码实现 |
| Ch 4 | 从零实现 GPT 模型 | GPT 架构：嵌入层 → Transformer块 → 输出层 |
| Ch 5 | 无监督预训练 | 自回归训练 + 损失计算 |
| Ch 6 | 文本分类微调 | 替换分类头进行情感分析 |
| Ch 7 | 指令微调 | 让模型理解并遵循指令 |
| Appx A | PyTorch 基础 | 张量操作与自动微分 |
| Appx D | 训练技巧 | 学习率调度、梯度裁剪 |
| Appx E | LoRA 微调 | 参数高效的低秩适配器 |

<br>

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行预训练示例
python ch05/01_main-chapter-code/gpt_train.py

# 3. 运行文本生成
python ch05/01_main-chapter-code/gpt_generate.py

# 4. 运行分类微调
python ch06/01_main-chapter-code/gpt_class_finetune.py
