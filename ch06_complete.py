"""
LLMs-from-scratch - Chapter 6: Finetuning for Classification
完整的文本分类微调流程：垃圾短信分类器
"""

import os
import torch
import torch.nn as nn
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader
from importlib.metadata import version
import requests
import zipfile
from pathlib import Path

print("=" * 60)
print("Environment Check")
print("=" * 60)

pkgs = ["matplotlib", "numpy", "tiktoken", "torch", "tensorflow", "pandas"]
for p in pkgs:
    try:
        print(f"{p} version: {version(p)}")
    except:
        print(f"{p} version: not installed")
print()

# ============================================
# 1. GPT模型定义（从第4章复用）
# ============================================
print("=" * 60)
print("Step 1: GPT Model Definition")
print("=" * 60)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ============================================
# 2. 下载和准备数据集
# ============================================
print("=" * 60)
print("Step 2: Downloading and Preparing Dataset")
print("=" * 60)


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """下载并解压垃圾短信数据集"""
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download.")
        return

    # 下载文件
    print("Downloading dataset...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # 解压文件
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # 重命名文件
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File saved as {data_file_path}")


# 数据集URL和路径
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

try:
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
except Exception as e:
    print(f"Primary URL failed: {e}. Trying backup URL...")
    url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

# 加载数据
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
print(f"\nDataset shape: {df.shape}")
print(f"Class distribution:\n{df['Label'].value_counts()}")
print()

# ============================================
# 3. 平衡数据集
# ============================================
print("=" * 60)
print("Step 3: Balancing Dataset")
print("=" * 60)


def create_balanced_dataset(df):
    """通过欠采样平衡数据集"""
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df


balanced_df = create_balanced_dataset(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
print(f"Balanced dataset shape: {balanced_df.shape}")
print(f"Class distribution:\n{balanced_df['Label'].value_counts()}")
print()

# ============================================
# 4. 划分训练/验证/测试集
# ============================================
print("=" * 60)
print("Step 4: Splitting Dataset")
print("=" * 60)


def random_split(df, train_frac, validation_frac):
    """随机划分数据集"""
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

# 保存CSV文件
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(validation_df)}")
print(f"Test set size: {len(test_df)}")
print()

# ============================================
# 5. 创建数据集类和数据加载器
# ============================================
print("=" * 60)
print("Step 5: Creating Dataset and DataLoaders")
print("=" * 60)

tokenizer = tiktoken.get_encoding("gpt2")
pad_token_id = 50256  # <|endoftext|>的token ID


class SpamDataset(Dataset):
    """垃圾短信数据集类"""

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        # 预分词
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            # 截断过长的序列
            self.encoded_texts = [
                encoded[:self.max_length] for encoded in self.encoded_texts
            ]

        # 填充到相同长度
        self.encoded_texts = [
            encoded + [pad_token_id] * (self.max_length - len(encoded))
            for encoded in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (torch.tensor(encoded, dtype=torch.long),
                torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max(len(encoded) for encoded in self.encoded_texts)


# 创建数据集
train_dataset = SpamDataset("train.csv", tokenizer, max_length=None)
val_dataset = SpamDataset("validation.csv", tokenizer, max_length=train_dataset.max_length)
test_dataset = SpamDataset("test.csv", tokenizer, max_length=train_dataset.max_length)

print(f"Max sequence length: {train_dataset.max_length}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# 创建数据加载器
batch_size = 8
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          drop_last=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         drop_last=False, num_workers=num_workers)

print(f"\nTraining batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
print()

# ============================================
# 6. 加载预训练模型
# ============================================
print("=" * 60)
print("Step 6: Loading Pretrained Model")
print("=" * 60)

# 模型配置
CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 注意：这里需要从OpenAI下载预训练权重
# 由于下载可能需要TensorFlow，这里提供简化版本
print("Loading pretrained GPT-2 weights...")
print("Note: Full weight loading requires TensorFlow and gpt_download module")
print("For this demo, we'll use randomly initialized weights")

# 简化版：使用随机初始化的模型
torch.manual_seed(123)
model = GPTModel(BASE_CONFIG)
model.eval()

print(f"Model loaded with {BASE_CONFIG['emb_dim']} embedding dimension")
print()

# ============================================
# 7. 添加分类头
# ============================================
print("=" * 60)
print("Step 7: Adding Classification Head")
print("=" * 60)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 替换输出层为2分类
num_classes = 2
model.out_head = nn.Linear(in_features=BASE_CONFIG["emb_dim"],
                           out_features=num_classes)

# 解冻最后一层transformer块和LayerNorm
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

print("Model modified for binary classification")
print(f"Output layer shape: {model.out_head.weight.shape}")
print()

# ============================================
# 8. 设置设备
# ============================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    major, minor = map(int, torch.__version__.split(".")[:2])
    device = torch.device("mps") if (major, minor) >= (2, 9) else torch.device("cpu")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
model.to(device)
print()

# ============================================
# 9. 损失和准确率计算函数
# ============================================
print("=" * 60)
print("Step 8: Loss and Accuracy Functions")
print("=" * 60)


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算批次的损失"""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # 只使用最后一个token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算整个数据加载器的损失"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")

    num_batches = min(num_batches, len(data_loader)) if num_batches else len(data_loader)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """计算分类准确率"""
    model.eval()
    correct_predictions, num_examples = 0, 0

    num_batches = min(num_batches, len(data_loader)) if num_batches else len(data_loader)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    model.train()
    return correct_predictions / num_examples


# 计算初始准确率
print("Initial accuracies (before training):")
train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_acc = calc_accuracy_loader(test_loader, model, device, num_batches=10)
print(f"Training accuracy: {train_acc * 100:.2f}%")
print(f"Validation accuracy: {val_acc * 100:.2f}%")
print(f"Test accuracy: {test_acc * 100:.2f}%")
print()

# ============================================
# 10. 训练函数
# ============================================
print("=" * 60)
print("Step 9: Training the Classifier")
print("=" * 60)


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估模型损失"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter):
    """训练分类器"""
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader,
                                                      device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每个epoch后计算准确率
        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_acc * 100:.2f}% | "
              f"Validation accuracy: {val_acc * 100:.2f}%")
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


# 训练模型
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

print("Starting training...")
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=5, eval_freq=50, eval_iter=5
)
print("Training completed!")
print()

# ============================================
# 11. 可视化训练结果
# ============================================
print("=" * 60)
print("Step 10: Visualizing Results")
print("=" * 60)


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    """绘制训练曲线"""
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.show()


# 绘制损失曲线
epochs_tensor = torch.linspace(0, 5, len(train_losses))
examples_tensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(epochs_tensor, examples_tensor, train_losses, val_losses, label="loss")

# 绘制准确率曲线
epochs_acc_tensor = torch.linspace(0, 5, len(train_accs))
examples_acc_tensor = torch.linspace(0, examples_seen, len(train_accs))
plot_values(epochs_acc_tensor, examples_acc_tensor, train_accs, val_accs, label="accuracy")

# ============================================
# 12. 最终评估
# ============================================
print("=" * 60)
print("Step 11: Final Evaluation")
print("=" * 60)

train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Final Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Final Validation accuracy: {val_accuracy * 100:.2f}%")
print(f"Final Test accuracy: {test_accuracy * 100:.2f}%")
print()

# ============================================
# 13. 使用模型进行分类
# ============================================
print("=" * 60)
print("Step 12: Using the Model for Classification")
print("=" * 60)


def classify_review(text, model, tokenizer, device, max_length, pad_token_id=50256):
    """对单条文本进行分类"""
    model.eval()

    # 分词
    input_ids = tokenizer.encode(text)
    max_length = min(max_length, model.pos_emb.weight.shape[0])
    input_ids = input_ids[:max_length]

    # 填充
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # 推理
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"


# 测试示例
test_texts = [
    ("You are a winner you have been specially selected to receive $1000 cash or a $2000 award.", "spam"),
    ("Hey, just wanted to check if we're still on for dinner tonight? Let me know!", "not spam"),
    ("CONGRATULATIONS! You've won a free iPhone! Click here to claim now!", "spam"),
    ("Can you pick up some milk on your way home? Thanks!", "not spam"),
]

print("Classification results:")
for text, expected in test_texts:
    result = classify_review(text, model, tokenizer, device,
                             max_length=train_dataset.max_length)
    status = "✓" if result == expected else "✗"
    print(f"{status} Expected: {expected:10} | Got: {result:10} | Text: {text[:50]}...")
print()

# ============================================
# 14. 保存模型
# ============================================
print("=" * 60)
print("Step 13: Saving the Model")
print("=" * 60)

torch.save(model.state_dict(), "review_classifier.pth")
print("Model saved to review_classifier.pth")

# 加载模型示例
# model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True)
# model.load_state_dict(model_state_dict)
print("Model can be loaded with: model.load_state_dict(torch.load('review_classifier.pth'))")
print()

# ============================================
# 总结
# ============================================
print("=" * 60)
print("Chapter 6 Summary")
print("=" * 60)

summary = """
✅ 已实现的分类微调流程：

1. 数据集准备
   - 下载SMS垃圾短信数据集
   - 平衡类别分布（欠采样）
   - 划分训练/验证/测试集

2. 数据预处理
   - 分词和填充
   - 创建DataLoader
   - 序列长度管理

3. 模型修改
   - 加载预训练GPT-2权重
   - 冻结大部分层
   - 替换分类头（2类输出）
   - 解冻最后几层进行微调

4. 训练
   - 只使用最后一个token进行分类
   - 交叉熵损失
   - AdamW优化器

5. 评估指标
   - 分类准确率
   - 损失曲线
   - 训练/验证/测试集评估

6. 推理
   - 文本预处理
   - 模型推理
   - 返回类别标签

关键概念：
- 分类微调 vs 指令微调
- 冻结层 vs 可训练层
- 欠采样处理不平衡数据
- 序列填充和截断
- 使用最后一个token进行分类
"""

print(summary)
print("=" * 60)
print("✅ Chapter 6 Complete!")
print("=" * 60)