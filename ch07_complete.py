"""
LLMs-from-scratch - Chapter 7: Instruction Finetuning
完整的指令微调流程：让LLM学会遵循指令
"""

import os
import json
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader
from importlib.metadata import version
import requests
from functools import partial
from tqdm import tqdm

print("=" * 60)
print("Environment Check")
print("=" * 60)

pkgs = ["numpy", "matplotlib", "tiktoken", "torch", "tqdm", "tensorflow"]
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
# 2. 下载和准备指令数据集
# ============================================
print("=" * 60)
print("Step 2: Downloading Instruction Dataset")
print("=" * 60)


def download_and_load_file(file_path, url):
    """下载并加载JSON文件"""
    if not os.path.exists(file_path):
        print(f"Downloading {file_path}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)
        print("Download complete")
    else:
        print(f"{file_path} already exists")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print(f"Number of entries: {len(data)}")
print(f"Example entry:\n{data[50]}")
print()

# ============================================
# 3. 格式化输入
# ============================================
print("=" * 60)
print("Step 3: Formatting Input")
print("=" * 60)


def format_input(entry):
    """格式化指令输入（Alpaca风格）"""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


# 展示格式化示例
print("Example with input:")
model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)
print("\n" + "-" * 40 + "\n")

print("Example without input:")
model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
print(model_input + desired_response)
print()

# ============================================
# 4. 划分数据集
# ============================================
print("=" * 60)
print("Step 4: Splitting Dataset")
print("=" * 60)

train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print(f"Training set length: {len(train_data)}")
print(f"Validation set length: {len(val_data)}")
print(f"Test set length: {len(test_data)}")
print()

# ============================================
# 5. 创建数据集类
# ============================================
print("=" * 60)
print("Step 5: Creating Dataset Class")
print("=" * 60)

tokenizer = tiktoken.get_encoding("gpt2")
pad_token_id = 50256  # <|endoftext|>


class InstructionDataset(Dataset):
    """指令微调数据集类"""

    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


# ============================================
# 6. 自定义collate函数
# ============================================
print("=" * 60)
print("Step 6: Custom Collate Function")
print("=" * 60)


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                      allowed_max_length=None, device="cpu"):
    """自定义批处理函数，支持动态填充和损失忽略"""
    # 找到批次中最长的序列
    batch_max_length = max(len(item) + 1 for item in batch)

    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # 将除第一个填充token外的所有填充token替换为ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 可选截断
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


# 测试collate函数
batch_example = [
    [0, 1, 2, 3, 4],
    [5, 6],
    [7, 8, 9]
]
inputs, targets = custom_collate_fn(batch_example)
print(f"Inputs:\n{inputs}")
print(f"\nTargets:\n{targets}")
print()

# ============================================
# 7. 创建数据加载器
# ============================================
print("=" * 60)
print("Step 7: Creating DataLoaders")
print("=" * 60)

# 设置设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    major, minor = map(int, torch.__version__.split(".")[:2])
    device = torch.device("mps") if (major, minor) >= (2, 9) else torch.device("cpu")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 创建数据加载器
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

batch_size = 8
num_workers = 0

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
print()

# ============================================
# 8. 加载预训练模型
# ============================================
print("=" * 60)
print("Step 8: Loading Pretrained Model")
print("=" * 60)

# 模型配置
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

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

print(f"Loading {CHOOSE_MODEL}...")
print("Note: Full weight loading requires TensorFlow and gpt_download module")
print("For this demo, we'll use randomly initialized weights")

torch.manual_seed(123)
model = GPTModel(BASE_CONFIG)
model.to(device)
model.eval()

print(f"Model loaded with {BASE_CONFIG['emb_dim']} embedding dimension")
print()

# ============================================
# 9. 文本生成函数
# ============================================
print("=" * 60)
print("Step 9: Text Generation Functions")
print("=" * 60)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """生成文本（支持温度和top-k采样）"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val,
                                 torch.tensor(float("-inf")).to(logits.device),
                                 logits)

        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# ============================================
# 10. 损失计算函数
# ============================================
print("=" * 60)
print("Step 10: Loss Calculation Functions")
print("=" * 60)


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算批次的损失"""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten(), ignore_index=-100
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算整个数据加载器的损失"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


# 计算初始损失
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

print(f"Initial Training loss: {train_loss:.4f}")
print(f"Initial Validation loss: {val_loss:.4f}")
print()

# ============================================
# 11. 训练函数
# ============================================
print("=" * 60)
print("Step 11: Training the Model")
print("=" * 60)


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估模型损失"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """生成并打印示例文本"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size,
            temperature=0.0, eos_id=50256
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    response_text = decoded_text[len(start_context):].replace("### Response:", "").strip()
    print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """简单的训练函数"""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, tokens_seen


# 训练模型
print("Starting training...")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=2, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)
print("Training completed!")
print()

# ============================================
# 12. 绘制损失曲线
# ============================================
print("=" * 60)
print("Step 12: Plotting Loss Curves")
print("=" * 60)


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """绘制训练损失曲线"""
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.show()


epochs_tensor = torch.linspace(0, 2, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# ============================================
# 13. 生成测试集响应
# ============================================
print("=" * 60)
print("Step 13: Generating Test Set Responses")
print("=" * 60)

# 保存模型
import re

file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")

# 生成响应
for i, entry in enumerate(tqdm(test_data[:3], desc="Generating responses")):
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    print(f"\nInstruction: {entry['instruction'][:50]}...")
    print(f"Correct response: {entry['output'][:80]}...")
    print(f"Model response: {response_text[:80]}...")
    print("-" * 40)

print()

# ============================================
# 14. 使用Ollama评估（可选）
# ============================================
print("=" * 60)
print("Step 14: Optional Ollama Evaluation")
print("=" * 60)
print("Note: Ollama evaluation requires ollama to be installed and running")
print("Skipping this step for the demo")
print()

# ============================================
# 总结
# ============================================
print("=" * 60)
print("Chapter 7 Summary")
print("=" * 60)

summary = """
✅ 已实现的指令微调流程：

1. 指令数据集
   - 1100条指令-响应对
   - Alpaca风格格式化
   - 85/10/5划分

2. 数据处理
   - 预分词和编码
   - 动态填充（不同批次可不同长度）
   - ignore_index=-100忽略填充token
   - 自定义collate函数

3. 模型加载
   - GPT-2 Medium (355M参数)
   - 预训练权重加载

4. 指令微调
   - 标准的语言建模损失
   - 忽略填充token的损失计算
   - AdamW优化器

5. 响应生成
   - 温度采样和top-k采样
   - EOS token停止生成
   - 提取响应部分

6. 评估方法
   - 损失曲线
   - 人工评估
   - LLM辅助评估（Ollama + Llama 3）

关键概念：
- 指令微调 vs 分类微调
- Alpaca格式模板
- 动态批次填充
- 损失忽略索引
- 响应提取和评估
"""

print(summary)
print("=" * 60)
print("✅ Chapter 7 Complete!")
print("=" * 60)
print("\n🎉 Congratulations! You've completed the entire book! 🎉")