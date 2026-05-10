"""
LLMs-from-scratch - Chapter 5: Pretraining on Unlabeled Data
完整的GPT模型训练流程：训练、评估、文本生成和加载预训练权重
"""

import os
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from importlib.metadata import version
import urllib.request

print("=" * 60)
print("Environment Check")
print("=" * 60)

pkgs = ["matplotlib", "numpy", "tiktoken", "torch", "tensorflow"]
for p in pkgs:
    try:
        print(f"{p} version: {version(p)}")
    except:
        print(f"{p} version: not installed")
print()

# ============================================
# 1. GPT模型配置和定义（从第4章复用）
# ============================================
print("=" * 60)
print("Step 1: GPT Model Configuration")
print("=" * 60)

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 256,  # 缩短的上下文长度（原始为1024）
    "emb_dim": 768,  # 嵌入维度
    "n_heads": 12,  # 注意力头数
    "n_layers": 12,  # 层数
    "drop_rate": 0.2,  # 更强正则化：提高Dropout比率
    "qkv_bias": False  # QKV偏置
}

print("GPT-2 124M Configuration (training):")
for key, value in GPT_CONFIG_124M.items():
    print(f"  {key}: {value}")
print()


# ============================================
# 2. 层归一化
# ============================================
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


# ============================================
# 3. GELU激活函数
# ============================================
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# ============================================
# 4. 前馈网络
# ============================================
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


# ============================================
# 5. 多头注意力
# ============================================
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


# ============================================
# 6. Transformer块
# ============================================
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


# ============================================
# 7. 完整GPT模型
# ============================================
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
# 8. 数据加载器（从第2章复用）
# ============================================
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers
    )
    return dataloader


# ============================================
# 9. 文本生成函数
# ============================================
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# ============================================
# 10. 损失计算函数
# ============================================
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
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


# ============================================
# 11. 训练函数
# ============================================
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        num_epochs,
        eval_freq,
        eval_iter,
        start_context,
        tokenizer,
        early_stopping_patience=5):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float("inf")
    epochs_without_improvement = 0

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
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每个epoch结束后执行一次学习率调度和早停判断
        epoch_train_loss, epoch_val_loss = evaluate_model(
            model, train_loader, val_loader, device, eval_iter
        )
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1} end: train {epoch_train_loss:.3f}, "
              f"val {epoch_val_loss:.3f}, lr {current_lr:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No val improvement for {epochs_without_improvement}/"
                  f"{early_stopping_patience} epochs")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            generate_and_print_sample(model, tokenizer, device, start_context)
            break

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


# ============================================
# 12. 加载数据集
# ============================================
print("=" * 60)
print("Step 2: Loading Dataset")
print("=" * 60)

# 下载或加载数据文件
file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    print("Downloading dataset...")
    response = urllib.request.urlopen(url)
    text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
    print("Download complete")
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    print("Dataset already exists")

print(f"Total characters: {len(text_data):,}")

# 初始化分词器
tokenizer = tiktoken.get_encoding("gpt2")
total_tokens = len(tokenizer.encode(text_data))
print(f"Total tokens: {total_tokens:,}")
print()

# ============================================
# 13. 创建训练和验证数据加载器
# ============================================
print("=" * 60)
print("Step 3: Creating Data Loaders")
print("=" * 60)

# 划分训练/验证集
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print(f"Training tokens: {sum(batch[0].numel() for batch in train_loader)}")
print(f"Validation tokens: {sum(batch[0].numel() for batch in val_loader)}")
print()

# ============================================
# 14. 设置设备
# ============================================
print("=" * 60)
print("Step 4: Setting up Device")
print("=" * 60)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

print(f"Using {device} device")
print()

# ============================================
# 15. 初始损失计算
# ============================================
print("=" * 60)
print("Step 5: Initial Loss Calculation")
print("=" * 60)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print(f"Initial Training loss: {train_loss:.4f}")
print(f"Initial Validation loss: {val_loss:.4f}")
print()

# ============================================
# 16. 训练模型
# ============================================
print("=" * 60)
print("Step 6: Training the Model")
print("=" * 60)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

num_epochs = 10
print(f"Training for {num_epochs} epochs...")
print()

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, scheduler, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer,
    early_stopping_patience=5
)
print()

# ============================================
# 17. 绘制损失曲线
# ============================================
print("=" * 60)
print("Step 7: Plotting Loss Curves")
print("=" * 60)


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
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


epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# ============================================
# 18. 高级文本生成（带温度缩放和top-k采样）
# ============================================
print("=" * 60)
print("Step 8: Advanced Text Generation")
print("=" * 60)


def generate(
        model,
        idx,
        max_new_tokens,
        context_size,
        temperature=1.0,
        top_k=None,
        top_p=0.9,
        eos_id=None):
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

        if temperature <= 0.0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus_mask = cumulative_probs <= top_p
                nucleus_mask[..., 0] = True
                filtered_probs = torch.where(
                    nucleus_mask,
                    sorted_probs,
                    torch.zeros_like(sorted_probs)
                )
                filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
                next_sorted_idx = torch.multinomial(filtered_probs, num_samples=1)
                idx_next = sorted_indices.gather(-1, next_sorted_idx)
            else:
                idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# 使用CPU进行推理
inference_device = torch.device("cpu")
model.to(inference_device)
model.eval()

torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("Generated text (with temperature + top-p):")
print(token_ids_to_text(token_ids, tokenizer))
print()


def interactive_chat(
        model,
        tokenizer,
        device,
        context_size,
        max_new_tokens=120,
        temperature=0.9,
        top_p=0.9):
    print("=" * 60)
    print("Interactive Chat (输入 exit 退出)")
    print("=" * 60)
    model.eval()
    dialogue_history = ""

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Chat ended.")
            break
        if not user_input:
            continue

        dialogue_history += f"\nUser: {user_input}\nAssistant:"
        prompt_ids = text_to_token_ids(dialogue_history, tokenizer).to(device)

        with torch.no_grad():
            output_ids = generate(
                model=model,
                idx=prompt_ids,
                max_new_tokens=max_new_tokens,
                context_size=context_size,
                temperature=temperature,
                top_p=top_p
            )

        generated_text = token_ids_to_text(output_ids, tokenizer)
        assistant_reply = generated_text[len(dialogue_history):].strip().split("\nUser:")[0].strip()
        if not assistant_reply:
            assistant_reply = "I am not sure yet. Could you rephrase that?"

        print(f"Assistant: {assistant_reply}")
        dialogue_history += f" {assistant_reply}"

# ============================================
# 19. 保存和加载模型权重
# ============================================
print("=" * 60)
print("Step 9: Saving and Loading Model Weights")
print("=" * 60)

# 保存模型
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")

# 保存模型和优化器状态
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "model_and_optimizer.pth")
print("Model and optimizer saved to model_and_optimizer.pth")

# 加载模型
new_model = GPTModel(GPT_CONFIG_124M)
new_model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
new_model.eval()
print("Model loaded from model.pth")
print()

# 启动终端交互式对话
interactive_chat(
    model=model,
    tokenizer=tokenizer,
    device=inference_device,
    context_size=GPT_CONFIG_124M["context_length"],
    max_new_tokens=120,
    temperature=0.9,
    top_p=0.9
)

# ============================================
# 20. 加载OpenAI预训练权重
# ============================================
print("=" * 60)
print("Step 10: Loading Pretrained Weights from OpenAI")
print("=" * 60)

print("Note: Loading pretrained weights requires TensorFlow installation")
print("This step is optional and can be skipped if TensorFlow is not available")

# 预训练模型配置
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

# 可选：下载和加载预训练权重（需要TensorFlow）
# from gpt_download import download_and_load_gpt2
# settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
# gpt = GPTModel(BASE_CONFIG)
# load_weights_into_gpt(gpt, params)

print("Pretrained weights loading is optional for this demo")
print()

# ============================================
# 总结
# ============================================
print("=" * 60)
print("Chapter 5 Summary")
print("=" * 60)

summary = """
✅ 已实现的训练流程组件：

1. 损失计算
   - 交叉熵损失 (Cross-entropy loss)
   - 困惑度 (Perplexity = exp(loss))

2. 训练循环
   - 批量损失计算
   - 优化器更新（AdamW）
   - 周期性验证评估

3. 文本生成策略
   - 贪心解码 (argmax)
   - 温度缩放 (Temperature scaling)
   - Top-k采样 (Top-k sampling)

4. 模型持久化
   - 保存/加载模型权重
   - 保存/加载优化器状态

5. 预训练权重加载
   - 从OpenAI加载GPT-2权重
   - 权重映射和转换

关键概念：
- 交叉熵损失：衡量预测分布与真实分布的差异
- 困惑度：模型不确定性的度量
- 温度缩放：控制生成的随机性
- Top-k采样：限制采样候选集大小
- 权重 tying：复用token嵌入作为输出层
"""

print(summary)
print("=" * 60)
print("✅ Chapter 5 Complete!")
print("=" * 60)