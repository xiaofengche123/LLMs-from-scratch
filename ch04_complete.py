"""
LLMs-from-scratch - Chapter 4: Implementing a GPT Model from Scratch
完整的GPT模型实现：从配置到文本生成
"""

import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from importlib.metadata import version

print("=" * 60)
print("Environment Check")
print("=" * 60)
print(f"matplotlib version: {version('matplotlib')}")
print(f"torch version: {version('torch')}")
print(f"tiktoken version: {version('tiktoken')}")
print()

# ============================================
# 1. GPT模型配置
# ============================================
print("=" * 60)
print("Step 1: GPT Model Configuration")
print("=" * 60)

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,  # 嵌入维度
    "n_heads": 12,  # 注意力头数
    "n_layers": 12,  # 层数
    "drop_rate": 0.1,  # Dropout比率
    "qkv_bias": False  # QKV偏置
}

print("GPT-2 124M Configuration:")
for key, value in GPT_CONFIG_124M.items():
    print(f"  {key}: {value}")
print()

# ============================================
# 2. 层归一化 (Layer Normalization)
# ============================================
print("=" * 60)
print("Step 2: Layer Normalization")
print("=" * 60)


class LayerNorm(nn.Module):
    """层归一化实现"""

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


# 演示层归一化
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)

ln = LayerNorm(emb_dim=6)
out_ln = ln(out)

print(f"Input shape: {batch_example.shape}")
print(f"After linear+ReLU: {out.shape}")
print(f"After LayerNorm: {out_ln.shape}")
print(f"Mean after LayerNorm: {out_ln.mean(dim=-1, keepdim=True)[0, 0].item():.4f}")
print(f"Var after LayerNorm: {out_ln.var(dim=-1, keepdim=True)[0, 0].item():.4f}")
print()

# ============================================
# 3. GELU激活函数
# ============================================
print("=" * 60)
print("Step 3: GELU Activation Function")
print("=" * 60)


class GELU(nn.Module):
    """GELU激活函数实现"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# 可视化GELU和ReLU
gelu, relu = GELU(), nn.ReLU()
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()
print()

# ============================================
# 4. 前馈网络 (FeedForward)
# ============================================
print("=" * 60)
print("Step 4: FeedForward Network")
print("=" * 60)


class FeedForward(nn.Module):
    """前馈神经网络模块"""

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


ffn = FeedForward(GPT_CONFIG_124M)
x_test = torch.rand(2, 3, GPT_CONFIG_124M["emb_dim"])
out_ffn = ffn(x_test)
print(f"Input shape: {x_test.shape}")
print(f"Output shape: {out_ffn.shape}")
print()

# ============================================
# 5. 多头注意力 (MultiHeadAttention)
# ============================================
print("=" * 60)
print("Step 5: Multi-Head Attention")
print("=" * 60)


class MultiHeadAttention(nn.Module):
    """高效的多头注意力实现"""

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


# 测试多头注意力
mha = MultiHeadAttention(
    d_in=GPT_CONFIG_124M["emb_dim"],
    d_out=GPT_CONFIG_124M["emb_dim"],
    context_length=GPT_CONFIG_124M["context_length"],
    dropout=GPT_CONFIG_124M["drop_rate"],
    num_heads=GPT_CONFIG_124M["n_heads"],
    qkv_bias=GPT_CONFIG_124M["qkv_bias"]
)
x_attn = torch.rand(2, 4, GPT_CONFIG_124M["emb_dim"])
out_attn = mha(x_attn)
print(f"Input shape: {x_attn.shape}")
print(f"Output shape: {out_attn.shape}")
print()

# ============================================
# 6. Transformer块
# ============================================
print("=" * 60)
print("Step 6: Transformer Block")
print("=" * 60)


class TransformerBlock(nn.Module):
    """Transformer块，结合注意力和前馈网络"""

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
        # 注意力块的快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 前馈网络的快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


# 测试Transformer块
torch.manual_seed(123)
x_trans = torch.rand(2, 4, GPT_CONFIG_124M["emb_dim"])
block = TransformerBlock(GPT_CONFIG_124M)
out_trans = block(x_trans)
print(f"Input shape: {x_trans.shape}")
print(f"Output shape: {out_trans.shape}")
print()

# ============================================
# 7. 完整的GPT模型
# ============================================
print("=" * 60)
print("Step 7: Complete GPT Model")
print("=" * 60)


class GPTModel(nn.Module):
    """完整的GPT模型实现"""

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


# 实例化模型
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

# 准备输入数据
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

print(f"Input batch shape: {batch.shape}")
print(f"Input batch:\n{batch}")

# 前向传播
out = model(batch)
print(f"\nOutput shape: {out.shape}")
print(f"Output logits (first 5 values): {out[0, 0, :5]}")
print()

# ============================================
# 8. 模型参数统计
# ============================================
print("=" * 60)
print("Step 8: Model Parameter Statistics")
print("=" * 60)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# 计算考虑权重 tying 后的参数
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Parameters with weight tying: {total_params_gpt2:,}")

# 计算模型大小
total_size_mb = total_params * 4 / (1024 * 1024)
print(f"Model size (float32): {total_size_mb:.2f} MB")

print("\n其他GPT-2配置:")
print("  GPT2-medium: emb_dim=1024, n_layers=24, n_heads=16")
print("  GPT2-large: emb_dim=1280, n_layers=36, n_heads=20")
print("  GPT2-XL: emb_dim=1600, n_layers=48, n_heads=25")
print()

# ============================================
# 9. 文本生成
# ============================================
print("=" * 60)
print("Step 9: Text Generation")
print("=" * 60)


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """简单的贪心解码文本生成"""
    for _ in range(max_new_tokens):
        # 如果超过上下文长度，只使用最后context_size个token
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步
        logits = logits[:, -1, :]

        # 获取概率最高的token
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # 添加到序列
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# 准备输入
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print(f"Encoded: {encoded}")
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(f"Encoded tensor shape: {encoded_tensor.shape}")

# 生成文本
model.eval()  # 禁用dropout
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print(f"\nOutput tokens: {out[0].tolist()}")
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(f"Generated text: {decoded_text}")
print("\n注意: 模型未训练，输出是随机的")
print()

# ============================================
# 10. 快捷连接演示
# ============================================
print("=" * 60)
print("Step 10: Shortcut Connections Demo")
print("=" * 60)


class ExampleDeepNeuralNetwork(nn.Module):
    """演示快捷连接效果的深度网络"""

    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    """打印模型梯度信息"""
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()(output, target)
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            grad_mean = param.grad.abs().mean().item()
            print(f"{name} gradient mean: {grad_mean:.6f}")


layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])

print("Without shortcut connections:")
torch.manual_seed(123)
model_without = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
print_gradients(model_without, sample_input)

print("\nWith shortcut connections:")
torch.manual_seed(123)
model_with = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with, sample_input)
print()

# ============================================
# 总结
# ============================================
print("=" * 60)
print("Chapter 4 Summary")
print("=" * 60)

summary = """
✅ 已实现的GPT模型组件：

1. 层归一化 (LayerNorm)
   - 稳定训练，加速收敛
   - 包含可训练的scale和shift参数

2. GELU激活函数
   - 平滑的非线性激活函数
   - GPT-2使用的标准激活函数

3. 前馈网络 (FeedForward)
   - 扩展维度到4倍，再压缩回来
   - 使用GELU激活

4. 多头注意力 (MultiHeadAttention)
   - 从第3章引入的高效实现
   - 支持因果掩码

5. Transformer块
   - 结合注意力和前馈网络
   - 使用层归一化和快捷连接
   - 防止梯度消失

6. 完整的GPT模型
   - 124M参数配置
   - 支持文本生成

关键特性：
- 快捷连接（残差连接）：防止梯度消失
- 层归一化：稳定训练
- 因果注意力：确保自回归生成
- Dropout：防止过拟合

模型大小：
- 124M参数（考虑weight tying）
- 约622 MB（float32）
"""

print(summary)
print("=" * 60)
print("✅ Chapter 4 Complete!")
print("=" * 60)