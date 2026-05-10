"""
LLMs-from-scratch - Chapter 3: Coding Attention Mechanisms
完整的注意力机制实现：从简单自注意力到多头因果注意力
"""

import torch
import torch.nn as nn
from importlib.metadata import version

print("=" * 60)
print("Environment Check")
print("=" * 60)
print(f"PyTorch version: {version('torch')}")
print()

# ============================================
# 1. 准备输入数据
# ============================================
print("=" * 60)
print("Step 1: Preparing Input Data")
print("=" * 60)

# 6个输入token，每个token是3维嵌入向量
# 对应句子: "Your journey starts with one step"
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)

print(f"Input shape: {inputs.shape}")
print(f"Number of tokens: {inputs.shape[0]}, Embedding dimension: {inputs.shape[1]}")
print()

# ============================================
# 2. 简单自注意力（无可训练权重）
# ============================================
print("=" * 60)
print("Step 2: Simple Self-Attention (No Trainable Weights)")
print("=" * 60)


def simple_self_attention(inputs):
    """简单的自注意力机制演示"""
    # Step 1: 计算注意力分数（点积）
    attn_scores = inputs @ inputs.T
    print(f"Attention scores shape: {attn_scores.shape}")

    # Step 2: 使用softmax归一化
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # Step 3: 计算上下文向量
    context_vecs = attn_weights @ inputs

    return attn_scores, attn_weights, context_vecs


# 计算第二个token的上下文向量
query = inputs[1]
attn_scores_2 = torch.tensor([torch.dot(x_i, query) for x_i in inputs])
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
context_vec_2 = sum(w * x_i for w, x_i in zip(attn_weights_2, inputs))

print(f"Query (token 2): {query}")
print(f"Attention scores for token 2: {attn_scores_2}")
print(f"Attention weights for token 2: {attn_weights_2}")
print(f"Context vector for token 2: {context_vec_2}")
print()

# 计算所有token的上下文向量
attn_scores_all, attn_weights_all, context_vecs_all = simple_self_attention(inputs)
print(f"All context vectors:\n{context_vecs_all}")
print()

# ============================================
# 3. 带可训练权重的自注意力 (SelfAttention_v1)
# ============================================
print("=" * 60)
print("Step 3: Self-Attention with Trainable Weights (v1)")
print("=" * 60)


class SelfAttention_v1(nn.Module):
    """使用手动参数的自注意力"""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


d_in, d_out = 3, 2
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
output_v1 = sa_v1(inputs)

print(f"Input shape: {inputs.shape}")
print(f"Output shape: {output_v1.shape}")
print(f"Output (first 2 tokens):\n{output_v1[:2]}")
print()

# ============================================
# 4. 使用Linear层的自注意力 (SelfAttention_v2)
# ============================================
print("=" * 60)
print("Step 4: Self-Attention with Linear Layers (v2)")
print("=" * 60)


class SelfAttention_v2(nn.Module):
    """使用nn.Linear的自注意力（推荐）"""

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
output_v2 = sa_v2(inputs)

print(f"Output shape: {output_v2.shape}")
print(f"Output (first 2 tokens):\n{output_v2[:2]}")
print()

# ============================================
# 5. 因果注意力（Causal Attention）
# ============================================
print("=" * 60)
print("Step 5: Causal Attention (Masking Future Tokens)")
print("=" * 60)


class CausalAttention(nn.Module):
    """因果自注意力（屏蔽未来token）"""

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # 注册上三角掩码（屏蔽未来token）
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        # 应用因果掩码
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


# 创建批次数据（2个样本）
batch = torch.stack((inputs, inputs), dim=0)
print(f"Batch shape: {batch.shape}")

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, dropout=0.0)
causal_output = ca(batch)

print(f"Causal attention output shape: {causal_output.shape}")
print(f"Output (first batch, first 2 tokens):\n{causal_output[0, :2]}")
print()

# ============================================
# 6. 多头注意力包装器（MultiHeadAttentionWrapper）
# ============================================
print("=" * 60)
print("Step 6: Multi-Head Attention (Wrapper Version)")
print("=" * 60)


class MultiHeadAttentionWrapper(nn.Module):
    """通过包装多个单头注意力实现多头注意力"""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        # 拼接所有头的输出
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)
mha_wrapper = MultiHeadAttentionWrapper(
    d_in=3, d_out=2, context_length=context_length,
    dropout=0.0, num_heads=2
)
wrapper_output = mha_wrapper(batch)

print(f"Output shape: {wrapper_output.shape}")
print(f"Output (first batch, first token):\n{wrapper_output[0, 0]}")
print()

# ============================================
# 7. 高效多头注意力（MultiHeadAttention）
# ============================================
print("=" * 60)
print("Step 7: Efficient Multi-Head Attention")
print("=" * 60)


class MultiHeadAttention(nn.Module):
    """高效的多头注意力实现（单次矩阵乘法）"""

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

        # 1. 线性投影
        keys = self.W_key(x)  # (b, num_tokens, d_out)
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        values = self.W_value(x)  # (b, num_tokens, d_out)

        # 2. 重塑为多头格式
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 3. 转置以便批量计算
        keys = keys.transpose(1, 2)  # (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 4. 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)  # (b, num_heads, num_tokens, num_tokens)

        # 5. 应用因果掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 6. 计算注意力权重
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # 7. 计算上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 8. 合并多头
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


torch.manual_seed(123)
mha_efficient = MultiHeadAttention(
    d_in=3, d_out=2, context_length=context_length,
    dropout=0.0, num_heads=2
)
efficient_output = mha_efficient(batch)

print(f"Output shape: {efficient_output.shape}")
print(f"Output (first batch, first token):\n{efficient_output[0, 0]}")
print()

# ============================================
# 8. 注意力机制可视化演示
# ============================================
print("=" * 60)
print("Step 8: Attention Visualization Demo")
print("=" * 60)


def visualize_attention_weights(attention_weights, tokens=None):
    """打印注意力权重矩阵"""
    if tokens is None:
        tokens = ["Your", "journey", "starts", "with", "one", "step"]

    print("\nAttention Weight Matrix (rows = query, cols = key):")
    print("    " + "  ".join(f"{t[:4]:>4}" for t in tokens))
    for i, (row, token) in enumerate(zip(attention_weights, tokens)):
        row_str = "  ".join(f"{w:.3f}" for w in row[:6])
        print(f"{token[:4]:>4} {row_str}")

    # 检查每行是否和为1
    row_sums = attention_weights.sum(dim=-1)
    print(f"\nRow sums (should be 1): {row_sums}")


# 使用简单自注意力的权重进行可视化
_, attn_weights_viz, _ = simple_self_attention(inputs)
visualize_attention_weights(attn_weights_viz)

# ============================================
# 9. 因果注意力掩码演示
# ============================================
print("\n" + "=" * 60)
print("Step 9: Causal Attention Mask Demo")
print("=" * 60)


def visualize_causal_mask(context_length=6):
    """可视化因果注意力掩码"""
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    print(f"Causal Mask (1 = masked, 0 = allowed):")
    print("    " + "  ".join(f"t{i}" for i in range(1, context_length + 1)))
    for i in range(context_length):
        row_str = "  ".join(f"{int(mask[i, j])}" for j in range(context_length))
        print(f"t{i + 1}  {row_str}")

    # 显示掩码后的注意力分数示例
    attn_scores = torch.randn(context_length, context_length)
    masked_scores = attn_scores.masked_fill(mask.bool(), -float('inf'))
    print(f"\nOriginal attention scores[0]: {attn_scores[0]}")
    print(f"Masked attention scores[0]: {masked_scores[0]}")


visualize_causal_mask()

# ============================================
# 10. Dropout效果演示
# ============================================
print("\n" + "=" * 60)
print("Step 10: Dropout in Attention Demo")
print("=" * 60)


def demo_dropout_in_attention():
    """演示Dropout在注意力机制中的应用"""
    torch.manual_seed(123)
    dropout = nn.Dropout(0.5)

    # 创建示例注意力权重
    sample_weights = torch.ones(6, 6) / 6  # 均匀分布
    print(f"Original weights (first row): {sample_weights[0]}")

    # 应用dropout
    dropped_weights = dropout(sample_weights)
    print(f"After dropout (first row): {dropped_weights[0]}")
    print("Note: Surviving weights are scaled by 1/(1-0.5)=2")


demo_dropout_in_attention()

# ============================================
# 总结
# ============================================
print("\n" + "=" * 60)
print("Chapter 3 Summary")
print("=" * 60)

summary = """
✅ 已实现的注意力机制：

1. 简单自注意力（无可训练权重）
   - 点积注意力分数计算
   - Softmax归一化
   - 加权求和得到上下文向量

2. 带可训练权重的自注意力
   - SelfAttention_v1: 手动参数
   - SelfAttention_v2: nn.Linear实现（推荐）

3. 因果注意力
   - 上三角掩码屏蔽未来token
   - 支持批次处理
   - 可选的Dropout正则化

4. 多头注意力
   - MultiHeadAttentionWrapper: 包装多个单头
   - MultiHeadAttention: 高效实现（权重分割）

关键概念：
- Query, Key, Value: 注意力机制的核心组件
- 缩放点积注意力: /√d_k 防止梯度消失
- 因果掩码: 确保自回归生成
- 多头注意力: 捕获不同子空间的特征
"""

print(summary)
print("=" * 60)
print("✅ Chapter 3 Complete!")
print("=" * 60)