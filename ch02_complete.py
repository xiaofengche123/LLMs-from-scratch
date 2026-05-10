"""
LLMs-from-scratch - Chapter 2: Working with Text
完整的数据预处理流程（修复版）
"""

import os
import re
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from importlib.metadata import version

print("=" * 50)
print("Environment Check")
print("=" * 50)
print(f"PyTorch version: {version('torch')}")
print(f"tiktoken version: {version('tiktoken')}")
print()

# ============================================
# 加载文本数据
# ============================================
print("=" * 50)
print("Step 1: Loading Text Data")
print("=" * 50)


def load_text():
    """加载文本文件"""
    if not os.path.exists("the-verdict.txt"):
        print("❌ Error: the-verdict.txt not found!")
        print("Please manually download the file from:")
        print(
            "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
        print(f"and save it to: {os.getcwd()}")
        return ""

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"✓ Total characters: {len(raw_text):,}")
    print(f"✓ Preview: {raw_text[:100]}...")
    print()
    return raw_text


raw_text = load_text()
if not raw_text:
    exit(1)

# ============================================
# 基础分词器（SimpleTokenizer）
# ============================================
print("=" * 50)
print("Step 2: Building Simple Tokenizer")
print("=" * 50)


class SimpleTokenizerV2:
    """简单的分词器，支持特殊标记"""

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        """将文本转换为 token IDs"""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """将 token IDs 转换回文本"""
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!\"()\'])', r'\1', text)
        return text


def build_vocab_from_text(text):
    """从文本构建词汇表"""
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: idx for idx, token in enumerate(all_tokens)}
    return vocab, preprocessed


vocab, _ = build_vocab_from_text(raw_text)
print(f"✓ Vocabulary size: {len(vocab)}")
print(f"✓ Sample tokens: {list(vocab.keys())[:10]}")
print()

# 测试简单分词器
tokenizer_v2 = SimpleTokenizerV2(vocab)
test_text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces."
print(f"Test encode: {tokenizer_v2.encode(test_text)[:10]}...")
print()

# ============================================
# 使用 GPT-2 的 BPE 分词器
# ============================================
print("=" * 50)
print("Step 3: Using GPT-2 BPE Tokenizer")
print("=" * 50)

bpe_tokenizer = tiktoken.get_encoding("gpt2")
print(f"✓ BPE vocabulary size: {bpe_tokenizer.n_vocab}")

# 测试 BPE 分词器
test_text_bpe = "Hello, do you like tea? <|endoftext|> In the sunlit terraces."
encoded = bpe_tokenizer.encode(test_text_bpe, allowed_special={"<|endoftext|>"})
decoded = bpe_tokenizer.decode(encoded)
print(f"✓ BPE encode result: {encoded[:10]}...")
print(f"✓ BPE decode result: {decoded[:80]}...")
print()

# ============================================
# 创建数据集和数据加载器
# ============================================
print("=" * 50)
print("Step 4: Creating Dataset and DataLoader")
print("=" * 50)


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
                         stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    return dataloader


# 测试数据加载器
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print(f"✓ Batch size: {inputs.shape[0]}")
print(f"✓ Sequence length: {inputs.shape[1]}")
print(f"✓ Input sample:\n{inputs[:2]}")
print(f"✓ Target sample:\n{targets[:2]}")
print()

# ============================================
# 创建 Token Embeddings
# ============================================
print("=" * 50)
print("Step 5: Creating Token Embeddings")
print("=" * 50)

vocab_size = 50257
output_dim = 256
max_length = 4

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)
print(f"✓ Token embeddings shape: {token_embeddings.shape}")
print(f"  (batch_size, sequence_length, embedding_dim)")

# ============================================
# 创建 Positional Embeddings
# ============================================
print("=" * 50)
print("Step 6: Creating Positional Embeddings")
print("=" * 50)

pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
position_indices = torch.arange(max_length)
pos_embeddings = pos_embedding_layer(position_indices)
print(f"✓ Position embeddings shape: {pos_embeddings.shape}")
print(f"  (sequence_length, embedding_dim)")

# ============================================
# 组合输入嵌入
# ============================================
print("=" * 50)
print("Step 7: Final Input Embeddings")
print("=" * 50)

input_embeddings = token_embeddings + pos_embeddings
print(f"✓ Final input embeddings shape: {input_embeddings.shape}")
print(f"  (batch_size, sequence_length, embedding_dim)")
print()

# ============================================
# 完整的数据处理流程演示（修复版）
# ============================================
print("=" * 50)
print("Complete Pipeline Summary")
print("=" * 50)


def complete_pipeline_demo():
    """演示完整的数据处理流程"""
    print("\n📊 Complete Data Processing Pipeline:")
    print("-" * 40)

    # 1. 原始文本
    sample_text = "The cat sat on the mat."
    print(f"1️⃣  Raw Text: {sample_text}")

    # 2. Tokenization
    tokens = bpe_tokenizer.encode(sample_text)
    print(f"2️⃣  Tokens (IDs): {tokens}")

    # 3. 转换为张量
    token_tensor = torch.tensor(tokens).unsqueeze(0)
    print(f"3️⃣  Token Tensor Shape: {token_tensor.shape}")

    # 4. Token Embeddings
    token_emb = token_embedding_layer(token_tensor)
    print(f"4️⃣  Token Embeddings Shape: {token_emb.shape}")

    # 5. Positional Embeddings（修复：使用实际的序列长度）
    seq_len = len(tokens)
    pos_embedding_layer_demo = torch.nn.Embedding(seq_len, output_dim)
    pos_indices = torch.arange(seq_len)
    pos_emb = pos_embedding_layer_demo(pos_indices)
    print(f"5️⃣  Position Embeddings Shape: {pos_emb.shape}")

    # 6. Final Embeddings
    final_emb = token_emb + pos_emb
    print(f"6️⃣  Final Embeddings Shape: {final_emb.shape}")
    print("\n✅ Data is ready for the LLM!")


complete_pipeline_demo()

# ============================================
# 实用函数总结
# ============================================
print("\n" + "=" * 50)
print("Utility Functions Summary")
print("=" * 50)


def text_to_embeddings(text, tokenizer, token_emb_layer, output_dim):
    """
    将文本直接转换为模型输入嵌入
    """
    # Tokenization
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    token_tensor = torch.tensor(token_ids).unsqueeze(0)

    # Token embeddings
    token_embeddings = token_emb_layer(token_tensor)

    # Positional embeddings（动态创建）
    seq_len = len(token_ids)
    pos_embedding_layer = torch.nn.Embedding(seq_len, output_dim)
    pos_indices = torch.arange(seq_len)
    pos_embeddings = pos_embedding_layer(pos_indices)

    # Combine
    input_embeddings = token_embeddings + pos_embeddings

    return input_embeddings, token_ids


# 测试实用函数
test_text = "Hello, world! This is a test."
print(f"\nTest text: {test_text}")
embeddings, token_ids = text_to_embeddings(
    test_text, bpe_tokenizer, token_embedding_layer, output_dim
)
print(f"Output embeddings shape: {embeddings.shape}")
print(f"Token IDs: {token_ids}")

print("\n" + "=" * 50)
print("✅ Chapter 2 Complete!")
print("=" * 50)