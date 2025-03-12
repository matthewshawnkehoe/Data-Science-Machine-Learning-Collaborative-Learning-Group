"""
This script implements core components of a GPT-like model as covered in Chapters 2-4.
It includes dataset preparation, multi-head attention, transformer blocks, and a
generation function. This script can be run as a standalone module.
"""

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2: Dataset Preparation
#####################################

class GPTDatasetV1(Dataset):
    """
    Dataset for preparing text sequences for training a GPT-like model.

    Parameters
    ----------
    txt : str
        Input text data.
    tokenizer : tiktoken.Encoding
        Tokenizer to encode text.
    max_length : int
        Maximum length of tokenized sequences.
    stride : int
        Step size for sliding window over text.
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk text into sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    Creates a PyTorch DataLoader for training.

    Parameters
    ----------
    txt : str
        Input text data.
    batch_size : int, optional
        Batch size for training, by default 4.
    max_length : int, optional
        Maximum sequence length, by default 256.
    stride : int, optional
        Step size for sliding window, by default 128.
    shuffle : bool, optional
        Whether to shuffle data, by default True.
    drop_last : bool, optional
        Whether to drop the last incomplete batch, by default True.
    num_workers : int, optional
        Number of worker threads, by default 0.

    Returns
    -------
    DataLoader
        PyTorch DataLoader instance.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

#####################################
# Chapter 3: Multi-Head Attention
#####################################

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention mechanism.

    Parameters
    ----------
    d_in : int
        Input embedding dimension.
    d_out : int
        Output embedding dimension.
    context_length : int
        Maximum sequence length.
    dropout : float
        Dropout rate.
    num_heads : int
        Number of attention heads.
    qkv_bias : bool, optional
        Whether to include bias terms in query, key, and value projections, by default False.
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Linear projections for queries, keys, and values
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        Forward pass of multi-head attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_tokens, d_in).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, num_tokens, d_out).
        """
        b, num_tokens, _ = x.shape
        keys, queries, values = self.W_key(x), self.W_query(x), self.W_value(x)

        keys, queries, values = [tensor.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2) for tensor in (keys, queries, values)]
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

#####################################
# Chapter 4: Transformer Components
#####################################

class LayerNorm(nn.Module):
    """
    Applies Layer Normalization to the input tensor.

    Parameters
    ----------
    emb_dim : int
        The embedding dimension of the input tensor.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # Small value to avoid division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))  # Scaling parameter
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # Shift parameter

    def forward(self, x):
        """
        Forward pass for Layer Normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_tokens, emb_dim).

        Returns
        -------
        torch.Tensor
            Layer-normalized tensor of the same shape as input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass for GELU activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated tensor.
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Feedforward neural network with two linear layers and GELU activation.
    """

    def __init__(self, cfg):
        """
        Parameters
        ----------
        cfg : dict
            Configuration dictionary containing model hyperparameters.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        Forward pass for the feedforward network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through feedforward layers.
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    Transformer block consisting of a multi-head attention layer
    followed by a feedforward network, with residual connections.
    """

    def __init__(self, cfg):
        """
        Parameters
        ----------
        cfg : dict
            Configuration dictionary containing model hyperparameters.
        """
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
        """
        Forward pass for the transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_tokens, emb_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as input.
        """
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_dim]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        return x


class GPTModel(nn.Module):
    """
    GPT-style model with token and positional embeddings,
    transformer blocks, and an output head.
    """

    def __init__(self, cfg):
        """
        Parameters
        ----------
        cfg : dict
            Configuration dictionary containing model hyperparameters.
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        Forward pass for the GPT model.

        Parameters
        ----------
        in_idx : torch.Tensor
            Input tensor of token indices (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_dim]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generates text using a GPT-like model.

    Parameters
    ----------
    model : GPTModel
        Trained GPT model.
    idx : torch.Tensor
        Initial input token indices.
    max_new_tokens : int
        Maximum number of new tokens to generate.
    context_size : int
        Context size limit for the model.

    Returns
    -------
    torch.Tensor
        Generated sequence including the input and new tokens.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()
