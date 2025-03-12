import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gpt import TransformerBlock, GPTModel, MultiHeadAttention, LayerNorm, FeedForward


if __name__ == "__main__":

    # Exercise 4.1: Parameters in the feed forward versus attention module

    print('\nExercise 4.1 Solution\n')

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    block = TransformerBlock(GPT_CONFIG_124M)
    print(block)

    total_params = sum(p.numel() for p in block.ff.parameters())
    print(f"\nTotal number of parameters in feed forward module: {total_params:,}")

    total_params = sum(p.numel() for p in block.att.parameters())
    print(f"\nTotal number of parameters in attention module: {total_params:,}")

    emb_dim = 768

    # Feed forward module
    linear1_params = emb_dim * (4 * emb_dim) + (4 * emb_dim)  # 1st Linear layer
    linear2_params = (4 * emb_dim) * emb_dim + emb_dim  # 2nd Linear layer
    total_ffn_params = linear1_params + linear2_params

    print("\nFeed Forward Module:")
    print(f"1st Linear layer: 768 inputs × 4×768 outputs + 4×768 bias units = {linear1_params:,} parameters")
    print(f"2nd Linear layer: 4×768 inputs × 768 outputs + 768 bias units = {linear2_params:,} parameters")
    print(f"Total Feed Forward Parameters: 1st Linear layer + 2nd Linear layer = 2,362,368 + 2,360,064 = {total_ffn_params:,} parameters\n")

    # Attention module
    w_query_params = emb_dim * emb_dim  # W_query
    w_key_params = emb_dim * emb_dim  # W_key
    w_value_params = emb_dim * emb_dim  # W_value
    out_proj_params = (emb_dim * emb_dim) + emb_dim  # out_proj

    total_attention_params = (3 * w_query_params) + out_proj_params

    print("Attention Module:")
    print(f"W_query: 768 inputs × 768 outputs = {w_query_params:,} parameters")
    print(f"W_key: 768 inputs × 768 outputs = {w_key_params:,} parameters")
    print(f"W_value: 768 inputs × 768 outputs = {w_value_params:,} parameters")
    print(f"Out projection: 768 inputs × 768 outputs + 768 bias units = {out_proj_params:,} parameters")
    print(f"Total Attention Parameters: W_query + W_key + W_value + out_proj = 3×589,824 + 590,592 = {total_attention_params:,} parameters")

    # Exercise 4.2:  Initializing larger GPT models

    print('\nExercise 4.2 Solution')

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }


    def get_config(base_config, model_name="gpt2-small"):
        GPT_CONFIG = base_config.copy()

        if model_name == "gpt2-small":
            GPT_CONFIG["emb_dim"] = 768
            GPT_CONFIG["n_layers"] = 12
            GPT_CONFIG["n_heads"] = 12

        elif model_name == "gpt2-medium":
            GPT_CONFIG["emb_dim"] = 1024
            GPT_CONFIG["n_layers"] = 24
            GPT_CONFIG["n_heads"] = 16

        elif model_name == "gpt2-large":
            GPT_CONFIG["emb_dim"] = 1280
            GPT_CONFIG["n_layers"] = 36
            GPT_CONFIG["n_heads"] = 20

        elif model_name == "gpt2-xl":
            GPT_CONFIG["emb_dim"] = 1600
            GPT_CONFIG["n_layers"] = 48
            GPT_CONFIG["n_heads"] = 25

        else:
            raise ValueError(f"Incorrect model name {model_name}")

        return GPT_CONFIG


    def calculate_size(model):  # based on chapter code

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params:,}")

        total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
        print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

        # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
        total_size_bytes = total_params * 4

        # Convert to megabytes
        total_size_mb = total_size_bytes / (1024 * 1024)

        print(f"Total size of the model: {total_size_mb:.2f} MB")

    # for model_abbrev in ("small", "medium", "large", "xl"):
    for model_abbrev in ("small", "medium", "large"):
        model_name = f"gpt2-{model_abbrev}"
        CONFIG = get_config(GPT_CONFIG_124M, model_name=model_name)
        model = GPTModel(CONFIG)
        print(f"\n{model_name}:")
        calculate_size(model)

    # Exercise 4.3: Using separate dropout parameters

    print('\nExercise 4.3 Solution\n')

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate_emb": 0.1,  # NEW: dropout for embedding layers
        "drop_rate_attn": 0.1,  # NEW: dropout for multi-head attention
        "drop_rate_shortcut": 0.1,  # NEW: dropout for shortcut connections
        "qkv_bias": False
    }


    class TransformerBlock2(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.att = MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate_attn"],  # NEW: dropout for multi-head attention
                qkv_bias=cfg["qkv_bias"])
            self.ff = FeedForward(cfg)
            self.norm1 = LayerNorm(cfg["emb_dim"])
            self.norm2 = LayerNorm(cfg["emb_dim"])
            self.drop_shortcut = nn.Dropout(cfg["drop_rate_shortcut"])

        def forward(self, x):
            # Shortcut connection for attention block
            shortcut = x
            x = self.norm1(x)
            x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            # Shortcut connection for feed-forward block
            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            return x


    class GPTModel2(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate_emb"])  # NEW: dropout for embedding layers

            self.trf_blocks = nn.Sequential(
                *[TransformerBlock2(cfg) for _ in range(cfg["n_layers"])])

            self.final_norm = LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits

    torch.manual_seed(123)
    block = GPTModel2(GPT_CONFIG_124M).trf_blocks[0]
    print(block)

    total_params = sum(p.numel() for p in block.ff.parameters())
    print(f"\nTotal number of parameters in feed forward module: {total_params:,}")

    total_params = sum(p.numel() for p in block.att.parameters())
    print(f"\nTotal number of parameters in attention module: {total_params:,}")



