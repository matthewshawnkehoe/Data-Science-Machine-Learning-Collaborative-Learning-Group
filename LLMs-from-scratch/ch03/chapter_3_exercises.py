import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SelfAttention_v1(nn.Module):
    """
    Self-attention mechanism implementation (version 1).

    This version manually computes queries, keys, and values using learnable weight matrices and calculates
    the attention scores using the dot product of queries and keys.
    """

    def __init__(self, d_in, d_out):
        """
        Initializes the SelfAttention_v1 module with given input and output dimensions.

        Parameters
        ----------
        d_in : int
            The input dimension.
        d_out : int
            The output dimension (also the dimension for queries, keys, and values).
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))  # Query weights
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))    # Key weights
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))  # Value weights

    def forward(self, x):
        """
        Forward pass for the SelfAttention_v1 module.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, num_tokens, d_in).

        Returns
        -------
        Tensor
            The context vector after applying self-attention.
        """
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # Calculate attention scores
        attn_scores = queries @ keys.T
        # Apply softmax to the attention scores
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Compute the context vector as weighted sum of values
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    """
    Self-attention mechanism implementation (version 2).

    This version uses nn.Linear layers for computing queries, keys, and values,
    instead of manually defined weight matrices.
    """

    def __init__(self, d_in, d_out):
        """
        Initializes the SelfAttention_v2 module with linear layers for queries, keys, and values.

        Parameters
        ----------
        d_in : int
            The input dimension.
        d_out : int
            The output dimension (also the dimension for queries, keys, and values).
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=False)  # Linear layer for queries
        self.W_key = nn.Linear(d_in, d_out, bias=False)    # Linear layer for keys
        self.W_value = nn.Linear(d_in, d_out, bias=False)  # Linear layer for values

    def forward(self, x):
        """
        Forward pass for the SelfAttention_v2 module.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, num_tokens, d_in).

        Returns
        -------
        Tensor
            The context vector after applying self-attention.
        """
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Calculate attention scores
        attn_scores = queries @ keys.T
        # Apply softmax to the attention scores
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=1)

        # Compute the context vector as weighted sum of values
        context_vec = attn_weights @ values
        return context_vec


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention mechanism that ensures attention is only applied to earlier tokens.

    This is typically used in autoregressive models, where the model cannot access future tokens
    when predicting the next token in a sequence.
    """

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        """
        Initializes the CausalSelfAttention module with linear layers for queries, keys, and values.

        Parameters
        ----------
        d_in : int
            The input dimension.
        d_out : int
            The output dimension.
        context_length : int
            Length of the context (for masking purposes).
        dropout : float
            Dropout probability for attention weights.
        qkv_bias : bool, optional
            Whether to add bias to query, key, and value layers (default is False).
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # Upper triangular mask

    def forward(self, x):
        """
        Forward pass for the CausalSelfAttention module.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, num_tokens, d_in).

        Returns
        -------
        Tensor
            The context vector after applying causal self-attention.
        """
        b, n_tokens, d_in = x.shape  # Batch size, number of tokens, input dimension
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Calculate attention scores with a causal mask to prevent attending to future tokens
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:n_tokens, :n_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute the context vector as weighted sum of values
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    """
    A wrapper around multiple causal self-attention heads.

    This class implements multi-head attention by stacking several instances of the CausalSelfAttention module
    and combining their results.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Initializes the MultiHeadAttentionWrapper module with multiple causal self-attention heads.

        Parameters
        ----------
        d_in : int
            The input dimension.
        d_out : int
            The output dimension.
        context_length : int
            Length of the context (for masking purposes).
        dropout : float
            Dropout probability for attention weights.
        num_heads : int
            The number of attention heads.
        qkv_bias : bool, optional
            Whether to add bias to query, key, and value layers (default is False).
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalSelfAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(d_out * num_heads, d_out * num_heads)

    def forward(self, x):
        """
        Forward pass for the MultiHeadAttentionWrapper module.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, num_tokens, d_in).

        Returns
        -------
        Tensor
            The combined context vector after applying multi-head attention.
        """
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.out_proj(context_vec)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism that computes attention in parallel over multiple heads.

    This class allows the model to attend to different parts of the input sequence using different attention
    heads, enabling the model to capture a wider range of relationships between tokens.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Initializes the MultiHeadAttention module with multiple attention heads.

        Parameters
        ----------
        d_in : int
            The input dimension.
        d_out : int
            The output dimension.
        context_length : int
            Length of the context (for masking purposes).
        dropout : float
            Dropout probability for attention weights.
        num_heads : int
            The number of attention heads.
        qkv_bias : bool, optional
            Whether to add bias to query, key, and value layers (default is False).
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


if __name__ == "__main__":

    # Exercise 3.1: Comparing SelfAttention_v1 and SelfAttention_v2

    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )

    torch.manual_seed(123)
    d_in, d_out = 3, 2
    sa_v1 = SelfAttention_v1(d_in, d_out)
    sa_v2 = SelfAttention_v2(d_in, d_out)

    sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
    sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
    sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)

    print('\nExercise 3.1 Solution\n')
    print('Inputs from Self_Attention_v1 ...\n')
    print(sa_v1(inputs))
    print('\nInputs from Self_Attention_v2 ...\n')
    print(sa_v2(inputs))

    # Exercise 3.2:  Returning two-dimensional embedding vectors

    print('\nExercise 3.2 Solution\n')

    batch = torch.stack((inputs, inputs), dim=0)
    print(f'Batch shape: {batch.shape}')  # 2 inputs with 6 tokens each, and each token has embedding dimension 3
    print('2 inputs with 6 tokens each, and each token has embedding dimension 3\n')

    max_length = 6
    context_length = max_length
    d_out = 1
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)

    context_vecs = mha(batch)

    print(context_vecs)
    print("\ncontext_vecs.shape:", context_vecs.shape)

    # Exercise 3.3: Initializing GPT-2 size attention modules

    print('\nExercise 3.3 Solution\n')

    context_length = 1024
    d_in, d_out = 768, 768
    num_heads = 12

    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The number of parameters is {count_parameters(mha)}\n')

    # Get parameter totals
    param_counts = {}
    for name, param in mha.named_parameters():
        num_params = param.numel()
        param_counts[name] = num_params
        print(f"{name}: shape {param.shape}, total parameters = {num_params}")

    print("\nParameter Counts:")
    print(f"W_query.weight:   {param_counts.get('W_query.weight', 0)}")
    print(f"W_key.weight:     {param_counts.get('W_key.weight', 0)}")
    print(f"W_value.weight:   {param_counts.get('W_value.weight', 0)}")
    print(f"out_proj.weight:  {param_counts.get('out_proj.weight', 0)}")
    print(f"out_proj.bias:    {param_counts.get('out_proj.bias', 0)}")

    total_params = (
            768 * 768  # W_query.weight
            + 768 * 768  # W_key.weight
            + 768 * 768  # W_value.weight
            + 768 * 768  # out_proj.weight
            + 768  # out_proj.bias
    )
    print(f"Total parameters: {total_params}")

