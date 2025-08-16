import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W_E = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.W_pos = nn.Embedding(config.context_len, config.embedding_dim)

    def forward(self, tokens):
        # Don't convert to tensor if already a tensor
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)
        
        embeddings = self.W_E(tokens)

        # Create positions tensor on the same device as tokens
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        position_embeddings = self.W_pos(positions)

        return embeddings + position_embeddings

class DeEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W_D = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def forward(self, x):
        embeddings = self.W_D(x)

        return embeddings


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.W_Q = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.W_K = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.W_V = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        
        self.W_out = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)

        self.num_heads = config.num_heads
        self.attention_dim = config.attention_dim

        # pre-compute attention denominator
        self.scale = math.sqrt(self.attention_dim)

        # attention sink
        self.sink_scalar = nn.Parameter(torch.zeros(self.num_heads))
        
        # Register causal mask as buffer
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.context_len, config.context_len)))
    
    def forward(self, x):
        B, T, C = x.shape  # B=batch_size, T=sequence_length, C=embedding_dim
        
        Q = einops.rearrange(self.W_Q(x), 'batch seq (head dim) -> batch head seq dim', head=self.num_heads)
        K = einops.rearrange(self.W_K(x), 'batch seq (head dim) -> batch head seq dim', head=self.num_heads)
        V = einops.rearrange(self.W_V(x), 'batch seq (head dim) -> batch head seq dim', head=self.num_heads)

        # Create causal mask for flash attention
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        
        # Use Flash Attention via scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=causal_mask, 
            dropout_p=0.0,
            is_causal=True
        )
        
        # Apply attention sink after flash attention
        # For sink implementation, we need to manually handle it since flash attention doesn't support custom sinks
        # Calculate attention scores for sink computation
        scores = (Q @ K.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # Expand attention sink scalar
        sink_expanded = einops.repeat(self.sink_scalar, 'head -> batch head seq 1', batch=B, seq=T)
        scores_with_sink = torch.cat([sink_expanded, scores], dim=-1)
        
        # Apply softmax to get sink probabilities
        attention_probs = torch.softmax(scores_with_sink, dim=-1)
        sink_prob, token_probs = attention_probs[..., :1], attention_probs[..., 1:]
        
        # Blend flash attention output with sink mechanism
        # Use token probabilities to weight the flash attention output
        token_prob_sum = token_probs.sum(dim=-1, keepdim=True)  # [B, H, T, 1]
        weighted_attn = attn_output * token_prob_sum
        
        QKV = einops.rearrange(weighted_attn, 'batch head seq dim -> batch seq (head dim)', head=self.num_heads)
        QKV_Out = self.W_out(QKV)

        return QKV_Out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer1 = nn.Linear(config.embedding_dim, config.mlp_dim, bias=True)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(config.mlp_dim, config.embedding_dim, bias=True)
    
    def forward(self, x):
        x = self.layer2(self.gelu(self.layer1(x)))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Attention_Layers = Attention(config)
        self.MLP_Layers = MLP(config)

        # layer norm
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)

    def forward(self, x):
        attention_out = self.Attention_Layers(self.layer_norm1(x))
        x = x + attention_out
        mlp_out = self.MLP_Layers(self.layer_norm2(x))
        x = x + mlp_out
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed = Embedding(config)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for i in range(config.num_blocks)
        ])

        self.final_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.deembed = DeEmbedding(config)

        # weight initialization (before weight tying)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        # tie weights after initialization
        self.deembed.W_D.weight = self.embed.W_E.weight


    def forward(self, x):

        x = self.embed(x)
        # print(f"after embed: {x}")

        for block in self.blocks:
            x = block(x)
        
        x = self.final_layer_norm(x)
        x = self.deembed(x)

        return x

