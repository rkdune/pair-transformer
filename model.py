import torch
import torch.nn as nn
import einops

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
        
        # Register causal mask as buffer
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.context_len, config.context_len)))
    
    def forward(self, x):
        B, T, C = x.shape
        
        Q = einops.rearrange(self.W_Q(x), 'batch seq (head dim) -> batch head seq dim', head=self.num_heads)
        K = einops.rearrange(self.W_K(x), 'batch seq (head dim) -> batch head seq dim', head=self.num_heads)
        V = einops.rearrange(self.W_V(x), 'batch seq (head dim) -> batch head seq dim', head=self.num_heads)

        # Calculate attention scores
        scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.attention_dim))
        
        # Apply causal mask
        scores = scores.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
        
        # Apply softmax
        QK = torch.softmax(scores, dim=-1)

        QKV = einops.rearrange(QK @ V, 'batch head seq dim -> batch seq (head dim)', head=self.num_heads)
        QKV_Out = self.W_out(QKV)

        return QKV_Out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer1 = nn.Linear(config.embedding_dim, config.mlp_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(config.mlp_dim, config.embedding_dim)
    
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
        x = self.layer_norm1(x)
        x = x + self.Attention_Layers(x)
        x = self.layer_norm2(x)
        return x + self.MLP_Layers(x)

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed = Embedding(config)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for i in range(config.num_blocks)
        ])

        self.deembed = DeEmbedding(config)
        self.deembed.W_D.weight = self.embed.W_E.weight  # tie weights

        # weight initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)


    def forward(self, x):

        x = self.embed(x)
        # print(f"after embed: {x}")

        for block in self.blocks:
            x = block(x)
        
        x = self.deembed(x)
        # print(f"after deembed: {x}")
        x = torch.softmax(x, dim=-1)

        return x

