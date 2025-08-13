import torch

class Config():
    # FIXED PARAMS
    vocab_size = 50257  # tiktoken vocab size for gpt2
    
    # CHANGEABLE MODEL ARCHITECTURE PARAMS
    embedding_dim = 720
    num_blocks = 2
    num_heads = 2
    context_len = 1024
    
    # TRAINING HYPERPARAMS
    batch_size = 8
    epochs = 1  # generally should keep this to 1
    lr = 3e-4
    muon_lr = 0.02 # from muon repo: "only the lr and weight decay have to be tuned"
    muon_momentum = 0.95
    betas = (0.9, 0.95)  # for controlling momentum
    eps = 1e-8
    weight_decay = 0.01
    use_muon = True  # whether to use Muon optimizer for hidden layers
    
    # LOGGING & OBSERVABILITY
    wandb_enabled = True
    print_per_layer_params = False
    run = None  # wandb run name

    # Inference
    temperature = 1.0
    
    def __init__(self, **kwargs):
        # Set all class attributes as instance attributes first
        for key in dir(self.__class__):
            if not key.startswith('_') and not callable(getattr(self.__class__, key)):
                setattr(self, key, getattr(self.__class__, key))
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self.__class__, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        
        # Derived params that depend on other params
        self.mlp_dim = 2 * self.embedding_dim
        self.attention_dim = self.embedding_dim // self.num_heads
        
        # Device detection
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # COUNTING PARAMETERS
        # learnable params
        self.learnable_params_dict = {
            "embedding": self.vocab_size * self.embedding_dim, 
            "positional_embedding": self.context_len * self.embedding_dim, 
            "MLPs (Weights)": self.num_blocks * 2 * self.embedding_dim * self.mlp_dim, 
            "MLPs (Biases)": self.num_blocks * (self.mlp_dim + self.embedding_dim), 
            "W_Qs": self.num_blocks * self.embedding_dim * self.embedding_dim, 
            "W_Ks": self.num_blocks * self.embedding_dim * self.embedding_dim, 
            "W_Vs": self.num_blocks * self.embedding_dim * self.embedding_dim, 
            "W_Out": self.num_blocks * self.embedding_dim * self.embedding_dim,
            "attention_sink_scalars": self.num_blocks * self.num_heads,
            "layer_norms": (self.num_blocks * 2 + 1) * 2 * self.embedding_dim}
        self.learnable_params = (lambda d: sum(d.values()))(self.learnable_params_dict)

        # non-learnable (fixed) params
        self.non_learnable_params_dict = {"deembedding (tied to embedding weights)": self.vocab_size * self.embedding_dim}
        self.non_learnable_params = (lambda d: sum(d.values()))(self.non_learnable_params_dict)
    
    def display_config(self, extended):
        if extended:
            print(f"learnable params dict: {self.learnable_params_dict}")
            print(f"total # of learnable params: {self.learnable_params:,}")
            print(f"non-learnable params dict: {self.non_learnable_params_dict}")
            print(f"total # of non-learnable params: {self.non_learnable_params:,}")
            print(f"** total # of params: {(self.learnable_params + self.non_learnable_params):,} **")
            print("*"*50)
        else:
            print(f"learnable params: {self.learnable_params:,}, non-learnable params: {self.non_learnable_params:,}, total params: {self.learnable_params + self.non_learnable_params:,}")