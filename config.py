import torch
from utils import Tokenizer

class Config():
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
    max_steps = None  # If None, train for full epochs; if set, limit to this many steps

    # LOGGING & OBSERVABILITY
    wandb_enabled = True
    print_per_layer_params = False
    run = None  # wandb run name
    
    # MODEL SAVING
    save_model = False  # Enable model saving (must be explicitly set to True)
    save_model_dir = "models"  # Directory to save models
    save_every = None  # If set, save model every N steps

    # MODEL COMPILATION
    torch_compile = True  # Enable torch.compile for faster training (disable with --torch_compile=False)

    # Inference
    temperature = 1.0

    # Gradient Accumulation
    accumulation_steps = 1

    # Tokenizer
    tokenizer = "gpt2"

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

        # vocab size depends on tokenizer (centralized management)
        self.vocab_size = Tokenizer.get_vocab_size(self.tokenizer)

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

        # gradient accumulation
        self.effective_batch_size = self.batch_size * self.accumulation_steps

    def display_config(self, extended=True):
        # This method is now deprecated - use print_model_params() from utils instead
        from utils import print_model_params
        print_model_params(self)

