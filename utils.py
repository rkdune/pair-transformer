import tiktoken
import argparse
import torch
import torch.distributed as dist
import json
import os
from datetime import datetime

def print_section_header(title, emoji=""):
    """Print a clean section header with consistent formatting."""
    separator = "═" * 63
    header = f"{emoji} {title}" if emoji else title
    print(f"\n{separator}")
    print(f"{header}")
    print(f"{separator}")

def format_number(num):
    """Format numbers with commas for readability."""
    if isinstance(num, (int, float)):
        return f"{num:,}"
    return str(num)

def print_device_info(device, config=None):
    """Print clean device and environment information."""
    print_section_header("DEVICE & ENVIRONMENT", "🔧")
    print(f"Device: {device}")
    cuda_available = "Yes" if torch.cuda.is_available() else "No"
    mps_available = "Yes" if torch.backends.mps.is_available() else "No"
    print(f"CUDA Available: {cuda_available}")
    print(f"MPS Available: {mps_available}")
    
    # Add distributed training info
    if config and config.distributed:
        print(f"Distributed Training: Enabled")
        print(f"World Size: {config.world_size}")
        print(f"Local Rank: {config.local_rank}")
        print(f"Backend: {config.backend}")
    else:
        print(f"Distributed Training: Disabled")

def print_model_params(config):
    """Print clean model parameter information.""" 
    print_section_header("MODEL PARAMETERS", "📊")
    print(f"Architecture: Transformer")
    print(f"Transformer Blocks: {config.num_blocks}")
    print(f"Attention Heads: {config.num_heads}")
    print(f"Embedding Dim: {config.embedding_dim}")
    print(f"Context Length: {format_number(config.context_len)}")
    print(f"Vocab Size: {format_number(config.vocab_size)}")
    
    # Show detailed parameter breakdown
    print(f"\nParameter Breakdown:")
    for param_type, count in config.learnable_params_dict.items():
        print(f"  {param_type}: {format_number(count)}")
    
    print(f"\nLearnable Parameters: {format_number(config.learnable_params)}")
    print(f"Non-learnable Parameters: {format_number(config.non_learnable_params)}")
    print(f"Total Parameters: {format_number(config.learnable_params + config.non_learnable_params)}")

def print_tokenizer_data_info(tokenizer_name, vocab_size, data_stats):
    """Print clean tokenizer and data information with metrics."""
    print_section_header("TOKENIZER & DATA", "🔤")
    print(f"Tokenizer: {tokenizer_name} (vocab_size: {format_number(vocab_size)})")
    print(f"Dataset: {format_number(data_stats['total_tokens'])} tokens loaded")
    print(f"Sequence Length: {data_stats['effective_seq_len']} tokens (after x/y shift)")
    print(f"Available Sequences: {format_number(data_stats['total_sequences'])} (non-overlapping)")
    print(f"Batches per Epoch: {format_number(data_stats['batches_per_epoch'])}")
    print(f"Data Utilization: {data_stats['utilization_rate']:.1%} ({format_number(data_stats['tokens_utilized'])} / {format_number(data_stats['total_tokens'])} tokens)")

def print_optimizer_info(optimizer_type, muon_params=None, adamw_params=None, muon_lr=None, adamw_lr=None):
    """Print clean optimizer information."""
    print_section_header("OPTIMIZER", "⚡")
    if muon_params and adamw_params:
        print(f"Type: Hybrid (Muon + AdamW)")
        print(f"Muon Parameters: {format_number(muon_params)} (hidden layers)")
        print(f"AdamW Parameters: {format_number(adamw_params)} (other)")
        print(f"Learning Rates: Muon={muon_lr:.6f}, AdamW={adamw_lr:.6f}")
    else:
        print(f"Type: {optimizer_type}")
        if adamw_lr:
            print(f"Learning Rate: {adamw_lr:.6f}")

def print_training_header(epochs, total_steps=None):
    """Print training section header."""
    print_section_header("TRAINING", "🚀")
    if total_steps:
        print(f"Training: {epochs} epoch(s), {total_steps} steps total")
    else:
        print(f"Training: {epochs} epoch(s)")

def print_training_progress(step, total_steps, loss, grad_norm, tok_per_sec, lr_display, time_per_step, is_final=False):
    """Print clean training progress with detailed learning rate info."""
    prefix = "Final" if is_final else f"Step {step}"
    print(f"{prefix}: Loss={loss:.3f}, Grad={grad_norm:.3f}, LR={lr_display}, Time/Step={time_per_step:.3f}s, Tok/s={tok_per_sec:,.0f}")

def print_model_saved(save_path):
    """Print clean model saved message."""
    print(f"\n💾 Model saved: {save_path}")

def print_inference_header():
    """Print inference section header."""
    print_section_header("INFERENCE & VALIDATION", "🎯")

class Tokenizer:
    _tokenizers = {}
    
    TOKENIZER_CONFIGS = {
        "gpt2": {"vocab_size": 50257},
        "o200k_base": {"vocab_size": 200019}
    }
    
    @classmethod
    def get_vocab_size(cls, tokenizer_name="gpt2"):
        """Get vocab size without loading the tokenizer"""
        if tokenizer_name in cls.TOKENIZER_CONFIGS:
            return cls.TOKENIZER_CONFIGS[tokenizer_name]["vocab_size"]
        else:
            print(f"{tokenizer_name} is not supported, defaulting to gpt2 (vocab size {cls.TOKENIZER_CONFIGS['gpt2']['vocab_size']}).")
            print(f"supported tokenizers: {list(cls.TOKENIZER_CONFIGS.keys())}")
            return cls.TOKENIZER_CONFIGS["gpt2"]["vocab_size"]
    
    @classmethod
    def get_tokenizer(cls, tokenizer_name="gpt2"):
        """Get or create tokenizer instance (memory efficient - only creates once)"""
        if tokenizer_name not in cls.TOKENIZER_CONFIGS:
            print(f"{tokenizer_name} is not supported, defaulting to gpt2.")
            print(f"supported tokenizers: {list(cls.TOKENIZER_CONFIGS.keys())}")
            tokenizer_name = "gpt2"
        
        if tokenizer_name not in cls._tokenizers:
            # print(f"!! loading tokenizer {tokenizer_name} (vocab size {cls.TOKENIZER_CONFIGS[tokenizer_name]['vocab_size']}) !!")
            cls._tokenizers[tokenizer_name] = tiktoken.get_encoding(tokenizer_name)
        
        return cls._tokenizers[tokenizer_name]

def parse_args():
    """Parse command line arguments and return config overrides"""
    parser = argparse.ArgumentParser(description='Train transformer model')
    
    # Model architecture parameters
    parser.add_argument('--num_blocks', type=int, help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads')
    parser.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    parser.add_argument('--context_len', type=int, help='Context length')
    
    # Training parameters  
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--use_muon', type=bool, help='Use Muon optimizer')
    parser.add_argument('--accumulation_steps', type=int, help='Gradient accumulation steps')
    parser.add_argument('--max_steps', type=int, help='Maximum number of training steps (overrides epochs if set)')
    
    # Tokenizer parameters
    parser.add_argument('--tokenizer', type=str, help='Tokenizer to use (e.g., gpt2, o200k_base)')
    
    # Logging parameters
    parser.add_argument('--wandb_enabled', type=bool, help='Enable wandb logging')
    parser.add_argument('--run', type=str, help='Name for wandb run')
    
    # Model saving parameters
    parser.add_argument('--save_model', type=bool, help='Enable model saving (default: False)')
    parser.add_argument('--save_model_dir', type=str, help='Directory to save models (default: models)')
    parser.add_argument('--save_every', type=int, help='Save model every N steps (optional)')
    
    # Model compilation parameters
    parser.add_argument('--torch_compile', type=bool, help='Enable torch.compile for faster training (default: True)')
    
    args = parser.parse_args()
    
    # Convert to dict, filtering out None values
    config_overrides = {k: v for k, v in vars(args).items() if v is not None}
    return config_overrides


def save_model(model, config, total_steps, final_loss=None, run_suffix=""):
    """
    Save model state dict and metadata to organized directory structure.
    
    Args:
        model: PyTorch model to save
        config: Config object with training parameters
        total_steps: Number of training steps completed
        final_loss: Final training loss (optional)
        run_suffix: Additional suffix for run directory (e.g., "step_1000")
    
    Returns:
        str: Path where model was saved
    """
    # Create base save directory
    os.makedirs(config.save_model_dir, exist_ok=True)
    
    # Generate unique run directory name
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    if config.wandb_enabled and config.run:
        run_dir_name = f"{config.run}_{timestamp}"
    else:
        run_dir_name = f"run_{timestamp}"
    
    if run_suffix:
        run_dir_name = f"{run_dir_name}_{run_suffix}"
    
    save_path = os.path.join(config.save_model_dir, run_dir_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Save model state dict (on CPU for portability)
    model_path = os.path.join(save_path, "model.pt")
    torch.save(model.cpu().state_dict(), model_path)
    
    # Prepare metadata
    metadata = {
        "model_architecture": {
            "vocab_size": config.vocab_size,
            "embedding_dim": config.embedding_dim,
            "num_blocks": config.num_blocks,
            "num_heads": config.num_heads,
            "context_len": config.context_len,
            "mlp_dim": config.mlp_dim,
        },
        "training_config": {
            "tokenizer": config.tokenizer,
            "batch_size": config.batch_size,
            "learning_rate": config.lr,
            "epochs": config.epochs,
            "max_steps": config.max_steps,
            "accumulation_steps": config.accumulation_steps,
            "use_muon": config.use_muon,
        },
        "training_stats": {
            "total_steps": total_steps,
            "final_loss": final_loss,
            "effective_batch_size": config.effective_batch_size,
        },
        "metadata": {
            "saved_at": datetime.now().isoformat(),
            "device": str(config.device),
            "wandb_run": config.run if config.wandb_enabled else None,
        }
    }
    
    # Save metadata as JSON
    metadata_path = os.path.join(save_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to: {save_path}")
    print(f"  - Model state: {model_path}")
    print(f"  - Metadata: {metadata_path}")
    
    return save_path


def load_model(model_path, device=None):
    """
    Load model from saved checkpoint.
    
    Args:
        model_path: Path to directory containing model.pth and metadata.json
        device: Device to load model on (if None, uses CPU)
    
    Returns:
        tuple: (model_state_dict, metadata_dict)
    """
    # Load metadata
    metadata_path = os.path.join(model_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load model state dict
    model_file_path = os.path.join(model_path, "model.pt")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    
    if device is None:
        device = torch.device("cpu")
    
    state_dict = torch.load(model_file_path, map_location=device)
    
    print(f"Model loaded from: {model_path}")
    print(f"  - Training steps: {metadata['training_stats']['total_steps']}")
    print(f"  - Final loss: {metadata['training_stats']['final_loss']}")
    print(f"  - Tokenizer: {metadata['training_config']['tokenizer']}")
    
    return state_dict, metadata

def setup_distributed(config):
    """
    Initialize distributed training process group.
    
    Args:
        config: Config object with distributed training parameters
        
    Returns:
        None
    """
    if not config.distributed:
        return
    
    if config.local_rank is None or config.world_size is None:
        raise ValueError("local_rank and world_size must be set for distributed training")
    
    # Set CUDA device for this process
    torch.cuda.set_device(config.local_rank)
    
    # Initialize the process group
    dist.init_process_group(
        backend=config.backend,
        init_method='env://',
        world_size=config.world_size,
        rank=config.local_rank
    )
    
    # Only print from rank 0 to avoid spam
    if is_rank_zero(config):
        print(f"Distributed training initialized:")
        print(f"  Backend: {config.backend}")
        print(f"  World size: {config.world_size}")
        print(f"  Current rank: {config.local_rank}")

def cleanup_distributed(config):
    """Clean up distributed training process group."""
    if config.distributed:
        dist.destroy_process_group()

def is_rank_zero(config):
    """Check if current process is rank 0 (master process)."""
    return not config.distributed or config.local_rank == 0

def reduce_tensor(tensor, config, op=dist.ReduceOp.SUM):
    """
    Reduce tensor across all processes in distributed training.
    
    Args:
        tensor: Tensor to reduce
        config: Config object with distributed training parameters  
        op: Reduce operation (SUM, MEAN, etc.)
        
    Returns:
        Reduced tensor
    """
    if not config.distributed:
        return tensor
    
    # Clone tensor to avoid modifying original
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=op)
    
    if op == dist.ReduceOp.SUM:
        # Average across processes for metrics like loss
        reduced_tensor = reduced_tensor / config.world_size
    
    return reduced_tensor

def barrier(config):
    """Synchronize all processes."""
    if config.distributed:
        dist.barrier()

inference_test_cases = [
    "Napoleon was born in the city of",
    "The quick brown fox jumps over the",
    "To be or not to be, that is the",
    "In the beginning was the",
    "The meaning of life is",
    "Once upon a time in a",
    "The capital of France is",
    "Shakespeare wrote many plays including",
    "The sun rises in the",
    "Artificial intelligence will",
    "The greatest invention of mankind is",
    "Love is like a",
    "The ocean is deep and",
    "Music is the language of",
    "Time flies when you're",
    "The secret to happiness is",
    "Winter is coming and",
    "Knowledge is power, but",
    "The moon shines bright in the",
    "Dreams are the windows to"
]