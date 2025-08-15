import tiktoken
import argparse
import torch
import json
import os
from datetime import datetime

class Tokenizer:
    _tokenizers = {}
    
    TOKENIZER_CONFIGS = {
        "gpt2": {"vocab_size": 50257},
        "o200k_base": {"vocab_size": 199997}
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
            print(f"!! loading tokenizer {tokenizer_name} (vocab size {cls.TOKENIZER_CONFIGS[tokenizer_name]['vocab_size']}) !!")
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


