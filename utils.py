import tiktoken
import argparse

tokenizer = tiktoken.get_encoding("gpt2")

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
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--use_muon', type=bool, help='Use Muon optimizer')
    
    # Logging parameters
    parser.add_argument('--wandb_enabled', type=bool, help='Enable wandb logging')
    parser.add_argument('--run', type=str, help='Name for wandb run')
    
    args = parser.parse_args()
    
    # Convert to dict, filtering out None values
    config_overrides = {k: v for k, v in vars(args).items() if v is not None}
    return config_overrides