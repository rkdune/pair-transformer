import tiktoken
import argparse

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
    
    args = parser.parse_args()
    
    # Convert to dict, filtering out None values
    config_overrides = {k: v for k, v in vars(args).items() if v is not None}
    return config_overrides


