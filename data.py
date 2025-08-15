import torch
from utils import Tokenizer

class DataLoader:
    def __init__(self, B, T, config_tokenizer, device='cpu', pre_tokenized=False):
        self.batch_size = B  # num of sequences processed together in each batch
        self.seq_len = T     # how many tokens are in each sequence
        self.device = device

        if not pre_tokenized:
            with open("data/tiny_shakespeare.txt", "r") as f:
                text = f.read()

            tokenizer = Tokenizer.get_tokenizer(config_tokenizer)
            encoding = tokenizer.encode(text)
            all_tokens = torch.tensor(encoding).to(device)
        else:
            # change this later to correctly deal with pre-tokenized data
            with open("data/tokenized_dataset.txt", "r") as f:
                all_tokens = f.read()
            print("using pre-tokenized data")

        # Sequential chunking (no overlaps)
        self.total_tokens = len(all_tokens)
        self.total_sequences = self.total_tokens // self.seq_len
        self.batches_per_epoch = self.total_sequences // self.batch_size
        
        # Only use tokens that fit into complete sequences
        tokens_to_use = self.batches_per_epoch * self.batch_size * self.seq_len
        self.tokens_utilized = tokens_to_use
        self.utilization_rate = tokens_to_use / self.total_tokens
        
        # Chunk data into non-overlapping sequences
        # Use only tokens that fit perfectly into batches (deterministic)
        sequences_to_use = self.batches_per_epoch * self.batch_size
        useful_tokens = all_tokens[:sequences_to_use * self.seq_len]
        self.sequences = useful_tokens.view(sequences_to_use, self.seq_len)
        
        # Create input (x) and target (y) sequences
        # x: tokens [0, 1, 2, ..., seq_len-1]  
        # y: tokens [1, 2, 3, ..., seq_len] (shifted by 1)
        self.x_sequences = self.sequences[:, :-1]  # Remove last token from each sequence
        self.y_sequences = self.sequences[:, 1:]   # Remove first token from each sequence
        
        # Reshape into batches: [num_batches, batch_size, seq_len-1]
        self.x_batches = self.x_sequences.view(self.batches_per_epoch, self.batch_size, self.seq_len - 1)
        self.y_batches = self.y_sequences.view(self.batches_per_epoch, self.batch_size, self.seq_len - 1)
        
        self.current_batch = 0

    def next_batch(self):
        """Return next batch of sequences in deterministic order."""
        if self.current_batch >= self.batches_per_epoch:
            # Reset for next epoch
            self.current_batch = 0
        
        x = self.x_batches[self.current_batch]
        y = self.y_batches[self.current_batch]
        
        self.current_batch += 1
        return x, y
    
    def get_stats(self):
        """Return statistics for logging."""
        return {
            'total_tokens': self.total_tokens,
            'total_sequences': self.total_sequences,
            'batches_per_epoch': self.batches_per_epoch,
            'tokens_utilized': self.tokens_utilized,
            'utilization_rate': self.utilization_rate,
            'effective_seq_len': self.seq_len - 1  # After x/y shift
        }
