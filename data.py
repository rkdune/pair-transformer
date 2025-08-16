import torch
import os
import glob
from utils import Tokenizer

class DataLoader:
    def __init__(self, B, T, config_tokenizer, device='cpu', data_source="fineweb", data_dir=None, use_validation=False):
        self.batch_size = B  # num of sequences processed together in each batch
        self.seq_len = T     # how many tokens are in each sequence
        self.device = device

        # Load data based on source
        if data_source == "tiny_shakespeare":
            with open("data/tiny_shakespeare.txt", "r") as f:
                text = f.read()
            tokenizer = Tokenizer.get_tokenizer(config_tokenizer)
            encoding = tokenizer.encode(text)
            all_tokens = torch.tensor(encoding, dtype=torch.long).to(device)
            print("Using tiny_shakespeare dataset")
        
        elif data_source == "fineweb":
            if data_dir is None:
                raise ValueError("data_dir must be specified when using fineweb data source")
            
            # Discover available files
            if use_validation:
                file_pattern = os.path.join(data_dir, "fineweb-edu_val_*.pt")
                data_files = sorted(glob.glob(file_pattern))
                if not data_files:
                    raise FileNotFoundError(f"No validation files found in {data_dir}")
                print(f"Using fineweb validation set: {len(data_files)} files")
            else:
                file_pattern = os.path.join(data_dir, "fineweb-edu_train_*.pt")
                data_files = sorted(glob.glob(file_pattern))
                if not data_files:
                    raise FileNotFoundError(f"No training files found in {data_dir}")
                print(f"Using fineweb training set: {len(data_files)} files")
            
            self.data_files = data_files
            self.current_file_idx = 0
            
            # Load first file to get started
            print(f"Loading first file: {os.path.basename(data_files[0])}")
            all_tokens = torch.load(data_files[0], map_location=device)
            
            # Ensure correct dtype and device
            if all_tokens.dtype != torch.long:
                all_tokens = all_tokens.long()
            all_tokens = all_tokens.to(device)
            
            print(f"Loaded {len(all_tokens):,} tokens from first file")
            
        else:
            raise ValueError(f"Unknown data_source: {data_source}. Must be 'tiny_shakespeare' or 'fineweb'")

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
        self.data_source = data_source
        
        # Store current tokens for potential file rotation
        self.current_tokens = all_tokens

    def _load_next_file(self):
        """Load the next file in the sequence for fineweb data."""
        if self.data_source != "fineweb" or not hasattr(self, 'data_files'):
            return False
        
        # Move to next file
        self.current_file_idx = (self.current_file_idx + 1) % len(self.data_files)
        
        print(f"Loading next file: {os.path.basename(self.data_files[self.current_file_idx])}")
        new_tokens = torch.load(self.data_files[self.current_file_idx], map_location=self.device)
        
        # Ensure correct dtype and device
        if new_tokens.dtype != torch.long:
            new_tokens = new_tokens.long()
        new_tokens = new_tokens.to(self.device)
        
        # Update data structures with new file
        self.current_tokens = new_tokens
        self.total_tokens = len(new_tokens)
        self.total_sequences = self.total_tokens // self.seq_len
        self.batches_per_epoch = self.total_sequences // self.batch_size
        
        # Recreate sequences and batches
        tokens_to_use = self.batches_per_epoch * self.batch_size * self.seq_len
        self.tokens_utilized = tokens_to_use
        self.utilization_rate = tokens_to_use / self.total_tokens
        
        sequences_to_use = self.batches_per_epoch * self.batch_size
        useful_tokens = new_tokens[:sequences_to_use * self.seq_len]
        self.sequences = useful_tokens.view(sequences_to_use, self.seq_len)
        
        self.x_sequences = self.sequences[:, :-1]
        self.y_sequences = self.sequences[:, 1:]
        
        self.x_batches = self.x_sequences.view(self.batches_per_epoch, self.batch_size, self.seq_len - 1)
        self.y_batches = self.y_sequences.view(self.batches_per_epoch, self.batch_size, self.seq_len - 1)
        
        print(f"Loaded {len(new_tokens):,} tokens, {self.batches_per_epoch} batches available")
        return True

    def next_batch(self):
        """Return next batch of sequences in deterministic order."""
        if self.current_batch >= self.batches_per_epoch:
            # For fineweb, try to load next file; for tiny_shakespeare, reset
            if self.data_source == "fineweb":
                if self._load_next_file():
                    self.current_batch = 0  # Reset to start of new file
                else:
                    # If we can't load next file, reset to beginning
                    self.current_batch = 0
            else:
                # For tiny_shakespeare, just reset
                self.current_batch = 0
        
        x = self.x_batches[self.current_batch]
        y = self.y_batches[self.current_batch]
        
        self.current_batch += 1
        return x, y
    
    def get_stats(self):
        """Return statistics for logging."""
        stats = {
            'total_tokens': self.total_tokens,
            'total_sequences': self.total_sequences,
            'batches_per_epoch': self.batches_per_epoch,
            'tokens_utilized': self.tokens_utilized,
            'utilization_rate': self.utilization_rate,
            'effective_seq_len': self.seq_len - 1,  # After x/y shift
            'data_source': self.data_source
        }
        
        # Add fineweb-specific stats
        if self.data_source == "fineweb" and hasattr(self, 'data_files'):
            stats.update({
                'total_files': len(self.data_files),
                'current_file': self.current_file_idx + 1,
                'current_file_name': os.path.basename(self.data_files[self.current_file_idx])
            })
        
        return stats


# fineweb-edu-100B and fineweb-edu-10B (pre-tokenized on o200k_base) and fineweb-edu-10b-gpt2 (pre-tokenized on gpt2) are in /data/dedalus-research