import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os
import glob
from utils import Tokenizer

class DataLoader:
    def __init__(self, B, T, config_tokenizer, device='cpu', data_source="fineweb 10B", data_dir=None, use_validation=False, distributed=False, rank=0, world_size=1):
        self.batch_size = B  # num of sequences processed together in each batch
        self.seq_len = T     # how many tokens are in each sequence
        self.device = device
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size

        # Load data based on source
        if data_source == "tiny_shakespeare":
            with open("data/tiny_shakespeare.txt", "r") as f:
                text = f.read()
            tokenizer = Tokenizer.get_tokenizer(config_tokenizer)
            encoding = tokenizer.encode(text)
            all_tokens = torch.tensor(encoding, dtype=torch.long).to(device)
            print(f"Using {data_source} dataset")
        
        elif data_source == "fineweb 10B" or data_source == "fineweb 100B":
            if data_dir is None:
                    raise ValueError("data_dir must be specified when not using tinyshakespeare")
            
            if data_source == "fineweb 10B":
                search_pattern = ["fineweb-edu_train_*.pt", "fineweb-edu_val_*.pt"]
                
            elif data_source == "fineweb 100B":
                search_pattern = ["fineweb-edu-100B_train_*.pt", "fineweb-edu-100B_val_*.pt"]
        
            # Discover available files
            if use_validation:
                file_pattern = os.path.join(data_dir, search_pattern[1])
                data_files = sorted(glob.glob(file_pattern))
                if not data_files:
                    raise FileNotFoundError(f"No validation files found in {data_dir}")
                print(f"Using {data_source} validation set: {len(data_files)} files")
            else:
                file_pattern = os.path.join(data_dir, search_pattern[0])
                data_files = sorted(glob.glob(file_pattern))
                if not data_files:
                    raise FileNotFoundError(f"No training files found in {data_dir}")
                print(f"Using {data_source} training set: {len(data_files)} files")

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
            raise ValueError(f"Unknown data_source: {data_source}. Must be 'tiny_shakespeare' or 'fineweb 10B' or 'fineweb 100B'")

        # Sequential chunking (no overlaps)
        self.total_tokens = len(all_tokens)
        self.total_sequences = self.total_tokens // self.seq_len
        
        # Calculate total batches available from current file
        total_file_batches = self.total_sequences // self.batch_size
        
        # For distributed training, each process gets a subset of batches
        if self.distributed:
            single_file_batches = total_file_batches // self.world_size
        else:
            single_file_batches = total_file_batches
        
        # For fineweb datasets, multiply by number of files to enable continuous training
        if data_source == "fineweb 10B" or data_source == "fineweb 100B":
            # This gives us a large batches_per_epoch that accounts for all files
            self.batches_per_epoch = len(self.data_files) * single_file_batches
        else:
            self.batches_per_epoch = single_file_batches
        
        # For distributed training, calculate sequences per rank directly to avoid precision loss
        if self.distributed:
            # Single source of truth: divide total sequences among ranks
            sequences_per_rank = self.total_sequences // self.world_size
            
            # Ensure sequences fit perfectly into batches
            single_file_batches = sequences_per_rank // self.batch_size
            sequences_to_use = single_file_batches * self.batch_size
            
            # Extract this rank's contiguous chunk based on actual sequences to use
            start_sequence_idx = self.rank * sequences_to_use
            end_sequence_idx = start_sequence_idx + sequences_to_use
            start_token_idx = start_sequence_idx * self.seq_len
            end_token_idx = end_sequence_idx * self.seq_len
            
            useful_tokens = all_tokens[start_token_idx:end_token_idx]
        else:
            # Single GPU: use all available data
            single_file_batches = total_file_batches
            sequences_to_use = single_file_batches * self.batch_size
            useful_tokens = all_tokens[:sequences_to_use * self.seq_len]
        
        # Calculate utilization stats based on actual data used
        self.tokens_utilized = len(useful_tokens)
        self.utilization_rate = self.tokens_utilized / self.total_tokens
        
        # Reshape into sequences
        self.sequences = useful_tokens.view(sequences_to_use, self.seq_len)
        
        # Create input (x) and target (y) sequences
        # x: tokens [0, 1, 2, ..., seq_len-1]  
        # y: tokens [1, 2, 3, ..., seq_len] (shifted by 1)
        self.x_sequences = self.sequences[:, :-1]  # Remove last token from each sequence
        self.y_sequences = self.sequences[:, 1:]   # Remove first token from each sequence
        
        # Reshape into batches: [num_batches, batch_size, seq_len-1]
        self.x_batches = self.x_sequences.view(single_file_batches, self.batch_size, self.seq_len - 1)
        self.y_batches = self.y_sequences.view(single_file_batches, self.batch_size, self.seq_len - 1)
        
        # Store single file batches for comparison
        self.single_file_batches = single_file_batches
        
        self.current_batch = 0
        self.data_source = data_source
        
        # Store current tokens for potential file rotation
        self.current_tokens = all_tokens

    def _load_next_file(self):
        """Load the next file in the sequence for fineweb data."""
        if not (self.data_source == "fineweb 10B" or self.data_source == "fineweb 100B") or not hasattr(self, 'data_files'):
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
        
        # Calculate batches for this file
        total_file_batches = self.total_sequences // self.batch_size
        
        if self.distributed:
            self.single_file_batches = total_file_batches // self.world_size
        else:
            self.single_file_batches = total_file_batches
        
        # For distributed training, calculate sequences per rank directly to avoid precision loss
        if self.distributed:
            # Single source of truth: divide total sequences among ranks
            sequences_per_rank = self.total_sequences // self.world_size
            
            # Ensure sequences fit perfectly into batches
            self.single_file_batches = sequences_per_rank // self.batch_size
            sequences_to_use = self.single_file_batches * self.batch_size
            
            # Extract this rank's contiguous chunk based on actual sequences to use
            start_sequence_idx = self.rank * sequences_to_use
            end_sequence_idx = start_sequence_idx + sequences_to_use
            start_token_idx = start_sequence_idx * self.seq_len
            end_token_idx = end_sequence_idx * self.seq_len
            
            useful_tokens = new_tokens[start_token_idx:end_token_idx]
        else:
            # Single GPU: use all available data
            self.single_file_batches = total_file_batches
            sequences_to_use = self.single_file_batches * self.batch_size
            useful_tokens = new_tokens[:sequences_to_use * self.seq_len]
        
        # Calculate utilization stats based on actual data used
        self.tokens_utilized = len(useful_tokens)
        self.utilization_rate = self.tokens_utilized / self.total_tokens
        
        # Reshape into sequences
        self.sequences = useful_tokens.view(sequences_to_use, self.seq_len)
        
        self.x_sequences = self.sequences[:, :-1]
        self.y_sequences = self.sequences[:, 1:]
        
        self.x_batches = self.x_sequences.view(self.single_file_batches, self.batch_size, self.seq_len - 1)
        self.y_batches = self.y_sequences.view(self.single_file_batches, self.batch_size, self.seq_len - 1)
        
        print(f"Loaded {len(new_tokens):,} tokens, {self.single_file_batches} batches available")
        return True

    def next_batch(self):
        """Return next batch of sequences in deterministic order."""
        # Check if we need to load next file (based on single file batches)
        if self.current_batch >= self.single_file_batches:
            # For fineweb, try to load next file; for tiny_shakespeare, reset
            if self.data_source == "fineweb 10B" or self.data_source == "fineweb 100B":
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
        if (self.data_source == "fineweb 10B" or self.data_source == "fineweb 100B") and hasattr(self, 'data_files'):
            stats.update({
                'total_files': len(self.data_files),
                'current_file': self.current_file_idx + 1,
                'current_file_name': os.path.basename(self.data_files[self.current_file_idx])
            })
        
        return stats


# fineweb-edu-100B and fineweb-edu-10B (pre-tokenized on o200k_base) and fineweb-edu-10b-gpt2 (pre-tokenized on gpt2) are in /data/dedalus-research