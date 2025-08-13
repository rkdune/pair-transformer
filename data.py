import torch
from utils import tokenizer

class DataLoader:
    def __init__(self, B, T, device='cpu'):
        self.batch_size = B # num of sequences processed together in each batch
        self.seq_len = T # how many tokens are in each sequence/batch
        self.device = device
    
        with open("data/tiny_shakespeare.txt", "r") as f:
            text = f.read()
        
        encoding = tokenizer.encode(text)
        self.tokens = torch.tensor(encoding)
        
        # Calculate max possible starting positions to avoid out-of-bounds
        self.max_start_pos = len(self.tokens) - self.seq_len - 1
        
        print(f"loaded {len(self.tokens)} tokens with batch size of {self.batch_size} sequences and {self.seq_len} tokens per sequence in the batch")
        print(f"max sequences per epoch: {self.max_start_pos // self.batch_size}")
        print("*"*50)
        
    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        
        # Random sampling: pick B random starting positions
        start_positions = torch.randint(0, self.max_start_pos, (B,))
        
        # Create batch by sampling random sequences
        x_list = []
        y_list = []
        
        for start_pos in start_positions:
            x_seq = self.tokens[start_pos:start_pos + T]
            y_seq = self.tokens[start_pos + 1:start_pos + T + 1]
            x_list.append(x_seq)
            y_list.append(y_seq)
        
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        
        # Move to device if specified
        if self.device != 'cpu':
            x = x.to(self.device)
            y = y.to(self.device)
        
        return x, y