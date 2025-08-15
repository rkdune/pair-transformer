import torch
from utils import Tokenizer

class DataLoader:
    def __init__(self, B, T, config_tokenizer, device='cpu', pre_tokenized=False):
        self.batch_size = B # num of sequences processed together in each batch
        self.seq_len = T # how many tokens are in each sequence/batch
        self.device = device

        if not pre_tokenized:

            with open("data/tiny_shakespeare.txt", "r") as f:
                text = f.read()

            tokenizer = Tokenizer.get_tokenizer(config_tokenizer)
            encoding = tokenizer.encode(text)
            self.tokens = torch.tensor(encoding).to(device)

        else:
            # change this later to correctly deal with pre-tokenized data
            with open("data/tokenized_dataset.txt", "r") as f:
                self.tokens = f.read()

            print("using pre-tokenized data")

        # Calculate max possible starting position to avoid out-of-bounds
        # E.g. if you have 1000 tokens total and have seq_len=100, then max_start_pos is 899.
        self.max_start_pos = len(self.tokens) - self.seq_len - 1
        
        # Store info for clean printing
        self.total_tokens = len(self.tokens)
        self.max_sequences = self.max_start_pos // self.batch_size

    def next_batch(self):
        B, T = self.batch_size, self.seq_len

        # Random sampling: pick B random starting positions
        start_positions = torch.randint(0, self.max_start_pos, (B,), device=self.device)

        # Vectorized batch creation
        indices_x = start_positions.unsqueeze(1) + torch.arange(T, device=self.device)
        indices_y = start_positions.unsqueeze(1) + torch.arange(1, T + 1, device=self.device)
        
        x = self.tokens[indices_x]
        y = self.tokens[indices_y]

        return x, y
