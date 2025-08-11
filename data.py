import torch
from utils import tokenizer

class DataLoader:
    def __init__(self, B, T):
        self.batch_size = B # num of sequences processed together in each batch
        self.seq_len = T # how many tokens are in each sequence/batch
    
        with open("data/tiny_shakespeare.txt", "r") as f:
            text = f.read()
        
        encoding = tokenizer.encode(text)
        self.tokens = torch.tensor(encoding)

        self.current_pos = 0 # maintain the index of the current data sample

        print(f"loaded {len(self.tokens)} tokens with batch size of {self.batch_size} sequences and {self.seq_len} tokens per sequence in the batch")
        print(f"each epoch has {len(self.tokens) / (self.batch_size * self.seq_len)} batches, with {self.seq_len * self.batch_size} tokens per batch, for a total of {self.seq_len * self.batch_size * (len(self.tokens) / (self.batch_size * self.seq_len))} tokens")
        print("*"*50)
        
    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        x = self.tokens[self.current_pos:self.current_pos+B*T]
        y = self.tokens[self.current_pos+1:self.current_pos+B*T+1]
        x = x.view(B, T)
        y = y.view(B, T)
        self.current_pos += B * T

        if (len(self.tokens) - self.current_pos + 1) < B * T:
            self.current_pos = 0

        return x, y