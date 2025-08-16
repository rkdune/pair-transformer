![Pair Transformer](pair-transformer.png)


### Setup

```bash
# Install uv package manager if not installed already
brew install uv  # or see https://docs.astral.sh/uv/getting-started/installation/

# Create virtual environment and install dependencies
uv init
uv venv
uv pip install -r requirements.txt
```

### Training

```bash
uv run train.py
```

```bash
# examples with flags
uv run train.py --run "a cool ablation" --accumulation_steps 4 --save_model=True
```

**⚠️ Important:** If you're training on CPU or MPS (Apple Silicon), you may need to disable torch.compile:
```bash
uv run train.py --torch_compile=False
```

#### Training Flags

```bash
# Model architecture parameters
--num_blocks 6              # Number of transformer blocks
--num_heads 8               # Number of attention heads
--embedding_dim 512         # Model embedding dimension
--context_len 1024          # Maximum sequence length

# Training hyperparameters
--lr 3e-4                   # Learning rate for AdamW optimizer
--epochs 1                  # Number of training epochs
--max_steps 1000            # Maximum training steps (override)
--use_muon True             # Use Muon optimizer for hidden layers
--accumulation_steps 4      # Gradient accumulation steps

# Tokenizer parameters
--tokenizer "gpt2"          # Tokenizer to use (gpt2, o200k_base)

# Model compilation
--torch_compile True        # Enable torch.compile for faster training (default: True)

# Model saving parameters
--save_model True           # Enable model saving (required for saves)
--save_model_dir "models"   # Directory to save models
--save_every 100            # Save model every N steps (optional)

# Logging parameters
--wandb_enabled True        # Enable Weights & Biases logging
--run "my_experiment"       # Name for wandb run
```

If you want to train in a notebook, press `run all` in the `notebooks/training.ipynb` file.

### File Structure

```
├── train.py          # Main training script and inference
├── config.py         # Model and training configuration
├── model.py          # Transformer architecture (Attention, MLP, etc.)
├── data.py           # DataLoader for tiny_shakespeare
├── optimizer.py      # Muon/AdamW optimizer setup
├── utils.py          # Shared utilities (tokenizer)
└── data/             # Training data directory
    └── tiny_shakespeare.txt
└── notebooks/        # Directory to hold jupyter notebooks
    └── training.ipynb
```

### Logging (Optional)

For [Weights & Biases](https://wandb.ai) logging, create `.env`:
```env
WANDB_API_KEY=your_api_key
WANDB_PROJECT=your_project_name
WANDB_ENTITY=your_entity_name
```

### for future math reference:

<img src="assets/whiteboard.webp" width="400"/>

### todos
- [x] fix softmax after all mlps, should only be on last
- [x] add layer normalization
- [x] add weight initialization
- [x] add causal masking for training
- [x] add tinyshakespere for some training data
- [x] add param counting
- [x] add logging w/ wandb
- [x] add muon
- [x] modularize notebook into python files
- [x] gradient accumulation
- [x] add attention sink
- [x] improve logging: needs to also show gradient norm, learning rate, time per step, tok/s
- [x] vectorized batch loading
- [x] multi tokenizer support
- [x] max steps flag
- [x] model saving
- [x] fix batching
- [x] on-by-default torch compile
- [x] organized console logs
- [x] data loader chunking
- [x] high precision matmul
- [ ] calculate mfu 
    - compute as actual tokens/sec divided by theoretical peak tokens/sec, where theoretical peak tokens/sec is (GPU count × theoretical peak FLOPS per GPU) divided by 6N + 12LHQT
    - DGX B200 spec sheet, for entire 8x system the theoretical peak FLOPS is 72petaFLOPS at FP8, and 144 petaFLOPS at FP4 precision.
    - 6N + 12LHQT from chinchilla paper.
        - N is 
- [ ] create ~20 different inference test cases other than napoleon