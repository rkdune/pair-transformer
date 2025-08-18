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

```bash
# gpu ready testing
uv run train.py --run "name" --max_steps=5


# gpu ready training run
uv run train.py --run "name"  

# with distributed training
uv run torchrun --nproc_per_node=8 train.py --max_steps=2000


```

If you are on a computer cluster that uses slurm
1. create a `train.slurm` file with the following contents:
```bash
#!/bin/bash
#SBATCH --job-name=stu             # Name of the job
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Each node runs 1 task that manages all GPUs
#SBATCH --gpus-per-task=8          # Number of GPUs to allocate per task
#SBATCH --cpus-per-task=8          # Must match >= GPUs on the task
#SBATCH --mem=48G                  # Total memory for job
#SBATCH --time=15:59:00            # Max time limit

#SBATCH --error=logs/stu_%j.err
#SBATCH --output=logs/stu_%j.out

# Logging
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Error handling
set -e
trap 'log_info "Error on line $LINENO"; exit 1' ERR

# Activate your virtual environment accordingly
source .venv/bin/activate

# Get the first node (master node) from the SLURM_JOB_NODELIST
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

# Get the IP address of the master node
MASTER_NODE_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address)

# Find an available port
RDZV_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# Log start of training run
log_info "Starting training run..."

# Run the script using torchrun
torchrun \
--nnodes 1 \
--nproc_per_node 8 \
--rdzv_id $SLURM_JOB_ID \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_NODE_ADDR:$RDZV_PORT \
--max-restarts 16 \
train.py

# Log end of training run
log_info "Job finished."
```
2. Run
```bash
sbatch train.slurm
```
Your logs will appear in `/logs`. Run `scancel job_number` to cancel a dispatched job, where job_number is the most recent job number.

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
--tokenizer "o200k_base"          # Tokenizer to use (gpt2, o200k_base)

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

For [Weights & Biases](https://wandb.ai) logging, create a `.env` file with the following:
```env
WANDB_API_KEY=your_api_key
WANDB_PROJECT=your_project_name
WANDB_ENTITY=your_entity_name
```

### For future math reference:

<img src="assets/whiteboard.webp" width="400"/>

### tmux
```bash
tmux new -s your_session_name_here
# run your original command here, e.g. uv run train.py
# to detach tmux, press Ctrl+B, then D
tmux ls # to see if active tmux sessions exist
tmux attach -t your_session_name_here # to reattach
exit # to close tmux session completely
```

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
- [x] create ~20 different inference test cases other than napoleon
- [x] flash attention
- [ ] cosine lr scheduling
- [ ] calculate mfu 
    - compute as actual tokens/sec divided by theoretical peak tokens/sec, where theoretical peak tokens/sec is (GPU count × theoretical peak FLOPS per GPU) divided by 6N + 12LHQT
    - DGX B200 spec sheet, for entire 8x system the theoretical peak FLOPS is 72petaFLOPS at FP8, and 144 petaFLOPS at FP4 precision.
    - 6N + 12LHQT from chinchilla paper.
