### getting started

```
# first install uv (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`, look on `https://docs.astral.sh/uv/getting-started/installation/` for more options)
```
```
# setup your venv
uv init
uv venv
```
```
# install requirements
uv pip install -r requirements.txt
```

### logging
if you have an account on [Weights & Biases (wandb.ai)](https://wandb.ai), create a .env file with the following values
```env
WANDB_API_KEY= your_wandb_api_key
WANDB_PROJECT= your_project name (usually within your "team")
WANDB_ENTITY= your_entity_name (usually your "team" name)
```
and change `config = Config(wandb_enabled=False)` to `config = Config(wandb_enabled=True)`

### training
open your jupyter notebook and select the kernel as the venv you just created.

press `run all`.

### for future math reference:

<img src="whiteboard.webp" width="400"/>

### remaining todos
- [x] fix softmax after all mlps, should only be on last
- [ ] add layer normalization
- [ ] add weight initialization
- [x] add causal masking for training
- [x] add tinyshakespere for some training data
- [x] add param counting
- [x] add logging w/ wandb
- [ ] add attention sink
- [ ] ... many other engineering concerns