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

open your jupyter notebook and select the kernel as the venv you just created.

press `run all`.

for future math reference:

<img src="whiteboard.webp" width="400"/>

### remaining todos
- [x] fix softmax after all mlps, should only be on last
- [ ] add layer normalization
- [ ] add weight initialization
- [ ] add causal masking for training
- [ ] add tinyshakespere for some training data
- [ ] add attention sink
- [ ] ... many other engineering concerns