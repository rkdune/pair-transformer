import torch

try:
    from muon import SingleDeviceMuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    print("Muon not available. Install with: pip install git+https://github.com/KellerJordan/Muon")

def create_optimizer(model, model_config):
    if model_config.use_muon and MUON_AVAILABLE:
        print("Using Muon optimizer for hidden layers")
        
        # Separate parameters for Muon optimization
        hidden_weights = []
        other_params = []
        
        for name, param in model.named_parameters():
            # Hidden weights are 2D+ parameters in transformer blocks
            if param.ndim >= 2 and any(block_name in name for block_name in ['blocks',
        'Attention_Layers', 'MLP_Layers']):
                hidden_weights.append(param)
                if model_config.print_per_layer_params:
                    print(f"  MUON: {name} - shape: {param.shape} - params: {param.numel():,}")
            else:
                # Embeddings, biases, and other parameters
                other_params.append(param)
                if model_config.print_per_layer_params:
                    print(f"  ADAMW: {name} - shape: {param.shape} - params: {param.numel():,}")

        print(f"Muon params: {sum(p.numel() for p in hidden_weights):,}")
        print(f"AdamW params: {sum(p.numel() for p in other_params):,}")
        print(f"Total: {model_config.non_learnable_params + sum(p.numel() for p in hidden_weights) + sum(p.numel() for p in other_params):,}")

        print("*"*50)
        
        param_groups = [
            {
                'params': hidden_weights, 
                'use_muon': True, 
                'lr': model_config.muon_lr, 
                'weight_decay': model_config.weight_decay,
                'momentum': model_config.muon_momentum
            },
            {
                'params': other_params, 
                'use_muon': False, 
                'lr': model_config.lr,
                'weight_decay': model_config.weight_decay,
                'betas': model_config.betas,
                'eps': model_config.eps
            }
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        if model_config.use_muon and not MUON_AVAILABLE:
            print("Muon requested but not available, falling back to AdamW")
        else:
            print("Using AdamW optimizer")
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.lr, 
                                     betas=model_config.betas, eps=model_config.eps, 
                                     weight_decay=model_config.weight_decay)


    return optimizer