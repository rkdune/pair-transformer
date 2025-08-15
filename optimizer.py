import torch

try:
    from muon import SingleDeviceMuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    print("Muon not available. Install with: pip install git+https://github.com/KellerJordan/Muon")

def create_optimizer(model, model_config):
    if model_config.use_muon and MUON_AVAILABLE:
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

        # Store optimizer info for clean printing
        muon_params = sum(p.numel() for p in hidden_weights)
        adamw_params = sum(p.numel() for p in other_params)
        
        # This will be called from the main training function for clean output
        model_config._optimizer_info = {
            'type': 'Hybrid',
            'muon_params': muon_params,
            'adamw_params': adamw_params,
            'muon_lr': model_config.muon_lr,
            'adamw_lr': model_config.lr
        }
        
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
            model_config._optimizer_info = {
                'type': 'AdamW (Muon fallback)',
                'muon_params': None,
                'adamw_params': sum(p.numel() for p in model.parameters()),
                'muon_lr': None,
                'adamw_lr': model_config.lr
            }
        else:
            model_config._optimizer_info = {
                'type': 'AdamW',
                'muon_params': None,
                'adamw_params': sum(p.numel() for p in model.parameters()),
                'muon_lr': None,
                'adamw_lr': model_config.lr
            }
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.lr, 
                                     betas=model_config.betas, eps=model_config.eps, 
                                     weight_decay=model_config.weight_decay)


    return optimizer