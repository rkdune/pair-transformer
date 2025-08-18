import torch
import torch.nn.functional as F
import numpy as np
import wandb
import os
import argparse
import time
from dotenv import load_dotenv
load_dotenv()

from utils import (Tokenizer, parse_args, save_model, print_device_info, 
                    print_training_header, print_training_progress, print_model_saved, 
                    print_inference_header, print_tokenizer_data_info, print_optimizer_info, inference_test_cases,
                    setup_distributed, cleanup_distributed, is_rank_zero, reduce_tensor, barrier, get_cosine_lr)

# imports from other files
from config import Config
from data import DataLoader
from model import Transformer
from optimizer import create_optimizer
from torch.nn.parallel import DistributedDataParallel as DDP


def inference(inference_config, inference_model, test_cases=inference_test_cases):
    print_inference_header()
    
    tokenizer = Tokenizer.get_tokenizer(inference_config.tokenizer)

    # For first 3 test cases, generate 100 tokens
    for i, text in enumerate(test_cases):
        tokens = tokenizer.encode(text)
        x = torch.tensor(tokens)
        x = x.unsqueeze(0)

        # Move input tensor to the same device as the model
        if inference_model:
            device = next(inference_model.parameters()).device
        else:
            inference_model = Transformer(inference_config)
            device = inference_config.device

        x = x.to(device)
        
        if i < 3:  # First 3 test cases: generate 100 tokens
            print(f"\nGenerating 100 tokens for: \"{text}\"")
            generated_text = text
            current_input = x.clone()
            
            for _ in range(100):
                # Forward pass
                with torch.no_grad():
                    out = inference_model(current_input)
                    next_token = out[0, -1].argmax().item()
                
                # Decode and append
                next_word = tokenizer.decode([next_token])
                generated_text += next_word
                
                # Update input for next iteration (sliding window)
                next_token_tensor = torch.tensor([[next_token]], device=device)
                current_input = torch.cat([current_input, next_token_tensor], dim=1)
                
                # Keep only the last context_len tokens to respect context window
                if current_input.size(1) > inference_config.context_len:
                    current_input = current_input[:, -inference_config.context_len:]
            
            print(f"\033[1m{generated_text}\033[0m\n")
            
        else:  # Remaining test cases: single token prediction
            with torch.no_grad():
                out = inference_model(x)
                pred_tokens = out.argmax(dim=-1)

            # Get next token prediction
            next_token = pred_tokens[0, -1].item()
            predicted_word = tokenizer.decode([next_token])
            
            print(f"{text}\033[1m{predicted_word}\033[0m")

def training(model_config):
    
    # Setup distributed training if enabled
    setup_distributed(model_config)
    
    # Set device - for distributed training, use local rank
    if model_config.distributed:
        device = f"cuda:{model_config.local_rank}"
        torch.cuda.set_device(model_config.local_rank)
    else:
        device = model_config.device
    
    # Display config and device info (only from rank 0)
    if is_rank_zero(model_config):
        model_config.display_config(extended=True)
        print_device_info(device, model_config)

    torch.manual_seed(42)

    if device.startswith("cuda"):
        torch.cuda.manual_seed(42)

    # Only initialize wandb from rank 0
    if model_config.wandb_enabled and is_rank_zero(model_config):
        # Initialize wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=model_config.run,
            config={
                "vocab_size": model_config.vocab_size,
                "embedding_dim": model_config.embedding_dim,
                "mlp_dim": model_config.mlp_dim,
                "num_blocks": model_config.num_blocks,
                "num_heads": model_config.num_heads,
                "context_len": model_config.context_len,
                "attention_dim": model_config.attention_dim,
                "device": model_config.device,
                "epochs": model_config.epochs,
                "lr": model_config.lr,
                "betas": model_config.betas,
                "eps": model_config.eps,
                "weight_decay": model_config.weight_decay,
                "use_muon": model_config.use_muon,
                "muon_lr": model_config.muon_lr,
                "muon_momentum": model_config.muon_momentum,
                "tokenizer": model_config.tokenizer,
                "learnable_params": model_config.learnable_params,
                "total_params": model_config.learnable_params + model_config.non_learnable_params
            }
        )

    train_loader = DataLoader(
        model_config.batch_size, 
        model_config.context_len, 
        model_config.tokenizer, 
        device,
        data_source=model_config.data_source,
        data_dir=model_config.data_dir,
        use_validation=model_config.use_validation,
        distributed=model_config.distributed,
        rank=model_config.local_rank,
        world_size=model_config.world_size
    )
    
    # Get data statistics
    data_stats = train_loader.get_stats()
    
    # Print clean tokenizer and data info (only from rank 0)
    if is_rank_zero(model_config):
        print_tokenizer_data_info(
            model_config.tokenizer, 
            model_config.vocab_size,
            data_stats
        )

    model = Transformer(model_config).to(device)
    
    # Apply torch.compile if enabled (default: True)
    if model_config.torch_compile and is_rank_zero(model_config):
        print("🚀 Compiling model with torch.compile for faster training...")
    elif is_rank_zero(model_config):
        print("⚠️  torch.compile disabled - training will be slower")
    
    if model_config.torch_compile:
        model = torch.compile(model, mode="reduce-overhead")
    
    # Wrap with DistributedDataParallel if distributed training is enabled
    if model_config.distributed:
        model = DDP(model, device_ids=[model_config.local_rank], output_device=model_config.local_rank)

    # Create optimizer based on config
    optimizer = create_optimizer(model, model_config)
    
    # Print clean optimizer info (only from rank 0)
    if hasattr(model_config, '_optimizer_info') and is_rank_zero(model_config):
        info = model_config._optimizer_info
        print_optimizer_info(
            info['type'],
            info.get('muon_params'),
            info.get('adamw_params'),
            info.get('muon_lr'),
            info.get('adamw_lr')
        )
    
    # Track global training time
    global_start_time = time.time()
    
    # Track total steps for steps limit
    total_steps = 0
    max_steps = model_config.max_steps
    final_loss = None  # Track final loss for saving

    # Calculate total steps using data loader stats
    batches_per_epoch = data_stats['batches_per_epoch']
    effective_batches_per_epoch = batches_per_epoch // model_config.accumulation_steps
    total_planned_steps = effective_batches_per_epoch * model_config.epochs
    
    # Print training header with theoretical start loss (only from rank 0)
    if is_rank_zero(model_config):
        print_training_header(model_config.epochs, total_planned_steps)
        print(f"Theoretical Start Loss: {np.log(model_config.vocab_size):.3f}")

    for epoch in range(model_config.epochs):
        if is_rank_zero(model_config):
            print(f"\nEpoch {epoch + 1}/{model_config.epochs}")
        
        for effective_batch in range(effective_batches_per_epoch):
            # Check if we've reached the step limit
            if max_steps is not None and total_steps >= max_steps:
                if is_rank_zero(model_config):
                    print(f"Reached maximum steps limit of {max_steps}. Stopping training.")
                # Save model before early stopping (if enabled, only from rank 0)
                if model_config.save_model and is_rank_zero(model_config):
                    # Get the actual model from DDP wrapper if needed
                    model_to_save = model.module if model_config.distributed else model
                    save_model(model_to_save, model_config, total_steps, final_loss, "max_steps_reached")
                break
            optimizer.zero_grad()
            loss_accum = 0.0

            for accum_step in range(model_config.accumulation_steps):
                batch_start_time = time.time()

                x, y = train_loader.next_batch()

                # Forward pass
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = loss / model_config.accumulation_steps  # Scale loss
                loss_accum += loss.item()

                # Backward pass
                loss.backward()

            # Reduce accumulated loss across all processes for logging
            if model_config.distributed:
                loss_tensor = torch.tensor(loss_accum, device=device)
                loss_accum = reduce_tensor(loss_tensor, model_config).item()
            
            # Calculate gradient norm and step after accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            optimizer.step()
            
            # Increment step counter and update final loss
            total_steps += 1
            final_loss = loss_accum
            
            batch_end_time = time.time()
            time_per_step = batch_end_time - batch_start_time

            # Periodic saving if enabled (only from rank 0)
            if (model_config.save_model and model_config.save_every is not None and 
                total_steps % model_config.save_every == 0 and is_rank_zero(model_config)):
                # Get the actual model from DDP wrapper if needed
                model_to_save = model.module if model_config.distributed else model
                save_model(model_to_save, model_config, total_steps, final_loss, f"step_{total_steps}")

            # Calculate tokens per second (for effective batch)
            total_tokens = model_config.effective_batch_size * model_config.context_len
            tokens_per_sec = total_tokens / time_per_step

            # Update learning rate with cosine annealing schedule
            if model_config.use_cosine_lr and max_steps is not None:
                # Calculate new learning rates
                new_muon_lr = get_cosine_lr(total_steps, max_steps, model_config.muon_lr, 
                                          model_config.min_lr_ratio, model_config.lr_warmup_ratio)
                new_adamw_lr = get_cosine_lr(total_steps, max_steps, model_config.lr, 
                                           model_config.min_lr_ratio, model_config.lr_warmup_ratio)
                
                # Update optimizer learning rates
                if model_config.use_muon and hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 1:
                    optimizer.param_groups[0]['lr'] = new_muon_lr  # Hidden weights (Muon)
                    optimizer.param_groups[1]['lr'] = new_adamw_lr  # Other params (AdamW)
                else:
                    optimizer.param_groups[0]['lr'] = new_adamw_lr

            # Get current learning rate
            if model_config.use_muon and hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 1:
                # For Muon optimizer, show both learning rates
                muon_lr = optimizer.param_groups[0]['lr']  # Hidden weights
                adamw_lr = optimizer.param_groups[1]['lr']  # Other params
                lr_display = f"Muon:{muon_lr:.6f}/AdamW:{adamw_lr:.6f}"
            else:
                # For AdamW optimizer
                current_lr = optimizer.param_groups[0]['lr']
                lr_display = f"{current_lr:.6f}"

            # Only log from rank 0
            if model_config.wandb_enabled and is_rank_zero(model_config):
                # Calculate global elapsed time
                global_elapsed_time = time.time() - global_start_time
                
                # Log to wandb
                wandb_metrics = {
                    "loss": loss_accum,
                    "epoch": epoch + 1,
                    "effective_batch": effective_batch,
                    "step": total_steps,
                    "grad_norm": grad_norm.item(),
                    "time_per_step": time_per_step,
                    "tokens_per_sec": tokens_per_sec,
                    "global_time": global_elapsed_time
                }

                # Add learning rate(s) to wandb
                if model_config.use_muon and hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 1:
                    wandb_metrics["muon_lr"] = optimizer.param_groups[0]['lr']
                    wandb_metrics["adamw_lr"] = optimizer.param_groups[1]['lr']
                else:
                    wandb_metrics["learning_rate"] = optimizer.param_groups[0]['lr']

                wandb.log(wandb_metrics)

            # Clean progress reporting (only from rank 0)
            if (total_steps % 100 == 0 or total_steps == 1 or effective_batch + 1 == effective_batches_per_epoch) and is_rank_zero(model_config):
                print_training_progress(total_steps, total_planned_steps, loss_accum, 
                                      grad_norm.item(), tokens_per_sec, lr_display, time_per_step,
                                      is_final=(effective_batch + 1 == effective_batches_per_epoch))
        
        # Break out of epoch loop if we've reached steps limit
        if max_steps is not None and total_steps >= max_steps:
            break

    # Save final model after training completion (if enabled, only from rank 0)
    if model_config.save_model and final_loss is not None and is_rank_zero(model_config):
        # Get the actual model from DDP wrapper if needed
        model_to_save = model.module if model_config.distributed else model
        save_path = save_model(model_to_save, model_config, total_steps, final_loss, "final")
        print_model_saved(save_path)

    # Clean up distributed training
    if model_config.distributed:
        cleanup_distributed(model_config)

    # Only finish wandb from rank 0
    if model_config.wandb_enabled and is_rank_zero(model_config):
        wandb.finish()

    # Return the unwrapped model for inference
    return model.module if model_config.distributed else model


if __name__ == "__main__":
    # Don't set CUDA_VISIBLE_DEVICES here - let torchrun handle it
    torch.set_float32_matmul_precision("high")
    
    # Configure CUDAGraph to skip dynamic shapes for better performance
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

    # Parse command line arguments
    config_overrides = parse_args()

    # Create config with defaults, then apply command line overrides
    # Set default wandb_enabled if not specified in overrides
    if 'wandb_enabled' not in config_overrides:
        config_overrides['wandb_enabled'] = True
        # config_overrides['run'] = "we are so back."
        # config_overrides['max_steps'] = 1000
    config = Config(**config_overrides)

    train = True
    if train:
        model = training(config)

    # Only run inference from rank 0 to avoid multiple inference outputs
    if is_rank_zero(config):
        inference(config, model)
