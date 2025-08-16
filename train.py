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
                    print_inference_header, print_tokenizer_data_info, print_optimizer_info)

# imports from other files
from config import Config
from data import DataLoader
from model import Transformer
from optimizer import create_optimizer


def inference(inference_config, inference_model, text="They fear us"):
    print_inference_header()
    
    tokenizer = Tokenizer.get_tokenizer(inference_config.tokenizer)
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
    out = inference_model(x)
    pred_tokens = out.argmax(dim=-1)

    # Get next token prediction
    next_token = pred_tokens[0, -1].item()
    predicted_word = tokenizer.decode([next_token])
    
    print(f"Input: \"{text}\"")
    print(f"Predicted: \"{predicted_word}\" (token: {next_token})")
    print(f"Result: \"{text}{predicted_word}\"")
    
    # Smart analysis of prediction quality
    unique_tokens = len(set(pred_tokens.flatten().tolist()))
    total_tokens = len(pred_tokens.flatten())
    
    if unique_tokens == 1:
        print(f"\nNote: Model is stuck on token {pred_tokens[0, 0].item()} - likely undertrained")
    elif unique_tokens < total_tokens * 0.3:
        print(f"\nNote: Low token diversity ({unique_tokens}/{total_tokens} unique) - may need more training")
    else:
        print(f"\nModel shows good token diversity ({unique_tokens}/{total_tokens} unique tokens)")

def training(model_config):
    
    device = model_config.device
    
    # Print device info
    print_device_info(device)

    torch.manual_seed(42)

    if device == "cuda":
        torch.cuda.manual_seed(42)

    if model_config.wandb_enabled:
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

    train_loader = DataLoader(model_config.batch_size, model_config.context_len, model_config.tokenizer, device)
    
    # Get data statistics
    data_stats = train_loader.get_stats()
    
    # Print clean tokenizer and data info
    print_tokenizer_data_info(
        model_config.tokenizer, 
        model_config.vocab_size,
        data_stats
    )

    model = Transformer(model_config).to(device)
    
    # Apply torch.compile if enabled (default: True)
    if model_config.torch_compile:
        print("🚀 Compiling model with torch.compile for faster training...")
        model = torch.compile(model)
    else:
        print("⚠️  torch.compile disabled - training will be slower")

    # Create optimizer based on config
    optimizer = create_optimizer(model, model_config)
    
    # Print clean optimizer info
    if hasattr(model_config, '_optimizer_info'):
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
    
    # Print training header with theoretical start loss
    print_training_header(model_config.epochs, total_planned_steps)
    print(f"Theoretical Start Loss: {np.log(model_config.vocab_size):.3f}")

    for epoch in range(model_config.epochs):
        print(f"\nEpoch {epoch + 1}/{model_config.epochs}")
        
        for effective_batch in range(effective_batches_per_epoch):
            # Check if we've reached the step limit
            if max_steps is not None and total_steps >= max_steps:
                print(f"Reached maximum steps limit of {max_steps}. Stopping training.")
                # Save model before early stopping (if enabled)
                if model_config.save_model:
                    save_model(model, model_config, total_steps, final_loss, "max_steps_reached")
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

            # Calculate gradient norm and step after accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            optimizer.step()
            
            # Increment step counter and update final loss
            total_steps += 1
            final_loss = loss_accum
            
            batch_end_time = time.time()
            time_per_step = batch_end_time - batch_start_time

            # Periodic saving if enabled
            if (model_config.save_model and model_config.save_every is not None and 
                total_steps % model_config.save_every == 0):
                save_model(model, model_config, total_steps, final_loss, f"step_{total_steps}")

            # Calculate tokens per second (for effective batch)
            total_tokens = model_config.effective_batch_size * model_config.context_len
            tokens_per_sec = total_tokens / time_per_step

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

            if model_config.wandb_enabled:
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

            # Clean progress reporting
            if total_steps % 2 == 0 or effective_batch + 1 == effective_batches_per_epoch:
                print_training_progress(total_steps, total_planned_steps, loss_accum, 
                                      grad_norm.item(), tokens_per_sec, lr_display, time_per_step,
                                      is_final=(effective_batch + 1 == effective_batches_per_epoch))
        
        # Break out of epoch loop if we've reached steps limit
        if max_steps is not None and total_steps >= max_steps:
            break

    # Save final model after training completion (if enabled)
    if model_config.save_model and final_loss is not None:  # Only save if enabled and we completed at least one step
        save_path = save_model(model, model_config, total_steps, final_loss, "final")
        print_model_saved(save_path)

    if model_config.wandb_enabled:
        wandb.finish()

    return model


if __name__ == "__main__":

    torch.set_float32_matmul_precision("high")
    
    # Parse command line arguments
    config_overrides = parse_args()

    # Create config with defaults, then apply command line overrides
    config = Config(wandb_enabled=True, **config_overrides)
    config.display_config(extended=True)

    train = True
    if train:
        model = training(config)

    inference(config, model, "Napoleon was born in the city of")
