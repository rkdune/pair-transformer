import torch
import torch.nn.functional as F
import numpy as np
import wandb
import os
import argparse
import time
from dotenv import load_dotenv
load_dotenv()

from utils import Tokenizer, parse_args, save_model

# imports from other files
from config import Config
from data import DataLoader
from model import Transformer
from optimizer import create_optimizer


def inference(inference_config, inference_model, text="They fear us"):
    tokenizer = Tokenizer.get_tokenizer(inference_config.tokenizer)
    tokens = tokenizer.encode(text)
    x = torch.tensor(tokens)
    x = x.unsqueeze(0)

    print("*"*50)

    # Move input tensor to the same device as the model
    if inference_model:
        print("using passed in model for inference")
        device = next(inference_model.parameters()).device

    else:
        inference_model = Transformer(inference_config)
        print("using random model for inference")
        device = inference_config.device

    x = x.to(device)
    out = inference_model(x)
    pred_tokens = out.argmax(dim=-1)
    print(f"predicted tokens: {pred_tokens}")

    # Only take the prediction from the last position (next token after "fox")
    next_token = pred_tokens[0, -1].item()
    predicted_word = tokenizer.decode([next_token])
    print(f"predicted word: {predicted_word}")

    print(f"full sentence: {text}{predicted_word}")

    print("*"*50)
    print("sanity check: all predicted tokens")
    for num, token in enumerate(pred_tokens.flatten()):
        decoded = tokenizer.decode([token.item()])

        if num == (len(pred_tokens.flatten()) - 1):
            print(f"** Token {token} -> '{decoded}' **")
        else:
            print(f"Token {token} -> '{decoded}'")

def training(model_config):
    device = model_config.device
    print(f"Using device: {device}")

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

    model = Transformer(model_config).to(device)
    # model = torch.compile(model) # temporary comment to resolve errors with metal

    # Create optimizer based on config
    optimizer = create_optimizer(model, model_config)
    
    # Track global training time
    global_start_time = time.time()
    
    # Track total steps for steps limit
    total_steps = 0
    max_steps = model_config.max_steps
    final_loss = None  # Track final loss for saving

    for epoch in range(model_config.epochs):
        print(f"Epoch {epoch + 1}/{model_config.epochs}")

        num_batches = int(len(train_loader.tokens) / (train_loader.batch_size * train_loader.seq_len))
        effective_batches = num_batches // model_config.accumulation_steps

        for effective_batch in range(effective_batches):
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
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
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
            
            # Periodic saving if enabled
            if (model_config.save_model and model_config.save_every is not None and 
                total_steps % model_config.save_every == 0):
                save_model(model, model_config, total_steps, final_loss, f"step_{total_steps}")

            batch_end_time = time.time()
            time_per_step = batch_end_time - batch_start_time

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

            if effective_batch % 10 == 0 or effective_batch + 1 == effective_batches:
                print(f"Eff. Batch {effective_batch+1}/{effective_batches}, Loss: {loss_accum:.4f}, Grad Norm: {grad_norm.item():.4f}, LR: {lr_display}, Time/Step: {time_per_step:.3f}s, Tok/s: {tokens_per_sec:.0f}")
        
        # Break out of epoch loop if we've reached steps limit
        if max_steps is not None and total_steps >= max_steps:
            break

    # Save final model after training completion (if enabled)
    if model_config.save_model and final_loss is not None:  # Only save if enabled and we completed at least one step
        save_model(model, model_config, total_steps, final_loss, "final")

    if model_config.wandb_enabled:
        wandb.finish()

    return model


if __name__ == "__main__":
    # Parse command line arguments
    config_overrides = parse_args()

    # Create config with defaults, then apply command line overrides
    config = Config(wandb_enabled=True, **config_overrides)
    config.display_config(extended=True)
    print(f"theoretical start loss: {np.log(config.vocab_size)}")

    train = True
    if train:
        model = training(config)

    inference(config, model, "Napoleon was born in the city of")
