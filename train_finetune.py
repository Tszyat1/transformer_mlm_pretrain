#!/usr/bin/env python3
"""
Fine-tuning script for MLM-pretrained model on SQuAD v2.0
Loads pretrained weights and fine-tunes for QA task
"""
import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from pathlib import Path

# Import model and dataset
from transformer_qa_mlm import TransformerQAWithMLM
from train import SQuADDataset  # Reuse the dataset from train.py

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=3, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}

def finetune_on_squad():
    """Fine-tune the pretrained model on SQuAD v2.0"""
    
    # Same configuration as train_final.py for fair comparison
    CONFIG = {
        # Data parameters
        'train_file': 'train-v2.0.json',
        'dev_file': 'dev-v2.0.json',
        'max_len': 384,
        'stride': 96,
        
        # Model parameters (must match pretraining)
        'd_model': 320,
        'n_heads': 10,
        'n_layers': 5,
        'd_ff': 1280,
        'dropout': 0.2,  # Same as non-pretrained version
        
        # Training parameters (same as train_final.py)
        'batch_size': 10,
        'accumulation_steps': 2,
        'epochs': 20,
        'lr': 1.2e-4,  # Same LR as non-pretrained
        'warmup_ratio': 0.08,
        'weight_decay': 0.01,
        'label_smoothing': 0.05,
        'gradient_clip': 1.0,
        
        # Data balancing (same as train_final.py)
        'neg_ratio': 0.4,
        'pos_weight': 1.1,
        'neg_weight': 0.9,
        
        # Early stopping
        'patience': 8,
        'min_delta': 0.001,
        
        # Pretrained model path
        'pretrained_model': 'pretrained/mlm_pretrained_model.pth'
    }
    
    print("="*70)
    print("FINE-TUNING PRETRAINED MODEL ON SQUAD v2.0")
    print("="*70)
    print(f"Device: {DEVICE}")
    print("\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print("="*70)
    
    # Check if pretrained model exists
    if not Path(CONFIG['pretrained_model']).exists():
        print(f"\n⚠️  Pretrained model not found at {CONFIG['pretrained_model']}")
        print("Please run pretraining first: python train_pretrain.py")
        return None
    
    # Initialize tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased', local_files_only=True)
    
    # Load datasets (same as train_final.py)
    print("\n[2/6] Loading datasets...")
    train_dataset = SQuADDataset(
        CONFIG['train_file'],
        tokenizer,
        max_len=CONFIG['max_len'],
        stride=CONFIG['stride'],
        neg_ratio=CONFIG['neg_ratio'],
        is_train=True
    )
    
    val_dataset = SQuADDataset(
        CONFIG['dev_file'],
        tokenizer,
        max_len=CONFIG['max_len'],
        stride=CONFIG['stride'],
        neg_ratio=1.0,
        is_train=False
    )
    
    print(f"  Training examples: {len(train_dataset):,}")
    print(f"  Validation examples: {len(val_dataset):,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'] * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Initialize model and load pretrained weights
    print("\n[3/6] Loading pretrained model...")
    model = TransformerQAWithMLM(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        d_ff=CONFIG['d_ff'],
        max_len=CONFIG['max_len'],
        dropout=CONFIG['dropout']
    ).to(DEVICE)
    
    # Load pretrained weights
    pretrained_ckpt = torch.load(CONFIG['pretrained_model'], map_location=DEVICE)
    
    # Load encoder weights (not the MLM head)
    if 'encoder_state_dict' in pretrained_ckpt:
        model.load_encoder_weights(pretrained_ckpt['encoder_state_dict'])
        print("  ✓ Loaded pretrained encoder weights")
    else:
        # Load full state dict but ignore MLM weights
        state_dict = pretrained_ckpt['model_state_dict']
        model_state = model.state_dict()
        for name, param in state_dict.items():
            if name in model_state and not name.startswith('mlm_'):
                model_state[name].copy_(param)
        print("  ✓ Loaded pretrained weights (excluding MLM head)")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Optimizer with parameter groups
    print("\n[4/6] Setting up optimizer and scheduler...")
    no_decay = ['bias', 'LayerNorm.weight', 'norm']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and not n.startswith('mlm_')],
            'weight_decay': CONFIG['weight_decay']
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and not n.startswith('mlm_')],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=CONFIG['lr'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine scheduler with warmup
    total_steps = len(train_loader) * CONFIG['epochs'] // CONFIG['accumulation_steps']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=CONFIG['patience'],
        min_delta=CONFIG['min_delta'],
        mode='min'
    )
    
    # Training metrics tracking
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    print("\n[5/6] Starting fine-tuning with early stopping...")
    print("="*70)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        # Training phase
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        
        for step, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            start_positions = batch['start_position'].to(DEVICE)
            end_positions = batch['end_position'].to(DEVICE)
            has_answer = batch['has_answer'].to(DEVICE)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                # Get QA predictions (not MLM)
                start_logits, end_logits = model(
                    input_ids, 
                    attention_mask, 
                    token_type_ids,
                    return_mlm=False  # Important: use QA heads
                )
                
                # Span losses
                start_loss = criterion(start_logits, start_positions)
                end_loss = criterion(end_logits, end_positions)
                
                # Weighted combination
                span_loss = (start_loss + end_loss) / 2
                
                # Apply answer type weights
                weights = torch.where(
                    has_answer == 1,
                    torch.full_like(span_loss, CONFIG['pos_weight']),
                    torch.full_like(span_loss, CONFIG['neg_weight'])
                )
                
                loss = (span_loss * weights).mean()
                loss = loss / CONFIG['accumulation_steps']
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % CONFIG['accumulation_steps'] == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * CONFIG['accumulation_steps']
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f"{loss.item() * CONFIG['accumulation_steps']:.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                token_type_ids = batch['token_type_ids'].to(DEVICE)
                start_positions = batch['start_position'].to(DEVICE)
                end_positions = batch['end_position'].to(DEVICE)
                
                start_logits, end_logits = model(
                    input_ids, 
                    attention_mask, 
                    token_type_ids,
                    return_mlm=False
                )
                
                start_loss = criterion(start_logits, start_positions)
                end_loss = criterion(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2
                
                val_loss += loss.mean().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Print epoch results
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best model (val_loss: {best_val_loss:.4f})")
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"\n⚠ Early stopping triggered at epoch {epoch}")
            print(f"  Best val loss: {best_val_loss:.4f}")
            break
        
        # Additional early stop if loss is good enough
        if avg_val_loss < 2.5:  # Lower threshold than non-pretrained
            print(f"\n✓ Target validation loss reached ({avg_val_loss:.4f} < 2.5)")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model checkpoint (val_loss: {best_val_loss:.4f})")
    
    # Save final model (compatible with existing evaluate.py)
    print("\n[6/6] Saving fine-tuned model...")
    os.makedirs('outputs', exist_ok=True)
    model_path = f'outputs/pretrained_finetuned_{torch.randint(0, 10**9, ()).item()}.pth'
    
    # Convert to format compatible with evaluate.py
    # Extract just the QA-relevant weights
    qa_state_dict = {}
    for name, param in model.named_parameters():
        if not name.startswith('mlm_'):
            # Remove MLM-specific layers for compatibility
            qa_state_dict[name] = param.data
    
    torch.save({
        'model_state_dict': qa_state_dict,
        'config': CONFIG,
        'vocab_size': tokenizer.vocab_size,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'pretrained': True  # Flag to indicate this was pretrained
    }, model_path)
    
    print(f"✓ Model saved to: {model_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Total epochs run: {len(train_losses)}")
    print("\n🎯 This model was PRETRAINED with MLM before fine-tuning!")
    print("   Expect significantly better HasAns F1 scores compared to non-pretrained.")
    
    return model_path

if __name__ == "__main__":
    model_path = finetune_on_squad()
    
    if model_path:
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("Run evaluation to see the improvement from pretraining:")
        print(f"\npython evaluate_pretrained.py \\")
        print(f"  --model {model_path} \\")
        print(f"  --data dev-v2.0.json \\")
        print(f"  --tok_dir ./bert-base-uncased \\")
        print(f"  --window_batch_size 24 \\")
        print(f"  --tune_threshold \\")
        print(f"  --thr_start -2 --thr_end 8 --thr_step 0.25")
        print("\n🚀 Expect HasAns F1 to be significantly higher than non-pretrained!")
        print("="*70)
