#!/usr/bin/env python3
"""
MLM Pretraining script for Transformer model
Pretrains on WikiText-103 with masked language modeling
"""
import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

from transformer_qa_mlm import TransformerQAWithMLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class WikiTextDataset(Dataset):
    """Dataset for MLM pretraining on WikiText."""
    def __init__(self, file_paths, tokenizer, max_len=384, mlm_probability=0.15, max_examples=100000):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_probability = mlm_probability
        self.examples = []
        
        # Handle both single file and list of files
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        print(f"Loading data from {len(file_paths)} file(s)...")
        
        all_texts = []
        for file_path in file_paths:
            if file_path.endswith('.parquet'):
                # Read Parquet file
                df = pd.read_parquet(file_path)
                # WikiText parquet files usually have a 'text' column
                texts = df['text'].tolist()
                all_texts.extend(texts)
            else:
                # Read raw text file (fallback)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                all_texts.append(text)
        
        # Process all texts
        print("Tokenizing and creating examples...")
        current_tokens = []
        
        for text in tqdm(all_texts, desc="Processing texts"):
            if not text or not text.strip():
                continue
            
            # Split text into sentences/paragraphs
            lines = text.split('\n') if '\n' in text else [text]
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('='):  # Skip empty lines and headers
                    continue
                
                tokens = tokenizer.tokenize(line)
                current_tokens.extend(tokens)
                
                # Create examples when we have enough tokens
                while len(current_tokens) >= max_len - 2:  # -2 for [CLS] and [SEP]
                    example_tokens = current_tokens[:max_len-2]
                    current_tokens = current_tokens[max_len-2:]
                    
                    # Convert to input_ids
                    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + example_tokens + ['[SEP]'])
                    input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))
                    
                    self.examples.append(input_ids)
                    
                    # Limit dataset size for faster training
                    if len(self.examples) >= max_examples:
                        break
                
                if len(self.examples) >= max_examples:
                    break
            
            if len(self.examples) >= max_examples:
                break
        
        print(f"Created {len(self.examples)} training examples")
        
        # Special tokens
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_ids = self.examples[idx].copy()
        
        # Create MLM labels
        labels = [-100] * len(input_ids)  # -100 = ignore in loss
        
        # Random masking
        for i in range(1, len(input_ids) - 1):  # Skip [CLS] and [SEP]
            if input_ids[i] == self.pad_token_id:
                continue
                
            if random.random() < self.mlm_probability:
                labels[i] = input_ids[i]  # Save true label
                
                # 80% mask, 10% random, 10% unchanged
                rand = random.random()
                if rand < 0.8:
                    input_ids[i] = self.mask_token_id
                elif rand < 0.9:
                    input_ids[i] = random.randint(0, self.vocab_size - 1)
                # else: keep unchanged
        
        # Create attention mask
        attention_mask = [1 if id != self.pad_token_id else 0 for id in input_ids]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'mlm_labels': torch.tensor(labels, dtype=torch.long)
        }

def pretrain_mlm():
    """Main MLM pretraining function."""
    set_seed(42)
    
    # Configuration
    CONFIG = {
        # Data - Updated for Parquet files
        'train_files': [
            'wikitext-103-raw-v1/train-00000-of-00002.parquet',
            'wikitext-103-raw-v1/train-00001-of-00002.parquet'
        ],
        'val_file': 'wikitext-103-raw-v1/validation-00000-of-00001.parquet',
        'max_len': 384,
        'mlm_probability': 0.15,
        'max_examples': 500000,  # Limit examples for faster training
        
        # Model (same architecture as fine-tuning)
        'd_model': 320,
        'n_heads': 10,
        'n_layers': 5,
        'd_ff': 1280,
        'dropout': 0.1,  # Lower dropout for pretraining
        
        # Training
        'batch_size': 16,
        'accumulation_steps': 2,
        'epochs': 10,  # Can increase for better pretraining
        'lr': 5e-4,  # Higher LR for pretraining
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
        
        # Early stopping
        'patience': 2,
        'min_delta': 0.01,
    }
    
    print("="*70)
    print("MLM PRETRAINING")
    print("="*70)
    print(f"Device: {DEVICE}")
    print("\nConfiguration:")
    for k, v in CONFIG.items():
        if k not in ['train_files']:  # Don't print the long file list
            print(f"  {k}: {v}")
    print(f"  train_files: {len(CONFIG['train_files'])} files")
    print("="*70)
    
    # Initialize tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased', local_files_only=True)
    
    # Check if WikiText exists (check for first training file)
    if not Path(CONFIG['train_files'][0]).exists():
        print(f"\n⚠️  WikiText-103 not found at {CONFIG['train_files'][0]}")
        print("\nPlease ensure your parquet files are in the correct location.")
        print("Expected structure:")
        print("  wikitext-103-raw-v1/")
        print("    ├── train-00000-of-00002.parquet")
        print("    ├── train-00001-of-00002.parquet")
        print("    └── validation-00000-of-00001.parquet")
        return None
    
    # Load datasets
    print("\n[2/5] Loading datasets...")
    print("Loading training data...")
    train_dataset = WikiTextDataset(
        CONFIG['train_files'],
        tokenizer,
        max_len=CONFIG['max_len'],
        mlm_probability=CONFIG['mlm_probability'],
        max_examples=CONFIG['max_examples']
    )
    
    print("Loading validation data...")
    val_dataset = WikiTextDataset(
        CONFIG['val_file'],
        tokenizer,
        max_len=CONFIG['max_len'],
        mlm_probability=CONFIG['mlm_probability'],
        max_examples=20000  # Smaller validation set
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'] * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    print("\n[3/5] Initializing model...")
    model = TransformerQAWithMLM(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        d_ff=CONFIG['d_ff'],
        max_len=CONFIG['max_len'],
        dropout=CONFIG['dropout']
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Optimizer
    print("\n[4/5] Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay'],
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
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Training metrics
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print("\n[5/5] Starting MLM pretraining...")
    print("="*70)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        # Training phase
        model.train()
        total_loss = 0
        total_acc = 0
        total_masked = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            mlm_labels = batch['mlm_labels'].to(DEVICE)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                loss, logits = model(
                    input_ids, 
                    attention_mask=attention_mask,
                    mlm_labels=mlm_labels,
                    return_mlm=True
                )
                loss = loss / CONFIG['accumulation_steps']
            
            # Calculate accuracy on masked tokens
            with torch.no_grad():
                mask = (mlm_labels != -100)
                if mask.sum() > 0:
                    predictions = logits[mask].argmax(dim=-1)
                    correct = (predictions == mlm_labels[mask]).sum().item()
                    total_acc += correct
                    total_masked += mask.sum().item()
            
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
            acc = (total_acc / max(1, total_masked)) * 100
            pbar.set_postfix({
                'loss': f"{loss.item() * CONFIG['accumulation_steps']:.4f}",
                'acc': f"{acc:.1f}%",
                'lr': f"{current_lr:.2e}"
            })
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = (total_acc / max(1, total_masked)) * 100
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_acc = 0
        val_masked = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                mlm_labels = batch['mlm_labels'].to(DEVICE)
                
                loss, logits = model(
                    input_ids,
                    attention_mask=attention_mask,
                    mlm_labels=mlm_labels,
                    return_mlm=True
                )
                
                val_loss += loss.item()
                
                # Calculate accuracy
                mask = (mlm_labels != -100)
                if mask.sum() > 0:
                    predictions = logits[mask].argmax(dim=-1)
                    correct = (predictions == mlm_labels[mask]).sum().item()
                    val_acc += correct
                    val_masked += mask.sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (val_acc / max(1, val_masked)) * 100
        
        # Print epoch results
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss - CONFIG['min_delta']:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best model (val_loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model checkpoint (val_loss: {best_val_loss:.4f})")
    
    # Save pretrained model
    print("\nSaving pretrained model...")
    os.makedirs('pretrained', exist_ok=True)
    
    # Save full model
    model_path = 'pretrained/mlm_pretrained_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.get_encoder_state_dict(),
        'config': CONFIG,
        'vocab_size': tokenizer.vocab_size,
        'best_val_loss': best_val_loss
    }, model_path)
    
    print(f"✓ Pretrained model saved to: {model_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("PRETRAINING COMPLETE")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {val_accuracy:.2f}%")
    print("\nNext step: Run fine-tuning on SQuAD")
    print("python train_finetune.py")
    print("="*70)
    
    return model_path

if __name__ == "__main__":
    pretrain_mlm()
