#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def extract_train_loss(logfile):
    iters, losses = [], []
    with open(logfile) as f:
        for line in f:
            m = re.search(r'iteration\s+(\d+)/.*lm loss:\s+([0-9.E+\-]+)', line)
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return iters, losses

def extract_val_loss(logfile):
    iters, losses = [], []
    with open(logfile) as f:
        for line in f:
            m = re.search(r'validation loss at iteration\s+(\d+)\s+\|.*lm loss value:\s+([0-9.E+\-]+)', line)
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return iters, losses

def extract_val_ppl(logfile):
    iters, ppls = [], []
    with open(logfile) as f:
        for line in f:
            m = re.search(r'validation loss at iteration\s+(\d+)\s+\|.*lm loss PPL:\s+([0-9.E+\-]+)', line)
            if m:
                iters.append(int(m.group(1)))
                ppls.append(float(m.group(2)))
    return iters, ppls

def get_log_stats(logfile):
    """Extract creation date and first/last iterations from log file."""
    creation_date = None
    first_iter = None
    last_iter = None
    avg_tflops = None
    avg_elapsed_time = None
    avg_tokens_per_s = None
    mem_usage = None
    
    with open(logfile) as f:
        for line in f:
            # Extract timestamp
            m = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if m and creation_date is None:
                creation_date = m.group(1)
            
            # Extract iteration numbers
            m = re.search(r'iteration\s+(\d+)/', line)
            if m:
                iter_num = int(m.group(1))
                if first_iter is None:
                    first_iter = iter_num
                last_iter = iter_num
            
            # Extract summary stats at end of log
            m = re.search(r'^throughput per GPU:\s+([0-9.]+)', line)
            if m:
                avg_tflops = float(m.group(1))
            m = re.search(r'^elapsed time per iteration:\s+([0-9.]+)', line)
            if m:
                avg_elapsed_time = float(m.group(1))
            m = re.search(r'^tokens/GPU/s:\s+([0-9.]+)', line)
            if m:
                avg_tokens_per_s = float(m.group(1))
            m = re.search(r'^mem usages:\s+([0-9.]+)', line)
            if m:
                mem_usage = float(m.group(1))
    
    return creation_date, first_iter, last_iter, avg_tflops, avg_elapsed_time, avg_tokens_per_s, mem_usage

def validate_and_print_stats(logs):
    """Validate log files and print statistics."""
    print("\n" + "="*80)
    print("LOG FILE STATISTICS")
    print("="*80 + "\n")
    
    for label, path in logs.items():
        # Check if file exists
        if not Path(path).exists():
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
        
        try:
            # Extract data
            train_iters, train_losses = extract_train_loss(path)
            val_iters, val_losses = extract_val_loss(path)
            val_iters_ppl, val_ppls = extract_val_ppl(path)
            creation_date, first_iter, last_iter, avg_tflops, avg_elapsed_time, avg_tokens_per_s, mem_usage = get_log_stats(path)
            
            # Validate we have data
            if not train_losses:
                print(f"ERROR: No training loss data found in {path}")
                sys.exit(1)
            if not val_losses:
                print(f"ERROR: No validation loss data found in {path}")
                sys.exit(1)
            if not val_ppls:
                print(f"ERROR: No validation perplexity data found in {path}")
                sys.exit(1)
            
            # Print statistics
            print(f"[{label}] {path}")
            print(f"  Created: {creation_date}")
            print(f"  Training iterations: {first_iter} - {last_iter} ({len(train_iters)} data points)")
            print(f"  Validation checkpoints: {len(val_iters)}")
            print(f"  Training loss:  min={min(train_losses):.6f}, max={max(train_losses):.6f}, final={train_losses[-1]:.6f}")
            print(f"  Val loss:       min={min(val_losses):.6f}, max={max(val_losses):.6f}, final={val_losses[-1]:.6f}")
            print(f"  Val PPL:        min={min(val_ppls):.2f}, max={max(val_ppls):.2f}, final={val_ppls[-1]:.2f}")
            if avg_tflops is not None:
                print(f"  Avg TFLOP/s/GPU:        {avg_tflops:.2f}")
            if avg_elapsed_time is not None:
                print(f"  Avg elapsed time/iter:  {avg_elapsed_time:.2f} ms")
            if avg_tokens_per_s is not None:
                print(f"  Avg tokens/GPU/s:       {avg_tokens_per_s:.2f}")
            if mem_usage is not None:
                print(f"  Mem usage:              {mem_usage:.4f}")
            print()
            
        except Exception as e:
            print(f"ERROR: Failed to parse {path}: {e}")
            sys.exit(1)

logs = {
    'BF16':  'output_bf16_baseline.log',
    'FP8':   'output_fp8_tensorwise.log',
    'NVFP4': 'output_nvfp4_selective.log',
}

colors = {'BF16': 'black', 'FP8': 'red', 'NVFP4': 'green'}
linestyles = {'BF16': '-', 'FP8': '-', 'NVFP4': '-'}

# Validate and print statistics
validate_and_print_stats(logs)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Training Loss
for label, path in logs.items():
    iters, losses = extract_train_loss(path)
    ax1.plot(iters, losses, label=label, color=colors[label],
             linestyle=linestyles[label], linewidth=1.0)

ax1.set_xlabel('Iteration', fontsize=13)
ax1.set_ylabel('Training LM Loss', fontsize=13)
ax1.set_title('WikiText-103 Training Loss: BF16 vs FP8 vs NVFP4', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# Validation Loss
for label, path in logs.items():
    iters, losses = extract_val_loss(path)
    ax2.plot(iters, losses, label=label, color=colors[label],
             linestyle=linestyles[label], linewidth=1.0, marker='o', markersize=3)

ax2.set_xlabel('Iteration', fontsize=13)
ax2.set_ylabel('Validation LM Loss', fontsize=13)
ax2.set_title('WikiText-103 Validation Loss: BF16 vs FP8 vs NVFP4', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

# Validation PPL
for label, path in logs.items():
    iters, ppls = extract_val_ppl(path)
    ax3.plot(iters, ppls, label=label, color=colors[label],
             linestyle=linestyles[label], linewidth=1.0, marker='o', markersize=3)

ax3.set_xlabel('Iteration', fontsize=13)
ax3.set_ylabel('Validation Perplexity', fontsize=13)
ax3.set_title('WikiText-103 Validation Perplexity: BF16 vs FP8 vs NVFP4', fontsize=14)
ax3.set_yscale('log')
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3, which='both')

fig.tight_layout()
fig.savefig('loss_comparison.png', dpi=150)
print("Plot saved to loss_comparison.png")
