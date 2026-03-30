#!/usr/bin/env python3
"""
distill_student.py — Knowledge Distillation from Pythia-410M to narrower students.

Usage:
    python3 distill_student.py --width 128
    python3 distill_student.py --width 256 --seed 42
    python3 distill_student.py --width 512 --num_steps 30000 --temperature 2.0

Trains a student transformer (same depth, narrower hidden dim) to match
the teacher's output distribution via KL divergence.

Typical training time on H100:
    d_S=128:  ~1-1.5 hours
    d_S=256:  ~1.5-2 hours
    d_S=512:  ~2-3 hours
    d_S=768:  ~3-4 hours
    d_S=1024: ~4-5 hours
"""

import argparse
import os
import sys
import time
import json
import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

# ============ ARGUMENT PARSING ============
parser = argparse.ArgumentParser(description="Distill Pythia-410M to a narrower student")
parser.add_argument("--width", type=int, required=True, help="Student hidden dimension (e.g., 128, 256, 512, 768, 1024)")
parser.add_argument("--num_steps", type=int, default=30000, help="Training steps (default 30000)")
parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature (default 2.0)")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size in sequences")
parser.add_argument("--seq_len", type=int, default=512, help="Sequence length (shorter than SAE for memory)")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--save_dir", type=str, default="checkpoints", help="Save directory")
parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")
parser.add_argument("--save_every", type=int, default=5000, help="Checkpoint every N steps")
parser.add_argument("--eval_every", type=int, default=500, help="Compute eval loss every N steps")
args = parser.parse_args()

# ============ CONFIGURATION ============
TEACHER_NAME = "EleutherAI/pythia-410m"
TEACHER_HIDDEN = 1024
TEACHER_LAYERS = 24
TEACHER_HEADS = 16
VOCAB_SIZE = 50304
DEVICE = "cuda"
DTYPE = torch.bfloat16

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs("results", exist_ok=True)

# Set seed for reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Figure out student architecture
# Attention heads must divide hidden dim evenly
# We keep depth the same (24 layers)
def get_num_heads(hidden_dim):
    """Find a reasonable number of attention heads for a given hidden dim."""
    for h in [16, 8, 4, 2, 1]:
        if hidden_dim % h == 0 and hidden_dim // h >= 16:
            return h
    return 1

student_heads = get_num_heads(args.width)
student_intermediate = args.width * 4  # Standard 4× FFN expansion

print(f"\n{'='*65}")
print(f"  KNOWLEDGE DISTILLATION — Pythia-410M → Student (d_S={args.width})")
print(f"{'='*65}")
print(f"  Teacher:         {TEACHER_NAME}")
print(f"  Teacher dim:     {TEACHER_HIDDEN}")
print(f"  Student dim:     {args.width}")
print(f"  Student heads:   {student_heads}")
print(f"  Student FFN:     {student_intermediate}")
print(f"  Depth:           {TEACHER_LAYERS} layers (same)")
print(f"  Temperature:     {args.temperature}")
print(f"  Learning rate:   {args.lr}")
print(f"  Batch:           {args.batch_size} × {args.seq_len} = {args.batch_size * args.seq_len:,} tok/step")
print(f"  Steps:           {args.num_steps:,}")
print(f"  Seed:            {args.seed}")
print(f"{'='*65}\n")


# ============ LOAD TEACHER ============
print("[1/4] Loading teacher (Pythia-410M)...")
tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
tokenizer.pad_token = tokenizer.eos_token

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_NAME, torch_dtype=DTYPE, device_map="cuda"
)
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

teacher_params = sum(p.numel() for p in teacher.parameters())
print(f"  ✓ Teacher: {teacher_params/1e6:.0f}M params")


# ============ CREATE STUDENT ============
print(f"[2/4] Creating student (d_S={args.width})...")

student_config = GPTNeoXConfig(
    hidden_size=args.width,
    num_hidden_layers=TEACHER_LAYERS,
    num_attention_heads=student_heads,
    intermediate_size=student_intermediate,
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=2048,
    rotary_pct=0.25,  # Same as Pythia
    use_cache=False,
)

student = GPTNeoXForCausalLM(student_config).to(DEVICE).to(DTYPE)
student.train()

student_params = sum(p.numel() for p in student.parameters())
print(f"  ✓ Student: {student_params/1e6:.1f}M params (randomly initialized)")
print(f"  ✓ Compression ratio: {teacher_params/student_params:.1f}×")


# ============ DATA PIPELINE ============
print("[3/4] Setting up data stream...")

from datasets import load_dataset

data_sources = [
    "monology/pile-uncopyrighted",
    "EleutherAI/the_pile_deduplicated",
    "NeelNanda/pile-10k",
]

dataset = None
for source in data_sources:
    try:
        dataset = load_dataset(source, split="train", streaming=True, trust_remote_code=True)
        next(iter(dataset))
        dataset = load_dataset(source, split="train", streaming=True, trust_remote_code=True)
        print(f"  ✓ Data: {source}")
        break
    except:
        continue

if dataset is None:
    print("  ❌ Could not load data!")
    sys.exit(1)

def tokenize_stream(dataset, tokenizer, batch_size, seq_len):
    buffer = []
    needed = batch_size * seq_len
    for example in dataset:
        text = example.get("text", "")
        if not text or len(text.strip()) < 50:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(tokens)
        while len(buffer) >= needed:
            batch = torch.tensor(buffer[:needed], dtype=torch.long).reshape(batch_size, seq_len)
            buffer = buffer[needed:]
            yield batch


# ============ OPTIMIZER ============
optimizer = torch.optim.AdamW(
    student.parameters(), 
    lr=args.lr, 
    betas=(0.9, 0.95),
    weight_decay=0.01,
)

# Learning rate schedule: linear warmup then cosine decay
warmup_steps = min(1000, args.num_steps // 10)

def get_lr(step):
    if step < warmup_steps:
        return args.lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(args.num_steps - warmup_steps, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

T = args.temperature


# ============ TRAINING LOOP ============
print(f"\n[4/4] Starting distillation training...")
print(f"  {'Step':>7} | {'Loss':>9} | {'LR':>10} | {'Tok/s':>8} | {'Elapsed':>8} | {'ETA':>6}")
print(f"  {'-'*62}")

start_time = time.time()
loss_history = []
eval_losses = []
running_loss = 0.0
running_count = 0

data_iter = tokenize_stream(dataset, tokenizer, args.batch_size, args.seq_len)

for step in range(1, args.num_steps + 1):
    # ---- GET DATA ----
    try:
        input_ids = next(data_iter).to(DEVICE)
    except StopIteration:
        data_iter = tokenize_stream(dataset, tokenizer, args.batch_size, args.seq_len)
        input_ids = next(data_iter).to(DEVICE)

    # ---- UPDATE LEARNING RATE ----
    current_lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = current_lr

    # ---- TEACHER FORWARD (frozen) ----
    with torch.no_grad():
        teacher_outputs = teacher(input_ids)
        teacher_logits = teacher_outputs.logits  # [batch, seq, vocab]
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

    # ---- STUDENT FORWARD ----
    student_outputs = student(input_ids)
    student_logits = student_outputs.logits  # [batch, seq, vocab]
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)

    # ---- KL DIVERGENCE LOSS ----
    # KL(teacher || student) scaled by T²
    loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (T ** 2)

    # ---- CHECK NaN ----
    if torch.isnan(loss):
        print(f"\n  ❌ NaN at step {step}! Saving emergency checkpoint...")
        torch.save({
            "student_state": student.state_dict(),
            "step": step,
            "config": {"width": args.width, "seed": args.seed},
        }, os.path.join(args.save_dir, f"distill_w{args.width}_EMERGENCY.pt"))
        sys.exit(1)

    # ---- BACKWARD + UPDATE ----
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()

    # ---- TRACK LOSS ----
    loss_val = loss.item()
    running_loss += loss_val
    running_count += 1

    # ---- LOGGING ----
    if step % args.log_every == 0 or step == 1:
        avg_loss = running_loss / running_count
        elapsed = time.time() - start_time
        tok_sec = (step * args.batch_size * args.seq_len) / elapsed
        eta_min = (args.num_steps - step) / (step / elapsed) / 60

        loss_history.append({
            "step": step,
            "loss": round(avg_loss, 6),
            "lr": round(current_lr, 8),
            "tokens_M": round(step * args.batch_size * args.seq_len / 1e6, 1),
        })

        elapsed_str = f"{elapsed/60:.0f}m" if elapsed < 3600 else f"{elapsed/3600:.1f}h"

        print(
            f"  {step:>7,} | "
            f"{avg_loss:>9.5f} | "
            f"{current_lr:>10.2e} | "
            f"{tok_sec/1e6:>6.2f}M | "
            f"{elapsed_str:>8} | "
            f"{eta_min:>4.0f}m"
        )

        running_loss = 0.0
        running_count = 0

    # ---- EVALUATION (separate from training loss, no gradient) ----
    if step % args.eval_every == 0:
        student.eval()
        eval_loss_sum = 0.0
        eval_batches = 10

        with torch.no_grad():
            eval_iter = tokenize_stream(dataset, tokenizer, args.batch_size, args.seq_len)
            for _ in range(eval_batches):
                try:
                    eval_ids = next(eval_iter).to(DEVICE)
                except StopIteration:
                    break

                t_logits = teacher(eval_ids).logits
                t_probs = F.softmax(t_logits / T, dim=-1)
                s_logits = student(eval_ids).logits
                s_log_probs = F.log_softmax(s_logits / T, dim=-1)
                e_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T ** 2)
                eval_loss_sum += e_loss.item()

        eval_avg = eval_loss_sum / eval_batches
        eval_losses.append({"step": step, "eval_loss": round(eval_avg, 6)})
        student.train()

    # ---- CHECKPOINT ----
    if step % args.save_every == 0:
        ckpt_path = os.path.join(args.save_dir, f"distill_w{args.width}_s{args.seed}_step{step}.pt")
        torch.save({
            "student_state": student.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": {
                "width": args.width,
                "seed": args.seed,
                "temperature": T,
                "lr": args.lr,
                "num_steps": args.num_steps,
            },
            "loss_history": loss_history,
            "eval_losses": eval_losses,
        }, ckpt_path)
        print(f"  💾 Checkpoint: {ckpt_path}")

    # Periodic cleanup
    if step % 1000 == 0:
        gc.collect()
        torch.cuda.empty_cache()


# ============ FINAL SAVE ============
final_path = os.path.join(args.save_dir, f"distill_w{args.width}_s{args.seed}_final.pt")
torch.save({
    "student_state": student.state_dict(),
    "config": {
        "width": args.width,
        "seed": args.seed,
        "temperature": T,
        "lr": args.lr,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "student_params": student_params,
        "teacher_params": teacher_params,
    },
    "loss_history": loss_history,
    "eval_losses": eval_losses,
}, final_path)

# Save loss curves as JSON
results_path = f"results/distill_w{args.width}_s{args.seed}.json"
final_train_loss = loss_history[-1]["loss"] if loss_history else None
final_eval_loss = eval_losses[-1]["eval_loss"] if eval_losses else None

# Detect the floor: average of last 10% of eval losses
if len(eval_losses) >= 10:
    last_10pct = eval_losses[-(len(eval_losses)//10):]
    floor_estimate = sum(e["eval_loss"] for e in last_10pct) / len(last_10pct)
    floor_std = (sum((e["eval_loss"] - floor_estimate)**2 for e in last_10pct) / len(last_10pct)) ** 0.5
else:
    floor_estimate = final_eval_loss
    floor_std = 0.0

results = {
    "width": args.width,
    "seed": args.seed,
    "student_params_M": round(student_params / 1e6, 1),
    "teacher_params_M": round(teacher_params / 1e6, 1),
    "compression_ratio": round(teacher_params / student_params, 1),
    "temperature": T,
    "num_steps": args.num_steps,
    "final_train_loss": final_train_loss,
    "final_eval_loss": final_eval_loss,
    "estimated_floor": round(floor_estimate, 6) if floor_estimate else None,
    "floor_std": round(floor_std, 6) if floor_std else None,
    "total_tokens_M": round(args.num_steps * args.batch_size * args.seq_len / 1e6, 1),
    "loss_history": loss_history,
    "eval_losses": eval_losses,
}

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

total_time = (time.time() - start_time) / 3600

print(f"\n{'='*65}")
print(f"  ✅ DISTILLATION COMPLETE — d_S={args.width}, seed={args.seed}")
print(f"{'='*65}")
print(f"  Student params:     {student_params/1e6:.1f}M")
print(f"  Total steps:        {args.num_steps:,}")
print(f"  Total tokens:       {args.num_steps * args.batch_size * args.seq_len / 1e6:.0f}M")
print(f"  Total time:         {total_time:.2f} hours")
print(f"  Final train loss:   {final_train_loss:.6f}")
print(f"  Final eval loss:    {final_eval_loss:.6f}")
print(f"  Estimated floor:    {floor_estimate:.6f} ± {floor_std:.6f}")
print(f"  Model saved to:     {final_path}")
print(f"  Results saved to:   {results_path}")
print(f"{'='*65}")
