#!/usr/bin/env python3
"""
measure_sae.py — Extract F, α, g(α), d*_S and predict loss floors from trained SAE.

Usage:
    python3 measure_sae.py --layer 8 --checkpoint checkpoints/sae_layer8_final.pt

Runs in ~10-15 minutes on H100.
"""

import argparse
import os
import json
import math
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="Measure SAE feature statistics")
parser.add_argument("--layer", type=int, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_tokens", type=int, default=50_000_000, help="Tokens for measurement (default 50M)")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seq_len", type=int, default=1024)
parser.add_argument("--alive_threshold", type=float, default=1e-5, help="Min frequency to count as alive")
args = parser.parse_args()

MODEL_NAME = "EleutherAI/pythia-410m"
HIDDEN_DIM = 1024
DEVICE = "cuda"
DTYPE = torch.bfloat16

print(f"\n{'='*65}")
print(f"  SAE MEASUREMENT — Layer {args.layer}")
print(f"{'='*65}\n")

# ============ LOAD MODELS ============
print("[1/4] Loading Pythia-410M...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map="cuda")
model.eval()

print("[2/4] Loading trained SAE...")
checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
sae_config = checkpoint["config"]
SAE_DIM = sae_config["d_sae"]

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)
        self.b_pre = nn.Parameter(torch.zeros(d_model))
    def encode(self, x):
        return F.relu(self.encoder(x - self.b_pre))
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z) + self.b_pre
        return x_hat, z

sae = SparseAutoencoder(HIDDEN_DIM, SAE_DIM).to(DEVICE).to(DTYPE)
sae.load_state_dict(checkpoint["model_state"])
sae.eval()
print(f"  ✓ SAE: {HIDDEN_DIM} → {SAE_DIM} ({sae_config['expansion']}×)")
print(f"  ✓ Trained on {sae_config.get('tokens_seen', '?')/1e6:.0f}M tokens")

# ============ HOOK ============
activation_cache = {}
def hook_fn(module, input, output):
    activation_cache["h"] = output[0] if isinstance(output, tuple) else output
hook = model.gpt_neox.layers[args.layer].register_forward_hook(hook_fn)

# ============ DATA ============
print(f"\n[3/4] Measuring feature statistics over {args.num_tokens/1e6:.0f}M tokens...")

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
    exit(1)

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

# ============ ACCUMULATE STATISTICS ============
# Per-feature accumulators (float32 for precision)
fire_count = torch.zeros(SAE_DIM, device=DEVICE, dtype=torch.float32)
act_sum = torch.zeros(SAE_DIM, device=DEVICE, dtype=torch.float32)
act_sq_sum = torch.zeros(SAE_DIM, device=DEVICE, dtype=torch.float32)
total_recon_loss = 0.0
total_positions = 0

tokens_per_batch = args.batch_size * args.seq_len
data_iter = tokenize_stream(dataset, tokenizer, args.batch_size, args.seq_len)
start = time.time()
batch_idx = 0

while total_positions < args.num_tokens:
    try:
        input_ids = next(data_iter).to(DEVICE)
    except StopIteration:
        break

    with torch.no_grad():
        model(input_ids)
        hidden = activation_cache["h"].reshape(-1, HIDDEN_DIM)

        # SAE encode
        z = sae.encode(hidden)  # [N, SAE_DIM]

        # Also measure reconstruction quality
        x_hat = sae.decode(z)
        recon = F.mse_loss(x_hat, hidden).item()
        total_recon_loss += recon

        # Accumulate per-feature stats
        active = (z > 0)
        fire_count += active.float().sum(dim=0)
        act_sum += (z * active.float()).sum(dim=0)
        act_sq_sum += (z * z).sum(dim=0)

    total_positions += tokens_per_batch
    batch_idx += 1

    if batch_idx % 100 == 0:
        elapsed = time.time() - start
        speed = total_positions / elapsed
        eta = (args.num_tokens - total_positions) / speed / 60
        print(f"    {total_positions/1e6:>6.0f}M / {args.num_tokens/1e6:.0f}M tokens | "
              f"recon: {recon:.5f} | ETA: {eta:.1f} min")

    if batch_idx % 500 == 0:
        gc.collect()
        torch.cuda.empty_cache()

hook.remove()
elapsed_total = time.time() - start

# ============ COMPUTE ALL STATISTICS ============
print(f"\n[4/4] Computing final statistics...")

# Per-feature frequency (fraction of tokens where feature fires)
freq = (fire_count / total_positions).cpu().numpy()

# Per-feature mean activation when active
safe_count = fire_count.clamp(min=1)
mean_act_when_active = (act_sum / safe_count).cpu().numpy()

# Per-feature E[z_i^2] over ALL tokens (our importance metric)
E_z_sq = (act_sq_sum / total_positions).cpu().numpy()

# Importance metrics
importance_v1 = freq * mean_act_when_active   # freq × mean strength
importance_v2 = E_z_sq                          # E[z²] — more theoretically grounded

# Alive features
alive_mask = freq > args.alive_threshold
F_alive = int(alive_mask.sum())
dead_features = SAE_DIM - F_alive

# Average active features per token
avg_active = float(fire_count.sum().item() / total_positions)

# Sparsity α
alpha = 1.0 - (avg_active / F_alive) if F_alive > 0 else 0.0

# Capacity function g(α)
if 0 < alpha < 1:
    g_alpha = 1.0 / ((1.0 - alpha) * math.log(1.0 / (1.0 - alpha)))
else:
    g_alpha = 1.0

# Critical width d*_S
d_star = F_alive / g_alpha if g_alpha > 0 else float('inf')

# Average reconstruction loss
avg_recon = total_recon_loss / batch_idx

# ============ LOSS FLOOR PREDICTIONS ============
sorted_imp = np.sort(importance_v2[alive_mask])[::-1]  # descending order
total_importance = float(sorted_imp.sum())

student_widths = [128, 256, 384, 512, 768, 1024]
predictions = {}

print(f"\n{'='*65}")
print(f"  SAE MEASUREMENTS — Layer {args.layer}")
print(f"{'='*65}")
print(f"  Measurement tokens:     {total_positions/1e6:.0f}M")
print(f"  Measurement time:       {elapsed_total/60:.1f} min")
print(f"  Avg reconstruction MSE: {avg_recon:.5f}")
print(f"  SAE width:              {SAE_DIM}")
print(f"  Dead features:          {dead_features} ({dead_features/SAE_DIM:.1%})")
print(f"  ─────────────────────────────────────────────")
print(f"  Alive features (F):     {F_alive}")
print(f"  Avg active per token:   {avg_active:.1f}")
print(f"  Sparsity (α):           {alpha:.6f}")
print(f"  Active fraction (1-α):  {1-alpha:.6f}")
print(f"  Capacity g(α):          {g_alpha:.2f}")
print(f"  Critical width d*_S:    {d_star:.0f}")
print(f"  Teacher width d_T:      1024")
print(f"  Ratio F / d_T:          {F_alive/1024:.1f}× (superposition ratio)")
print(f"  ─────────────────────────────────────────────")
print(f"  Total importance:       {total_importance:.6f}")
print(f"{'='*65}")

print(f"\n  PREDICTED LOSS FLOORS:")
print(f"  {'Width':<8} {'Capacity':<10} {'Kept':<8} {'Dropped':<9} {'Floor':>12} {'% Dropped':>10}")
print(f"  {'-'*60}")

for d_s in student_widths:
    capacity = d_s * g_alpha
    F_s = min(int(capacity), F_alive)
    dropped = F_alive - F_s
    floor_val = float(sorted_imp[F_s:].sum()) if F_s < F_alive else 0.0
    pct = (floor_val / total_importance * 100) if total_importance > 0 else 0.0

    predictions[str(d_s)] = {
        "d_S": d_s,
        "capacity": round(capacity, 1),
        "features_kept": F_s,
        "features_dropped": dropped,
        "predicted_floor": round(floor_val, 8),
        "pct_importance_dropped": round(pct, 2),
    }

    marker = " ← teacher width" if d_s == 1024 else ""
    marker = " ← d*_S" if abs(d_s - d_star) < 100 else marker
    print(f"  d_S={d_s:<5} {capacity:>8.0f}    {F_s:>6}  {dropped:>7}   {floor_val:>12.6f} {pct:>8.1f}%{marker}")

# ============ IMPORTANCE DISTRIBUTION ANALYSIS ============
print(f"\n  IMPORTANCE DISTRIBUTION:")
# Show top features
print(f"  Top 10 features account for: {sorted_imp[:10].sum()/total_importance*100:.1f}% of total importance")
print(f"  Top 100 features:            {sorted_imp[:100].sum()/total_importance*100:.1f}%")
print(f"  Top 1000 features:           {sorted_imp[:1000].sum()/total_importance*100:.1f}%")
print(f"  Bottom 50% features:         {sorted_imp[F_alive//2:].sum()/total_importance*100:.1f}%")

# Fit power law (log-log regression)
ranks = np.arange(1, len(sorted_imp) + 1)
log_ranks = np.log(ranks[sorted_imp > 0])
log_imps = np.log(sorted_imp[sorted_imp > 0])
if len(log_ranks) > 100:
    # Fit on middle portion to avoid edge effects
    mid = len(log_ranks) // 10
    end = 9 * len(log_ranks) // 10
    slope, intercept = np.polyfit(log_ranks[mid:end], log_imps[mid:end], 1)
    print(f"  Power law exponent (β):      {-slope:.3f} (I_i ∝ rank^{slope:.2f})")

# ============ SAVE EVERYTHING ============
results = {
    "layer": args.layer,
    "model": MODEL_NAME,
    "sae_checkpoint": args.checkpoint,
    "measurement_tokens": total_positions,
    "measurement_time_min": round(elapsed_total / 60, 2),
    "avg_reconstruction_mse": round(avg_recon, 6),
    "sae_width": SAE_DIM,
    "dead_features": dead_features,
    "F_alive": F_alive,
    "avg_active_per_token": round(avg_active, 2),
    "alpha": round(alpha, 8),
    "g_alpha": round(g_alpha, 4),
    "d_star_S": round(d_star, 1),
    "total_importance": round(total_importance, 8),
    "predictions": predictions,
    "alive_threshold": args.alive_threshold,
    # Save full importance distribution for plotting
    "sorted_importance": sorted_imp.tolist(),
    # Save feature frequencies for analysis
    "feature_frequencies_alive": freq[alive_mask].tolist(),
}

out_path = f"results/measurements_layer{args.layer}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to: {out_path}")
print(f"\n{'='*65}")
print(f"  ✅ MEASUREMENT COMPLETE — Layer {args.layer}")
print(f"{'='*65}")
print(f"\n  Use these predictions to compare against Experiment 3 (distillation).")
print(f"  Key number: d*_S = {d_star:.0f} (critical student width)")
print(f"  Students narrower than {d_star:.0f} WILL have a loss floor.")
print(f"  Students wider than {d_star:.0f} SHOULD approach zero loss.")
