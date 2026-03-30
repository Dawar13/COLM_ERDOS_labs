#!/usr/bin/env python3
"""
train_sae.py — Train a Sparse Autoencoder on Pythia-410M at a specified layer.

Usage:
    python3 train_sae.py --layer 8
    python3 train_sae.py --layer 12 --num_tokens 300000000 --l1_coeff 3e-4
    python3 train_sae.py --layer 8 --resume checkpoints/sae_layer8_150M.pt

Typical training time: ~45-60 min on H100 for 300M tokens.
"""

import argparse
import os
import sys
import time
import json
import math
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ ARGUMENT PARSING ============
parser = argparse.ArgumentParser(description="Train SAE on Pythia-410M")
parser.add_argument("--layer", type=int, required=True, help="Target layer (e.g., 8, 12, 16)")
parser.add_argument("--num_tokens", type=int, default=300_000_000, help="Total training tokens (default 300M)")
parser.add_argument("--expansion", type=int, default=32, help="SAE expansion factor (default 32x)")
parser.add_argument("--l1_coeff", type=float, default=3e-4, help="L1 sparsity coefficient")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size in sequences")
parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length in tokens")
parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint directory")
parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
args = parser.parse_args()

# ============ CONFIGURATION ============
MODEL_NAME = "EleutherAI/pythia-410m"
HIDDEN_DIM = 1024  # Pythia-410M hidden size
SAE_DIM = HIDDEN_DIM * args.expansion
TOKENS_PER_BATCH = args.batch_size * args.seq_len
NUM_BATCHES = args.num_tokens // TOKENS_PER_BATCH
DEVICE = "cuda"
DTYPE = torch.bfloat16
LOG_EVERY = 50          # Log every N batches
SAVE_EVERY_TOKENS = 50_000_000  # Checkpoint every 50M tokens

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs("results", exist_ok=True)

print(f"\n{'='*65}")
print(f"  SAE TRAINING — Pythia-410M Layer {args.layer}")
print(f"{'='*65}")
print(f"  Model:           {MODEL_NAME}")
print(f"  Target layer:    {args.layer} / 24")
print(f"  SAE dimensions:  {HIDDEN_DIM} → {SAE_DIM} ({args.expansion}×)")
print(f"  L1 coefficient:  {args.l1_coeff}")
print(f"  Learning rate:   {args.lr}")
print(f"  Batch:           {args.batch_size} seqs × {args.seq_len} tok = {TOKENS_PER_BATCH:,} tok/batch")
print(f"  Total tokens:    {args.num_tokens/1e6:.0f}M ({NUM_BATCHES:,} batches)")
print(f"  Device:          {DEVICE} ({DTYPE})")
print(f"{'='*65}\n")


# ============ SAE MODEL ============
class SparseAutoencoder(nn.Module):
    """
    Single-layer sparse autoencoder for mechanistic interpretability.
    Encodes hidden states into a sparse overcomplete basis.
    """
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        # Pre-encoder bias (centers the input distribution)
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        # Encoder: d_model → d_sae (with ReLU activation for sparsity)
        self.encoder = nn.Linear(d_model, d_sae, bias=True)

        # Decoder: d_sae → d_model (reconstructs the hidden state)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

        # Initialize decoder columns as unit vectors
        # This is important — without it, decoder norms drift and training is unstable
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode hidden state → sparse feature activations."""
        return F.relu(self.encoder(x - self.b_pre))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse activations → reconstructed hidden state."""
        return self.decoder(z) + self.b_pre

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ============ DATA PIPELINE ============
def create_data_iterator(tokenizer, batch_size, seq_len, skip_batches=0):
    """
    Stream tokens from The Pile (Pythia's training data).
    Yields batches of shape [batch_size, seq_len].
    """
    from datasets import load_dataset

    print("  Loading data stream from The Pile...")

    # Try multiple data sources in order of preference
    data_sources = [
        ("monology/pile-uncopyrighted", {}),
        ("EleutherAI/the_pile_deduplicated", {}),
        ("NeelNanda/pile-10k", {}),  # Tiny fallback for testing
    ]

    dataset = None
    for source_name, kwargs in data_sources:
        try:
            print(f"  Trying: {source_name}...")
            dataset = load_dataset(
                source_name,
                split="train",
                streaming=True,
                trust_remote_code=True,
                **kwargs,
            )
            # Test that we can read from it
            sample = next(iter(dataset))
            if "text" in sample:
                print(f"  ✓ Using: {source_name}")
                # Re-create iterator since we consumed one element
                dataset = load_dataset(
                    source_name,
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                    **kwargs,
                )
                break
            else:
                print(f"  ✗ No 'text' field in {source_name}")
                dataset = None
        except Exception as e:
            print(f"  ✗ Failed: {source_name} — {e}")
            dataset = None

    if dataset is None:
        print("\n  ❌ ERROR: Could not load any data source!")
        print("  Try running: pip install datasets --upgrade")
        sys.exit(1)

    # Tokenize and yield batches
    buffer = []
    needed = batch_size * seq_len
    batches_yielded = 0
    batches_skipped = 0

    for example in dataset:
        text = example.get("text", "")
        if not text or len(text.strip()) < 50:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(tokens)

        while len(buffer) >= needed:
            # Skip batches if resuming
            if batches_skipped < skip_batches:
                buffer = buffer[needed:]
                batches_skipped += 1
                if batches_skipped % 1000 == 0:
                    print(f"  Skipping to resume position: {batches_skipped}/{skip_batches}...")
                continue

            batch = torch.tensor(buffer[:needed], dtype=torch.long).reshape(batch_size, seq_len)
            buffer = buffer[needed:]
            batches_yielded += 1
            yield batch


# ============ LOAD PYTHIA-410M ============
print("[1/3] Loading Pythia-410M...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="cuda",
)
model.eval()
for param in model.parameters():
    param.requires_grad = False

num_layers = model.config.num_hidden_layers
assert args.layer < num_layers, f"Layer {args.layer} invalid. Model has {num_layers} layers (0-{num_layers-1})."
print(f"  ✓ Loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, {num_layers} layers")


# ============ ACTIVATION HOOK ============
activation_cache = {}

def hook_fn(module, input, output):
    """Capture hidden state output from the target layer."""
    if isinstance(output, tuple):
        activation_cache["h"] = output[0]
    else:
        activation_cache["h"] = output

hook_handle = model.gpt_neox.layers[args.layer].register_forward_hook(hook_fn)
print(f"  ✓ Hook on layer {args.layer}")


# ============ INITIALIZE SAE ============
print(f"\n[2/3] Initializing SAE ({HIDDEN_DIM} → {SAE_DIM})...")
sae = SparseAutoencoder(HIDDEN_DIM, SAE_DIM).to(DEVICE).to(DTYPE)
optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999))

# Track tokens seen (for resuming)
tokens_seen = 0
start_batch = 0

# Resume from checkpoint if specified
if args.resume and os.path.exists(args.resume):
    print(f"  Resuming from: {args.resume}")
    ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
    sae.load_state_dict(ckpt["model_state"])
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    tokens_seen = ckpt.get("config", {}).get("tokens_seen", 0)
    start_batch = tokens_seen // TOKENS_PER_BATCH
    print(f"  ✓ Resumed at {tokens_seen/1e6:.0f}M tokens (batch {start_batch})")

sae_params = sum(p.numel() for p in sae.parameters())
print(f"  ✓ SAE params: {sae_params/1e6:.1f}M")


# ============ TRAINING LOOP ============
print(f"\n[3/3] Starting training...")
print(f"  {'Batch':>8} | {'Tokens':>8} | {'Recon':>8} | {'L1':>7} | {'L0':>6} | {'Alive':>12} | {'Speed':>10} | {'ETA':>6}")
print(f"  {'-'*85}")

start_time = time.time()
stats_history = []
best_recon = float('inf')

# Create data iterator (skips batches if resuming)
data_iter = create_data_iterator(tokenizer, args.batch_size, args.seq_len, skip_batches=start_batch)

for batch_idx in range(start_batch, NUM_BATCHES):
    # ---- GET DATA ----
    try:
        input_ids = next(data_iter).to(DEVICE)
    except StopIteration:
        print(f"\n  ⚠ Data stream exhausted at batch {batch_idx}. Restarting stream...")
        data_iter = create_data_iterator(tokenizer, args.batch_size, args.seq_len, skip_batches=0)
        input_ids = next(data_iter).to(DEVICE)

    # ---- TEACHER FORWARD (frozen) ----
    with torch.no_grad():
        model(input_ids)
        hidden = activation_cache["h"]  # [batch, seq, 1024]

    # Reshape: [batch × seq, hidden_dim]
    hidden_flat = hidden.reshape(-1, HIDDEN_DIM).detach()

    # ---- SAE FORWARD ----
    x_hat, z = sae(hidden_flat)

    # ---- COMPUTE LOSSES ----
    recon_loss = F.mse_loss(x_hat, hidden_flat)
    l1_loss = z.abs().mean()
    total_loss = recon_loss + args.l1_coeff * l1_loss

    # ---- CHECK FOR NaN ----
    if torch.isnan(total_loss):
        print(f"\n  ❌ NaN detected at batch {batch_idx}! Saving emergency checkpoint...")
        torch.save({"model_state": sae.state_dict(), "config": {"tokens_seen": tokens_seen}},
                    os.path.join(args.save_dir, f"sae_layer{args.layer}_EMERGENCY.pt"))
        print("  Try restarting with --lr 1e-4 or --l1_coeff 1e-4")
        sys.exit(1)

    # ---- BACKWARD + UPDATE ----
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
    optimizer.step()

    # Normalize decoder weights to unit vectors (prevents drift)
    with torch.no_grad():
        sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=0)

    tokens_seen += TOKENS_PER_BATCH

    # ---- LOGGING ----
    if batch_idx % LOG_EVERY == 0 or batch_idx == start_batch:
        with torch.no_grad():
            l0 = (z > 0).float().sum(dim=-1).mean().item()
            alive_in_batch = int((z > 0).any(dim=0).sum().item())
            frac_alive = alive_in_batch / SAE_DIM

        elapsed = time.time() - start_time
        tok_sec = (tokens_seen - start_batch * TOKENS_PER_BATCH) / max(elapsed, 1)
        remaining = args.num_tokens - tokens_seen
        eta_min = remaining / max(tok_sec, 1) / 60

        print(
            f"  {batch_idx:>8,} | "
            f"{tokens_seen/1e6:>6.0f}M | "
            f"{recon_loss.item():>8.5f} | "
            f"{l1_loss.item():>7.4f} | "
            f"{l0:>5.0f}  | "
            f"{alive_in_batch:>5}/{SAE_DIM} ({frac_alive:>4.1%}) | "
            f"{tok_sec/1e6:>6.2f}M/s | "
            f"{eta_min:>4.0f}m"
        )

        stats_history.append({
            "batch": batch_idx,
            "tokens_M": round(tokens_seen / 1e6, 1),
            "recon_loss": round(recon_loss.item(), 6),
            "l1_loss": round(l1_loss.item(), 5),
            "L0": round(l0, 1),
            "alive": alive_in_batch,
            "frac_alive": round(frac_alive, 4),
        })

        # Health warnings
        if frac_alive < 0.15 and batch_idx > 200:
            print(f"  ⚠ WARNING: Only {frac_alive:.0%} features alive! Consider reducing --l1_coeff")
        if recon_loss.item() > 1.0 and batch_idx > 500:
            print(f"  ⚠ WARNING: Recon loss still high. Check if --l1_coeff is too large")
        if l0 < 10 and batch_idx > 200:
            print(f"  ⚠ WARNING: L0 very low ({l0:.0f}). SAE too sparse — reduce --l1_coeff")

        # Track best
        if recon_loss.item() < best_recon:
            best_recon = recon_loss.item()

    # ---- CHECKPOINT ----
    if tokens_seen % SAVE_EVERY_TOKENS < TOKENS_PER_BATCH:
        ckpt_path = os.path.join(args.save_dir, f"sae_layer{args.layer}_{int(tokens_seen/1e6)}M.pt")
        torch.save({
            "model_state": sae.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "d_model": HIDDEN_DIM,
                "d_sae": SAE_DIM,
                "expansion": args.expansion,
                "l1_coeff": args.l1_coeff,
                "lr": args.lr,
                "target_layer": args.layer,
                "tokens_seen": tokens_seen,
                "model_name": MODEL_NAME,
            },
            "stats": stats_history,
        }, ckpt_path)
        print(f"  💾 Checkpoint: {ckpt_path}")

    # Clear cache to prevent memory buildup
    if batch_idx % 500 == 0:
        gc.collect()
        torch.cuda.empty_cache()


# ============ FINAL SAVE ============
hook_handle.remove()

final_path = os.path.join(args.save_dir, f"sae_layer{args.layer}_final.pt")
torch.save({
    "model_state": sae.state_dict(),
    "config": {
        "d_model": HIDDEN_DIM,
        "d_sae": SAE_DIM,
        "expansion": args.expansion,
        "l1_coeff": args.l1_coeff,
        "lr": args.lr,
        "target_layer": args.layer,
        "tokens_seen": tokens_seen,
        "model_name": MODEL_NAME,
    },
    "stats": stats_history,
}, final_path)

# Save training stats as JSON
stats_path = f"results/training_stats_layer{args.layer}.json"
with open(stats_path, "w") as f:
    json.dump({
        "config": {
            "model": MODEL_NAME,
            "layer": args.layer,
            "expansion": args.expansion,
            "l1_coeff": args.l1_coeff,
            "lr": args.lr,
            "total_tokens": tokens_seen,
        },
        "training_curves": stats_history,
    }, f, indent=2)

total_hours = (time.time() - start_time) / 3600

print(f"\n{'='*65}")
print(f"  ✅ SAE TRAINING COMPLETE — Layer {args.layer}")
print(f"{'='*65}")
print(f"  Total tokens:       {tokens_seen/1e6:.0f}M")
print(f"  Total time:         {total_hours:.2f} hours")
print(f"  Best recon loss:    {best_recon:.5f}")
if stats_history:
    print(f"  Final recon loss:   {stats_history[-1]['recon_loss']:.5f}")
    print(f"  Final L0:           {stats_history[-1]['L0']:.0f}")
    print(f"  Final alive:        {stats_history[-1]['alive']}/{SAE_DIM}")
print(f"  Model saved to:     {final_path}")
print(f"  Stats saved to:     {stats_path}")
print(f"{'='*65}")
print(f"\n  NEXT STEP: python3 measure_sae.py --layer {args.layer} --checkpoint {final_path}")
