#!/usr/bin/env python3
"""
plot_results.py — Generate all paper figures from Experiment 2 & 3 results.

Usage:
    python3 plot_results.py

Reads from results/ folder and generates publication-ready figures.
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox_inches': 'tight',
})

os.makedirs("figures", exist_ok=True)

# ============ LOAD ALL RESULTS ============
print("Loading results...")

# Load SAE measurements
sae_results = {}
for f in sorted(glob.glob("results/measurements_layer*.json")):
    with open(f) as fh:
        data = json.load(fh)
        layer = data["layer"]
        sae_results[layer] = data
        print(f"  ✓ SAE layer {layer}: F={data['F_alive']}, α={data['alpha']:.6f}, d*_S={data['d_star_S']:.0f}")

# Load distillation results
distill_results = {}
for f in sorted(glob.glob("results/distill_w*.json")):
    with open(f) as fh:
        data = json.load(fh)
        width = data["width"]
        seed = data["seed"]
        if width not in distill_results:
            distill_results[width] = []
        distill_results[width].append(data)
        print(f"  ✓ Distill d_S={width}, seed={seed}: floor≈{data.get('estimated_floor', '?')}")

# Load SAE training stats
sae_training = {}
for f in sorted(glob.glob("results/training_stats_layer*.json")):
    with open(f) as fh:
        data = json.load(fh)
        layer = data["config"]["layer"]
        sae_training[layer] = data
        print(f"  ✓ SAE training stats layer {layer}")

if not sae_results and not distill_results:
    print("\n  No results found in results/ folder. Run experiments first.")
    exit(0)


# ============ FIGURE 1: SAE TRAINING CURVES ============
if sae_training:
    print("\nGenerating Figure 1: SAE Training Curves...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = {8: '#2196F3', 12: '#4CAF50', 16: '#FF9800'}
    
    for layer, data in sorted(sae_training.items()):
        curves = data["training_curves"]
        tokens = [c["tokens_M"] for c in curves]
        recon = [c["recon_loss"] for c in curves]
        l0 = [c["L0"] for c in curves]
        alive = [c["alive"] for c in curves]
        color = colors.get(layer, '#666666')
        
        axes[0].plot(tokens, recon, color=color, label=f'Layer {layer}', linewidth=1.5)
        axes[1].plot(tokens, l0, color=color, label=f'Layer {layer}', linewidth=1.5)
        axes[2].plot(tokens, alive, color=color, label=f'Layer {layer}', linewidth=1.5)
    
    axes[0].set_xlabel('Tokens (M)')
    axes[0].set_ylabel('Reconstruction MSE')
    axes[0].set_title('Reconstruction Loss')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Tokens (M)')
    axes[1].set_ylabel('L0 (avg active features)')
    axes[1].set_title('Sparsity (L0)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Tokens (M)')
    axes[2].set_ylabel('Alive Features')
    axes[2].set_title('Feature Utilization')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/fig1_sae_training.png')
    plt.savefig('figures/fig1_sae_training.pdf')
    print("  ✓ Saved: figures/fig1_sae_training.png")


# ============ FIGURE 2: IMPORTANCE DISTRIBUTION ============
if sae_results:
    print("Generating Figure 2: Feature Importance Distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = {8: '#2196F3', 12: '#4CAF50', 16: '#FF9800'}
    
    for layer, data in sorted(sae_results.items()):
        imp = np.array(data["sorted_importance"])
        ranks = np.arange(1, len(imp) + 1)
        color = colors.get(layer, '#666666')
        ax.loglog(ranks, imp, color=color, label=f'Layer {layer} (F={data["F_alive"]})', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Feature Rank (by importance)')
    ax.set_ylabel('Importance (E[z²])')
    ax.set_title('Feature Importance Distribution — Pythia-410M')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/fig2_importance_distribution.png')
    plt.savefig('figures/fig2_importance_distribution.pdf')
    print("  ✓ Saved: figures/fig2_importance_distribution.png")


# ============ FIGURE 3: PREDICTED vs ACTUAL LOSS FLOOR (THE HEADLINE) ============
if sae_results and distill_results:
    print("Generating Figure 3: Predicted vs Actual Loss Floor (HEADLINE FIGURE)...")
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Use layer 12 as primary prediction (or whichever layer is available)
    primary_layer = 12 if 12 in sae_results else sorted(sae_results.keys())[0]
    sae_data = sae_results[primary_layer]
    
    # Plot predicted curve
    widths_pred = []
    floors_pred = []
    for w_str, pred in sorted(sae_data["predictions"].items(), key=lambda x: int(x[0])):
        widths_pred.append(int(w_str))
        floors_pred.append(pred["predicted_floor"])
    
    ax.plot(widths_pred, floors_pred, 'b-', linewidth=2, label=f'Predicted (Layer {primary_layer} SAE)', zorder=3)
    
    # Plot predictions from other layers as dashed lines
    colors = {8: '#2196F3', 12: '#4CAF50', 16: '#FF9800'}
    for layer, data in sorted(sae_results.items()):
        if layer == primary_layer:
            continue
        w = []
        f = []
        for w_str, pred in sorted(data["predictions"].items(), key=lambda x: int(x[0])):
            w.append(int(w_str))
            f.append(pred["predicted_floor"])
        ax.plot(w, f, '--', color=colors.get(layer, '#999'), linewidth=1.5, 
                label=f'Predicted (Layer {layer})', alpha=0.6, zorder=2)
    
    # Plot actual distillation results
    for width, runs in sorted(distill_results.items()):
        floors = [r["estimated_floor"] for r in runs if r.get("estimated_floor") is not None]
        if not floors:
            continue
        mean_floor = np.mean(floors)
        
        if len(floors) > 1:
            std_floor = np.std(floors)
            ax.errorbar(width, mean_floor, yerr=std_floor, fmt='ro', markersize=10,
                       capsize=5, capthick=2, linewidth=2, zorder=5, label='_nolegend_')
        else:
            ax.plot(width, mean_floor, 'ro', markersize=10, zorder=5, label='_nolegend_')
    
    # Add one label for actual points
    ax.plot([], [], 'ro', markersize=10, label='Actual (distillation)')
    
    # Plot d*_S vertical line
    d_star = sae_data["d_star_S"]
    ax.axvline(x=d_star, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'd*_S = {d_star:.0f}')
    
    ax.set_xlabel('Student Hidden Width (d_S)')
    ax.set_ylabel('Distillation Loss Floor')
    ax.set_title('Predicted vs Actual Distillation Loss Floor — Pythia-410M')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1100)
    
    plt.tight_layout()
    plt.savefig('figures/fig3_predicted_vs_actual.png')
    plt.savefig('figures/fig3_predicted_vs_actual.pdf')
    print("  ✓ Saved: figures/fig3_predicted_vs_actual.png (THE HEADLINE FIGURE)")


# ============ FIGURE 4: DISTILLATION LOSS CURVES ============
if distill_results:
    print("Generating Figure 4: Distillation Loss Curves...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cmap = plt.cm.viridis
    widths = sorted(distill_results.keys())
    
    for i, width in enumerate(widths):
        color = cmap(i / max(len(widths) - 1, 1))
        for run in distill_results[width]:
            if run.get("eval_losses"):
                steps = [e["step"] for e in run["eval_losses"]]
                losses = [e["eval_loss"] for e in run["eval_losses"]]
                label = f'd_S={width}' if run["seed"] == distill_results[width][0]["seed"] else '_nolegend_'
                ax.plot(steps, losses, color=color, linewidth=1.5, alpha=0.8, label=label)
            elif run.get("loss_history"):
                steps = [e["step"] for e in run["loss_history"]]
                losses = [e["loss"] for e in run["loss_history"]]
                label = f'd_S={width}' if run["seed"] == distill_results[width][0]["seed"] else '_nolegend_'
                ax.plot(steps, losses, color=color, linewidth=1.5, alpha=0.8, label=label)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Distillation Loss (KL Divergence)')
    ax.set_title('Distillation Loss Curves at Different Student Widths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/fig4_loss_curves.png')
    plt.savefig('figures/fig4_loss_curves.pdf')
    print("  ✓ Saved: figures/fig4_loss_curves.png")


# ============ FIGURE 5: FLOOR COMPARISON TABLE ============
if sae_results and distill_results:
    print("\nGenerating comparison table...")
    primary_layer = 12 if 12 in sae_results else sorted(sae_results.keys())[0]
    sae_data = sae_results[primary_layer]
    
    print(f"\n{'='*75}")
    print(f"  PREDICTED vs ACTUAL LOSS FLOORS (SAE Layer {primary_layer})")
    print(f"{'='*75}")
    print(f"  {'Width':<8} {'Predicted':<12} {'Actual':<12} {'Error':<12} {'% Error':<10} {'Match?'}")
    print(f"  {'-'*65}")
    
    comparison = []
    for w_str, pred in sorted(sae_data["predictions"].items(), key=lambda x: int(x[0])):
        width = int(w_str)
        predicted = pred["predicted_floor"]
        
        if width in distill_results:
            floors = [r["estimated_floor"] for r in distill_results[width] if r.get("estimated_floor")]
            if floors:
                actual = np.mean(floors)
                error = abs(predicted - actual)
                pct_error = (error / actual * 100) if actual > 0.001 else 0
                match = "✓" if pct_error < 30 else "~" if pct_error < 50 else "✗"
                print(f"  d_S={width:<4} {predicted:<12.6f} {actual:<12.6f} {error:<12.6f} {pct_error:<10.1f} {match}")
                comparison.append({
                    "width": width, "predicted": predicted, "actual": actual,
                    "error": error, "pct_error": pct_error
                })
            else:
                print(f"  d_S={width:<4} {predicted:<12.6f} {'(no data)':<12}")
        else:
            print(f"  d_S={width:<4} {predicted:<12.6f} {'(not run)':<12}")
    
    if comparison:
        avg_pct = np.mean([c["pct_error"] for c in comparison])
        print(f"\n  Average % error: {avg_pct:.1f}%")
        
        # Save comparison
        with open("results/predicted_vs_actual.json", "w") as f:
            json.dump(comparison, f, indent=2)
    
    print(f"{'='*75}")


# ============ SUMMARY ============
print(f"\n{'='*65}")
print(f"  ALL FIGURES GENERATED")
print(f"{'='*65}")
for f in sorted(glob.glob("figures/*.png")):
    print(f"  {f}")
print(f"{'='*65}")
