# Experiment 2: SAE Training & Measurement on Pythia-410M

## Status: ✅ Complete

## Purpose
Measure the teacher's internal feature structure (F, α, importance distribution) to generate quantitative predictions of distillation loss floors.

## Key Result

| Layer | Alive Features (F) | Avg Active/Token (L0) | Sparsity (α) | g(α) | Critical d*_S |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 8 | 31,006 | 234.9 | 0.9924 | 27.04 | 1,147 |
| 12 | 28,665 | 218.3 | 0.9924 | 26.92 | 1,065 |
| 16 | 29,169 | 249.0 | 0.9915 | 24.60 | 1,186 |

**Notable:** d*_S > d_T (1024) at all layers — the teacher itself is under capacity.

## Files

### Scripts
- `train_sae.py` — SAE training (parameterized by layer)
- `measure_sae.py` — Feature extraction and floor prediction

### Results (in `results/exp2_sae/`)
- `measurements_layer{8,12,16}.json` — Full measurements including importance distributions
- `training_stats_layer{8,12,16}.json` — Training loss curves

## Reproduction

```bash
# Train SAE on layer 12 (300M tokens, ~35 min on H100)
python train_sae.py --layer 12 --l1_coeff 8e-4 --num_tokens 300000000

# Measure features (~10 min)
python measure_sae.py --layer 12 --checkpoint ../../checkpoints/sae_layer12_final.pt
```

## SAE Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Expansion factor | 32× | 1024 → 32,768 features |
| L1 coefficient | 8e-4 | Tuned for L0 ≈ 100-300 |
| Learning rate | 3e-4 | Adam optimizer |
| Training tokens | 300M | From The Pile |
| Data source | EleutherAI/the_pile_deduplicated | Same distribution as Pythia training |
| L1 formulation | `z.abs().sum(dim=-1).mean()` | Sum over features, mean over batch |
