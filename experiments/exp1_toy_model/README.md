# Experiment 1: Toy Model Validation

## Status: Planned

## Purpose
Prove that the loss floor exists in a controlled setting where we know the ground truth (F, α, importance distribution). Validate that our formula predicts its location.

## Setup
- Elhage et al. (2022) single-layer autoencoder
- Input: x ∈ ℝⁿ (n-dimensional, sparse)
- Encode: h = W × x (h ∈ ℝᵈ)
- Decode: x̂ = ReLU(Wᵀ × h + b)
- Loss: ||x - x̂||²

## Configuration Sweep
- n (features): [10, 20, 40]
- d_T (teacher width): [3, 5, 8, 10]
- α (sparsity): [0.80, 0.90, 0.95, 0.99]
- Total: 48 configurations
- Seeds per config: 20

## Expected Outputs
- Figure: Loss floor curves (loss vs d_S) for multiple configurations
- Figure: Predicted vs actual floor scatter plot (R² metric)
- Figure: Naive vs refined formula comparison
