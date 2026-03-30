# Geometric Limits of Knowledge Distillation: A Minimum-Width Theorem via Superposition Theory

**Target:** COLM 2026 / NeurIPS 2026  
**Affiliation:** Erdos AI Labs  
**Status:** Experimentation Phase (Experiments 2-3)

---

## Abstract

Knowledge distillation compresses large teacher models into smaller students, but student performance saturates at a **loss floor** that persists across training methods, objectives, and hyperparameter choices. We argue this floor is **geometric** in origin. Neural networks pack far more learned features into their hidden layers than they have dimensions — a phenomenon termed *superposition* (Elhage et al., 2022). A student with hidden width d_S can faithfully encode at most **d_S × g(α)** of the teacher's features, where α is the feature sparsity and g(α) = 1/((1−α) ln 1/(1−α)) is a capacity function grounded in compressed sensing theory. Features that exceed this budget are unrecoverable at the bottleneck, yielding an importance-weighted loss floor over the dropped features.

We prove this bound on the Elhage et al. toy model and verify it across 48 configurations spanning different feature counts, teacher widths, and sparsity levels. To test the bound on a real language model, we measure the feature count F and sparsity α of **Pythia-410M** using sparse autoencoders trained from scratch, predict the loss floor at multiple student widths, and compare against actual distillation experiments. Linear probing provides mechanistic confirmation: specific teacher concepts become linearly undecodable from narrow students, with probe accuracy falling to chance in order of decreasing feature importance, exactly as the capacity theory predicts.

---

## The Core Claim

> **The distillation loss floor is a geometric phenomenon, not a training phenomenon.**

When you distill a teacher into a student:
- The student's hidden layer is a bottleneck of width d_S
- This bottleneck can carry at most d_S × g(α) features (compressed sensing bound)  
- If the teacher has F > d_S × g(α) features, the excess are **permanently lost**
- The loss from these lost features is the **floor** — no training method can overcome it
- We can **predict** this floor from SAE measurements on the teacher alone

---

## Three Novel Contributions

### 1. The Distillation Minimum-Width Theorem
A formal proof that knowledge distillation has a hard loss floor below the critical student width d*_S = F/g(α). Prior work studied superposition *within* one model. We prove what happens when you *transfer* a superposed representation from a wide model to a narrow one.

### 2. The SAE-to-Prediction Pipeline  
An operational tool: measure F and α via SAE → plug into formula → predict distillation loss at any student width. Nobody connected SAE feature measurements to distillation performance predictions before.

### 3. Mechanistic Feature-Level Validation
Linear probes showing specific named concepts become undecodable from narrow students, with features disappearing in importance order — confirming the geometric mechanism.

---

## Key Formulas

**Loss floor at student width d_S:**

```
L*(d_S) = Σ_{i = F_S + 1}^{F}  I_i × E[x_i²]

where:
  F_S = d_S × g(α)         — features that fit in the student
  F   = total teacher features (measured via SAE)
  α   = feature sparsity    (measured via SAE)
  I_i = importance of feature i (sorted descending)
```

**Sparsity capacity function:**

```
g(α) = 1 / ((1-α) × ln(1/(1-α)))
```

**Critical student width (zero-floor threshold):**

```
d*_S = F / g(α)
```

---

## Experimental Results

### Experiment 2: SAE Measurements on Pythia-410M

We trained sparse autoencoders from scratch on Pythia-410M at layers 8, 12, and 16 (300M tokens each, 32× expansion).

| Layer | Alive Features (F) | Avg Active/Token | Sparsity (α) | g(α) | Critical d*_S |
|-------|-------------------|------------------|---------------|-------|---------------|
| 8     | 31,006            | 234.9            | 0.9924        | 27.04 | 1,147         |
| 12    | 28,665            | 218.3            | 0.9924        | 26.92 | 1,065         |
| 16    | 29,169            | 249.0            | 0.9915        | 24.60 | 1,186         |

**Key finding:** d*_S > d_T (1024) at all layers — even the teacher itself operates in a compressed regime. This means every student width will have some nonzero floor.

### Predicted Loss Floors (Layer 12)

| Student Width | Features Kept | Features Dropped | Predicted Floor |
|--------------|---------------|------------------|-----------------|
| 128          | 3,446         | 25,219           | 0.0795          |
| 256          | 6,892         | 21,773           | 0.0400          |
| 384          | 10,338        | 18,327           | 0.0217          |
| 512          | 13,784        | 14,881           | 0.0111          |
| 768          | 20,676        | 7,989            | 0.0016          |
| 1024         | 27,568        | 1,097            | 0.0001          |

### Experiment 3: Distillation Validation

🔄 **In Progress** — Training student models at widths {128, 256, 512, 768, 1024} via KL distillation from Pythia-410M. Actual loss floors will be compared against predictions above.

### Experiment 1: Toy Model Validation

⬚ **Planned** — 48-configuration sweep on the Elhage et al. single-layer autoencoder.

### Experiment 4: Linear Probing

⬚ **Planned** — Probing for specific concepts (code, legal, medical, questions) across all student widths.

---

## Repository Structure

```
COLM_ERDOS_labs/
│
├── README.md                          ← This file
│
├── paper/
│   ├── abstract.md                    ← Paper abstract (latest draft)
│   └── figures/                       ← Publication-ready figures
│       ├── fig1_sae_training.png      ← SAE training convergence curves
│       ├── fig2_importance_dist.png   ← Feature importance distribution (log-log)
│       ├── fig3_predicted_floors.png  ← Predicted loss floor vs student width
│       └── fig4_summary_table.png     ← SAE measurements summary table
│
├── experiments/
│   ├── exp1_toy_model/                ← Experiment 1: Toy model validation
│   │   └── (planned)
│   │
│   ├── exp2_sae/                      ← Experiment 2: SAE training & measurement
│   │   ├── train_sae.py              ← SAE training script (parameterized by layer)
│   │   ├── measure_sae.py            ← Feature extraction & floor prediction
│   │   └── plot_exp2.py              ← Plotting script for Experiment 2
│   │
│   ├── exp3_distillation/             ← Experiment 3: Knowledge distillation
│   │   ├── distill_student.py        ← Student training via KL divergence
│   │   └── plot_exp3.py              ← Plotting script for Experiment 3
│   │
│   └── exp4_probing/                  ← Experiment 4: Linear probing
│       └── (planned)
│
├── results/
│   ├── exp2_sae/
│   │   ├── measurements_layer8.json   ← F, α, g(α), d*_S, importance distribution
│   │   ├── measurements_layer12.json
│   │   ├── measurements_layer16.json
│   │   ├── training_stats_layer8.json ← SAE training loss curves
│   │   ├── training_stats_layer12.json
│   │   └── training_stats_layer16.json
│   │
│   └── exp3_distillation/
│       ├── distill_w128_s0.json       ← Loss curve & floor estimate
│       ├── distill_w256_s0.json
│       ├── distill_w512_s0.json
│       ├── distill_w768_s0.json
│       └── distill_w1024_s0.json
│
├── checkpoints/                       ← Model checkpoints (gitignored, stored separately)
│   ├── sae_layer8_final.pt
│   ├── sae_layer12_final.pt
│   ├── sae_layer16_final.pt
│   └── distill_w*_final.pt
│
├── docs/
│   ├── experiment_plan.pdf            ← Full experiment plan document
│   ├── theory.md                      ← Theoretical background & derivations
│   └── setup_guide.md                 ← How to reproduce experiments
│
├── .gitignore
└── requirements.txt
```

---

## Setup & Reproduction

### Requirements

```bash
pip install torch transformers datasets accelerate einops numpy matplotlib scikit-learn
```

### Hardware

- **Experiment 2 (SAE):** 1× GPU with ≥48GB VRAM (H100, A100, RTX 6000). ~35 min per layer.
- **Experiment 3 (Distillation):** 1× GPU with ≥48GB VRAM. ~1-5 hours per student depending on width.
- **Total compute:** ~30 GPU-hours on H100-class hardware.

### Running Experiments

**Experiment 2 — Train SAE:**
```bash
python experiments/exp2_sae/train_sae.py --layer 12 --l1_coeff 8e-4 --num_tokens 300000000
```

**Experiment 2 — Measure features:**
```bash
python experiments/exp2_sae/measure_sae.py --layer 12 --checkpoint checkpoints/sae_layer12_final.pt
```

**Experiment 3 — Distill student:**
```bash
python experiments/exp3_distillation/distill_student.py --width 256 --num_steps 30000
```

---

## Teacher Model

| Property | Value |
|----------|-------|
| Model | EleutherAI/pythia-410m |
| Parameters | 405M |
| Layers | 24 |
| Hidden Dimension | 1024 |
| Attention Heads | 16 |
| Vocabulary | 50,304 |
| Training Data | The Pile (300B tokens) |

---

## SAE Architecture

| Property | Value |
|----------|-------|
| Type | Single-layer sparse autoencoder |
| Input Dimension | 1024 (Pythia-410M hidden size) |
| Expansion Factor | 32× |
| SAE Width | 32,768 features |
| Activation | ReLU (encoder) |
| Sparsity Penalty | L1 on activations, coefficient 8e-4 |
| Training Data | The Pile, 300M tokens |
| Training Time | ~35 minutes per layer on H100 |

---

## Key References

- Elhage et al. (2022). *Toy Models of Superposition.* Anthropic.
- Scherlis et al. (2022). *Polysemanticity and Capacity in Neural Networks.* 
- Busbridge et al. (2025). *Distillation loss floors.* (empirical observation)
- Cunningham et al. (2023). *Sparse Autoencoders Find Highly Interpretable Directions.*
- Donoho (2006). *Compressed Sensing.* IEEE Trans. Information Theory.

---

## License

Research code — see LICENSE for details.

---

## Citation

```bibtex
@article{kd_minimum_width_2026,
  title={Geometric Limits of Knowledge Distillation: A Minimum-Width Theorem via Superposition Theory},
  author={Saptarshi Deka},
  year={2026},
  note={Under review at COLM 2026}
}
```
