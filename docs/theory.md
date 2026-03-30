# Theoretical Background & Derivations

## 1. The Superposition Framework

Modern neural networks represent **F >> d** features in a d-dimensional hidden space by exploiting **sparsity**. If features are rarely active simultaneously, they can share dimensions through non-orthogonal directions with manageable interference.

**Key insight (Elhage et al., 2022):** A model with hidden dimension d stores F features, where F can be 10-30× larger than d. This "superposition" is possible because features are sparse — on any given input, only a small fraction (1-α) are active.

## 2. The Capacity Function g(α)

From compressed sensing theory (Donoho, 2006; Candès & Tao, 2005):

A d-dimensional space can faithfully represent up to **d × g(α)** sparse features, where:

```
g(α) = 1 / ((1-α) × ln(1/(1-α)))
```

Reference values:

| α (sparsity) | 1-α (active fraction) | g(α) | Features per dimension |
|:---:|:---:|:---:|:---:|
| 0.90 | 0.100 | 4.3 | 4.3× |
| 0.95 | 0.050 | 6.7 | 6.7× |
| 0.99 | 0.010 | 21.7 | 21.7× |
| 0.992 | 0.008 | 27.0 | 27.0× |
| 0.999 | 0.001 | 145 | 145× |

This is a **phase transition**: below d × g(α) features, recovery is clean. Above it, recovery fails catastrophically.

## 3. The Bottleneck Argument

During distillation, the student's hidden layer of width d_S is a **bottleneck**. All information must pass through this layer.

**Theorem (informal):** A bottleneck of width d_S can transmit at most d_S × g(α) features. Features beyond this limit are destroyed, and by the **data processing inequality**, downstream layers cannot recover them.

**Proof sketch:**
1. The student's hidden layer has rank ≤ d_S (linear algebra)
2. With sparsity α, the effective capacity is d_S × g(α) (compressed sensing)
3. If F > d_S × g(α), then F − d_S × g(α) features cannot pass through
4. The data processing inequality (information theory) says destroyed information is irrecoverable
5. This holds for EVERY possible student encoder — it's a property of dimensional limits

## 4. The Loss Floor Formula

### Naive Formula (Uniform Importance)

If all features are equally important:

```
L*(d_S) = ε_∞ × (1 - d_S × g(α) / F)
```

Where ε_∞ is the maximum possible distillation loss.

### Refined Formula (Importance-Weighted)

Features have different importances. The model keeps the most important features and drops the rest:

```
L*(d_S) = Σ_{i = F_S + 1}^{F}  I_i × E[x_i²]
```

Where:
- Features are sorted by importance: I_1 ≥ I_2 ≥ ... ≥ I_F
- F_S = d_S × g(α) features are **kept** (most important)
- F − F_S features are **dropped** (least important)
- The floor equals the total importance of dropped features

## 5. The Critical Width

```
d*_S = F / g(α)
```

This is the minimum student width for zero-floor distillation. Below d*_S, a nonzero floor is **guaranteed**. Above d*_S, the floor vanishes (in principle).

## 6. Important Caveats

### Phase Transition Width
The capacity threshold d_S × g(α) is a **phase transition**, not a knife edge. The transition occurs over a band of approximately ±10-20% around the predicted threshold. Our formula uses a sharp cutoff approximation, introducing ~5-15% error.

### Lower Bound Nature
The formula gives a **minimum** loss. The actual loss may be higher if the student fails to achieve optimal capacity allocation. But it can NEVER be lower — this is the hard geometric limit.

### Multi-Layer Considerations
Real transformers have multiple layers. A feature lost at one layer might theoretically be reconstructed at a subsequent layer. We assume this does not occur for truly independent features, consistent with the data processing inequality. The multi-layer prediction is technically a **conjecture** validated empirically (Experiment 3).
