# Experiment 3: Knowledge Distillation on Pythia-410M

## Status: 🔄 In Progress

## Purpose
Test whether the formula's predictions (from Experiment 2's SAE measurements) match actual distillation behavior. This is the **headline experiment** — it bridges theory to practice.

## Setup

**Teacher:** Pythia-410M (frozen, 1024 hidden dim, 24 layers)

**Students:** Same depth (24 layers), variable hidden width:

| Student | Hidden Dim | ~Params | Relation to d*_S | Expected Floor |
|---------|-----------|---------|------------------|----------------|
| A | 128 | ~5M | Way below d*_S | Large floor |
| B | 256 | ~20M | Below d*_S | Moderate floor |
| C | 512 | ~70M | Below d*_S | Smaller floor |
| D | 768 | ~150M | Near d*_S | Small floor |
| E (control) | 1024 | ~410M | Same as teacher | Near-zero floor |

**Distillation method:** Standard KL divergence with temperature scaling

```
loss = KL(softmax(teacher_logits/T), log_softmax(student_logits/T)) × T²
T = 2.0, optimizer = AdamW, lr = 3e-4, steps = 30,000
```

## Predicted vs Actual (Layer 12 predictions)

| d_S | Predicted Floor | Actual Floor | Match? |
|-----|----------------|--------------|--------|
| 128 | 0.0795 | (pending) | |
| 256 | 0.0400 | (pending) | |
| 512 | 0.0111 | (pending) | |
| 768 | 0.0016 | (pending) | |
| 1024 | 0.0001 | (pending) | |

## Files

### Scripts
- `distill_student.py` — Train one student at any width
- `plot_exp3.py` — Generate headline figure (predicted vs actual)

### Results (in `results/exp3_distillation/`)
- `distill_w{width}_s{seed}.json` — Loss curves and floor estimates per student

## Reproduction

```bash
# Train student at width 256
python distill_student.py --width 256 --num_steps 30000

# Train narrowest student with 3 seeds (for error bars)
python distill_student.py --width 128 --seed 0
python distill_student.py --width 128 --seed 1
python distill_student.py --width 128 --seed 2
```
