# Setup & Reproduction Guide

## Hardware Requirements

| Experiment | GPU Memory | GPU Time | Notes |
|-----------|-----------|----------|-------|
| Exp 1: Toy Model | None (CPU only) | ~30 min | Runs on laptop |
| Exp 2: SAE Training | ≥48 GB | ~35 min/layer | H100, A100, or RTX 6000 |
| Exp 2: SAE Measurement | ≥48 GB | ~10 min/layer | Same GPU as training |
| Exp 3: Distillation | ≥48 GB | 1-5 hrs/student | Depends on student width |
| Exp 4: Linear Probing | ≥48 GB | ~2 hrs total | Logistic regression on CPU |

**Total compute budget:** ~30 GPU-hours on H100-class hardware.

## Software Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/COLM_ERDOS_labs.git
cd COLM_ERDOS_labs

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## Running Experiments

### Experiment 2: SAE Training & Measurement

**Step 1: Train SAE on a specific layer**
```bash
python experiments/exp2_sae/train_sae.py --layer 12 --l1_coeff 8e-4 --num_tokens 300000000
```

Key hyperparameters:
- `--l1_coeff 8e-4`: Sparsity penalty. Tuned to produce L0 ≈ 100-300 (features active per token)
- `--num_tokens 300000000`: 300M tokens from The Pile. Standard for SAE training.
- `--expansion 32`: 32× expansion (1024 → 32,768 SAE features)

**What to watch during training:**
```
HEALTHY:  L0 = 100-300, alive > 60%, recon < 0.2
TOO DENSE: L0 > 1000 → increase --l1_coeff
TOO SPARSE: L0 < 20 → decrease --l1_coeff
```

**Step 2: Measure features**
```bash
python experiments/exp2_sae/measure_sae.py --layer 12 --checkpoint checkpoints/sae_layer12_final.pt
```

This outputs: F (alive features), α (sparsity), g(α), d*_S, and predicted loss floors.

**Step 3: Repeat for layers 8 and 16**
```bash
python experiments/exp2_sae/train_sae.py --layer 8 --l1_coeff 8e-4 --num_tokens 300000000
python experiments/exp2_sae/measure_sae.py --layer 8 --checkpoint checkpoints/sae_layer8_final.pt

python experiments/exp2_sae/train_sae.py --layer 16 --l1_coeff 8e-4 --num_tokens 300000000
python experiments/exp2_sae/measure_sae.py --layer 16 --checkpoint checkpoints/sae_layer16_final.pt
```

### Experiment 3: Knowledge Distillation

**Train students at different widths:**
```bash
# Narrowest student (3 seeds for error bars)
python experiments/exp3_distillation/distill_student.py --width 128 --seed 0
python experiments/exp3_distillation/distill_student.py --width 128 --seed 1
python experiments/exp3_distillation/distill_student.py --width 128 --seed 2

# Other widths (1 seed each)
python experiments/exp3_distillation/distill_student.py --width 256
python experiments/exp3_distillation/distill_student.py --width 512
python experiments/exp3_distillation/distill_student.py --width 768
python experiments/exp3_distillation/distill_student.py --width 1024
```

### Parallel Execution (Multiple GPUs)

If you have multiple GPUs on one machine:
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python experiments/exp3_distillation/distill_student.py --width 256

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python experiments/exp3_distillation/distill_student.py --width 512
```

## Output Files

After all experiments, you should have:

```
results/exp2_sae/
  measurements_layer{8,12,16}.json    ← Core measurements (F, α, predictions)
  training_stats_layer{8,12,16}.json  ← Training curves

results/exp3_distillation/
  distill_w{128,256,512,768,1024}_s*.json  ← Loss curves & floor estimates

paper/figures/
  fig1_sae_training.png               ← SAE convergence
  fig2_importance_distribution.png     ← Feature importance (log-log)
  fig3_predicted_vs_actual.png         ← THE HEADLINE FIGURE
  fig4_loss_curves.png                 ← Distillation loss over time
```

## Cloud Setup (RunPod)

We used RunPod for GPU compute:
1. Deploy an H100 PCIe 80GB or RTX PRO 6000 96GB pod
2. Open Jupyter Lab from the pod dashboard
3. Upload experiment scripts to `/root/kd_experiment/`
4. Run experiments from the Jupyter terminal
5. Download results before stopping the pod

**Cost estimate:** ~$40-60 total across all experiments.
