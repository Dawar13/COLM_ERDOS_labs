# Abstract

## Geometric Limits of Knowledge Distillation: A Minimum-Width Theorem via Superposition Theory

Knowledge distillation compresses large teacher models into smaller students, but student performance saturates at a loss floor that persists across training methods, objectives, and hyperparameter choices. We argue this floor is geometric in origin. Neural networks pack far more learned features into their hidden layers than they have dimensions, a phenomenon termed superposition (Elhage et al., 2022). A student with hidden width d_S can faithfully encode at most d_S × g(α) of the teacher's features, where α is the feature sparsity and g(α) = 1/((1−α) ln 1/(1−α)) is a capacity function grounded in compressed sensing theory. Features that exceed this budget are unrecoverable at the bottleneck, yielding an importance-weighted loss floor over the dropped features.

We prove this bound on the Elhage et al. toy model and verify it across 48 configurations spanning different feature counts, teacher widths, and sparsity levels. To test the bound on a real language model, we measure the feature count F and sparsity α of Pythia-410M using sparse autoencoders trained from scratch, predict the loss floor at four student widths, and compare against actual distillation experiments. Linear probing provides mechanistic confirmation: specific teacher concepts — code detection, legal text recognition, medical terminology — become linearly undecodable from narrow students, with probe accuracy falling to chance in order of decreasing feature importance, exactly as the capacity theory predicts.

Our results connect representation geometry to distillation limits and provide a practical tool: given a teacher's SAE statistics, one can estimate the distillation loss floor at any student width before committing to a training run.

**Keywords:** knowledge distillation, superposition, sparse autoencoders, mechanistic interpretability, compressed sensing, representation geometry
