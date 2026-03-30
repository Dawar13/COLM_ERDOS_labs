# Experiment 4: Linear Probing (Mechanistic Validation)

## Status: Planned

## Purpose
Confirm that the loss floor is caused by specific features being ABSENT from the student's representation — not by poor training. Show that features disappear in importance order, as the theory predicts.

## Concept Datasets (5-8 binary classification tasks)

| Concept | Positive Source | Negative Source | Expected Importance |
|---------|----------------|-----------------|-------------------|
| Is question? | QA datasets | Statements | HIGH |
| Is French? | CC-100 French | English text | HIGH-MEDIUM |
| Contains code? | The Stack / GitHub | Prose | MEDIUM |
| About sports? | Sports articles | General web | MEDIUM-LOW |
| Legal text? | FreeLaw, EDGAR | General web | LOW |
| Medical text? | PubMed, MIMIC | General web | LOW |
| Base64 encoded? | Generated base64 | Normal text | VERY LOW |

## Methodology
1. Collect ~10,000 labeled examples per concept (5K positive, 5K negative)
2. Extract hidden states from teacher and all students at the target layer
3. Mean-pool across sequence length → one vector per example
4. Train logistic regression probe (80/20 split)
5. Report accuracy per (model, concept) pair

## Expected Pattern
- Common concepts survive even in narrow students
- Rare concepts drop to chance (~50%) in narrow students
- Features die in importance order (rarest first)
- Sharp transition, not gradual fade

## Key Threshold
50% accuracy = random chance for binary classification = concept is ABSENT from representation
