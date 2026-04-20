# Architecture Notes

## High-level pipeline

1. Load CIFAR-10 with normalization.
2. Flatten image tensors from 3x32x32 to 3072 features.
3. Forward pass through baseline or prunable MLP.
4. Compute CE loss and optional sparsity term.
5. Optimize with Adam.
6. Evaluate on validation and test splits.

## Layer design

### Baseline

`Linear -> ReLU -> Linear -> ReLU -> Linear`

### Prunable

`PrunableLinear -> ReLU -> PrunableLinear -> ReLU -> PrunableLinear`

Each prunable layer carries:

- weight
- bias
- gate_scores

and computes `effective_weight = weight * sigmoid(gate_scores)`.

## Why per-weight gates

Per-weight gates provide fine-grained pruning pressure and support differentiable optimization without hard thresholding during training.
