# Case Study Report: GateLens CIFAR

## Objective

Build a PyTorch feedforward classifier that learns to prune its own weights during training using learnable gates.

## Method Summary

Each linear layer uses trainable gate scores with the same shape as the weight matrix:

- `gates = sigmoid(gate_scores)`
- `pruned_weights = weight * gates`
- output computed using `pruned_weights`

This keeps the operation differentiable, so gradients flow to both `weight` and `gate_scores`.

## Why L1-like regularization induces sparsity

The objective is:

`TotalLoss = CrossEntropyLoss + lambda * SparsityLoss`

with:

`SparsityLoss = sum(sigmoid(gate_scores))`

This is an L1-style penalty on effective gate magnitudes. Since each active gate contributes a linear cost, optimization prefers smaller gate values whenever they do not strongly hurt task performance. Larger `lambda` strengthens this pressure and yields more sparse connectivity.

## Result Table (Measured from completed run)

| Lambda | Test Accuracy (%) | Sparsity (% gates < 1e-2) |
|---:|---:|---:|
| 1.0e-06 (low) | 44.63 | 0.00 |
| 1.0e-05 (medium) | 43.77 | 0.00 |
| 1.0e-04 (high) | 43.07 | 0.00 |

Optional baseline row:

| Model | Test Accuracy (%) | Sparsity |
|---:|---:|---:|
| Baseline MLP | 45.99 | N/A |

## Expected Trade-off Observations

- Low `lambda`: best prunable accuracy in this run.
- Medium and high `lambda`: lower accuracy than low lambda, consistent with stronger regularization.
- Sparsity stayed at 0.00% for threshold `1e-2` after 1 epoch, indicating more training and/or larger lambda is needed for visible pruning at this threshold.

## Reproducibility and Artifacts

Run training to generate:

- model checkpoints (`*.pt`)
- gate histograms (`gate_hist_lambda_*.png`)
- sparsity-vs-accuracy plot (`sparsity_vs_accuracy.png`)
- auto-generated result report (`outputs/results_report.md`)

## Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Train three lambdas plus baseline:

```bash
python train.py --epochs 10 --batch-size 128 --lambdas 1e-6,1e-5,1e-4 --run-baseline
```

Quick smoke test:

```bash
python train.py --epochs 1 --batch-size 256 --lambdas 1e-6,1e-5,1e-4 --run-baseline
```
