# GateLens-CIFAR

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)

GateLens-CIFAR is a PyTorch project for learning sparse neural connections through trainable gates on top of standard MLP weights. It trains baseline and prunable models on CIFAR-10, compares accuracy versus sparsity across multiple regularization strengths, and includes a Streamlit frontend for interactive inference.

## Quick links

- Repository: https://github.com/SQUADRON-LEADER/GateLens-CIFAR
- Report: [REPORT.md](REPORT.md)
- Training script: [train.py](train.py)
- Streamlit app: [app.py](app.py)

## Table of contents

1. [Project at a glance](#project-at-a-glance)
2. [Architecture summary](#architecture-summary)
3. [Why this induces sparsity](#why-this-induces-sparsity)
4. [Repository structure](#repository-structure)
5. [Setup](#setup)
6. [Training](#training)
7. [Measured results snapshot](#measured-results-snapshot)
8. [Streamlit frontend](#streamlit-frontend)
9. [Deploy on Streamlit Community Cloud](#deploy-on-streamlit-community-cloud)
10. [Generated artifacts](#generated-artifacts)
11. [License and usage](#license-and-usage)

## Project at a glance

| Item | Details |
|---|---|
| Problem | CIFAR-10 image classification with parameter pruning |
| Core idea | Multiply each weight by a learned sigmoid gate |
| Model families | BaselineMLP and PrunableMLP |
| Pruning objective | CrossEntropy + lambda * sum(sigmoid(gate_scores)) |
| Dataset | CIFAR-10 |
| Frontend | Streamlit app for checkpoint-based inference |
| Main outputs | .pt checkpoints, gate histograms, sparsity-accuracy plot, markdown report |

## Architecture summary

| Component | Description |
|---|---|
| PrunableLinear | Linear layer with trainable weight, bias, and gate_scores |
| Gate function | gates = sigmoid(gate_scores) |
| Effective weights | pruned_weights = weight * gates |
| Sparsity proxy | L1-like penalty via sum of gate activations |
| Sparsity metric | percent of gates below threshold (default: 1e-2) |

## Why this induces sparsity

The sparsity term is:

SparsityLoss = sum(sigmoid(gate_scores))

This behaves like an L1-style pressure on effective gates. Each active connection contributes linear cost, so training is encouraged to deactivate weaker connections whenever accuracy allows. Increasing lambda increases pruning pressure and usually shifts the model toward higher sparsity with a potential accuracy trade-off.

## Mathematical formulation

For a single prunable layer with input x, weight matrix W, bias b, and gate-score matrix G:

1. Gate activation:
	g = sigmoid(G)
2. Effective weights:
	W_eff = W * g
3. Layer output:
	y = W_eff x + b

For a dataset D = {(x_i, t_i)} and model parameters theta:

TotalLoss(theta) = CE(theta; D) + lambda * S(theta)

Where:

- CE is standard cross-entropy classification loss.
- S(theta) = sum(sigmoid(G_l)) across all prunable layers l.

Interpretation:

- lambda controls the regularization strength.
- Larger lambda penalizes active gates more strongly.
- The optimization therefore balances predictive performance and effective sparsity.

## Repository structure

| File | Purpose |
|---|---|
| model.py | PrunableLinear, PrunableMLP, BaselineMLP |
| train.py | End-to-end training pipeline and lambda sweep |
| utils.py | Train/eval helpers, plots, and report writer |
| app.py | Streamlit frontend for interactive inference |
| REPORT.md | Case-study style summary and measured results |
| outputs/ | Trained checkpoints and generated visual artifacts |

## Setup

1. Create and activate a Python environment.
2. Install dependencies.

```bash
pip install -r requirements.txt
```

## Training

Train baseline plus three prunable models:

```bash
python train.py --epochs 10 --batch-size 128 --lambdas 1e-6,1e-5,1e-4 --run-baseline
```

Quick smoke test:

```bash
python train.py --epochs 1 --batch-size 256 --lambdas 1e-6,1e-5,1e-4 --run-baseline
```

### Training protocol (recommended)

| Stage | Goal | Suggested setting |
|---|---|---|
| Smoke run | Verify pipeline and outputs | epochs=1, batch-size=256 |
| Baseline fit | Establish non-pruning reference | --run-baseline enabled |
| Lambda sweep | Observe sparsity-accuracy trade-off | 1e-6, 1e-5, 1e-4 |
| Extended run | Improve pruning behavior | 20-50 epochs, tune lambda |

### Reproducibility checklist

1. Keep the same random seed (`--seed`).
2. Keep hidden dimensions fixed during lambda comparison.
3. Use same train/val split (`--val-split`) across runs.
4. Report both accuracy and sparsity threshold used.

## Measured results snapshot

From the current report run:

| Lambda | Test Accuracy (%) | Sparsity (% gates < 1e-2) |
|---:|---:|---:|
| 1.0e-06 | 44.63 | 0.00 |
| 1.0e-05 | 43.77 | 0.00 |
| 1.0e-04 | 43.07 | 0.00 |

| Model | Test Accuracy (%) |
|---|---:|
| Baseline MLP | 45.99 |

### How to interpret these numbers

| Observation | Practical meaning |
|---|---|
| Similar accuracy across low lambdas | Regularization is not yet strong enough to force heavy pruning |
| Sparsity at 0.00% with threshold 1e-2 | Gates may be shrinking but not below the reporting threshold |
| Baseline above prunable in short runs | Pruning objective often needs longer training to catch up |
| Trade-off emerges with larger lambda or longer epochs | Better sparsity usually costs some accuracy |

## Streamlit frontend

Run locally:

```bash
streamlit run app.py
```

Features:

1. Select any saved checkpoint in outputs.
2. Upload image or pick a CIFAR test sample.
3. View top-1 prediction and top-5 probabilities.
4. Inspect model type and gate sparsity for prunable checkpoints.

## Deploy on Streamlit Community Cloud

| Step | Action |
|---|---|
| 1 | Push this repository to GitHub |
| 2 | Open Streamlit Community Cloud and create a new app |
| 3 | Choose repository and branch main |
| 4 | Set entry file to app.py |
| 5 | Deploy; dependencies install from requirements.txt |

Note: if you need CIFAR test-sample mode in cloud, keep the data folder tracked or update app.py to download CIFAR at runtime.

## Troubleshooting

| Issue | Likely cause | Fix |
|---|---|---|
| `Import "streamlit" could not be resolved` | Package missing in active environment | `pip install -r requirements.txt` |
| Push rejected by GitHub for large files | Dataset/checkpoints tracked by git | Ensure `.gitignore` excludes `data/` and `outputs/` |
| Streamlit asks for email on first run | First-time Streamlit setup prompt | Press Enter to skip |
| No checkpoints found in app | `outputs/` empty or wrong path | Train first or set correct checkpoint folder in sidebar |
| Low sparsity after short run | Too few epochs / weak lambda | Increase epochs and test larger lambda |

## Generated artifacts

| Artifact | Description |
|---|---|
| baseline_mlp.pt | Baseline checkpoint |
| prunable_lambda_*.pt | Prunable checkpoints for each lambda |
| gate_hist_lambda_*.png | Gate value distribution per lambda |
| sparsity_vs_accuracy.png | Trade-off visualization |
| results_report.md | Auto-generated markdown summary |

## FAQ

### Why use an MLP instead of a CNN on CIFAR-10?
This repository focuses on demonstrating differentiable pruning mechanics clearly. MLPs make gate behavior easier to inspect.

### How is sparsity measured?
Sparsity is reported as percentage of gate values below a threshold (default 1e-2).

### Can I change hidden layers?
Yes. Use `--hidden-dims` with 2 or 3 comma-separated values, for example `1024,512,256`.

### Why are outputs ignored by git?
Checkpoint and dataset files are large; ignoring them keeps repository size small and push-safe.

## License and usage

This repository is intended for educational and research demonstration of differentiable pruning ideas on CIFAR-10.
