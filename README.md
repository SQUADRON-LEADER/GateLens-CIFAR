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

## Generated artifacts

| Artifact | Description |
|---|---|
| baseline_mlp.pt | Baseline checkpoint |
| prunable_lambda_*.pt | Prunable checkpoints for each lambda |
| gate_hist_lambda_*.png | Gate value distribution per lambda |
| sparsity_vs_accuracy.png | Trade-off visualization |
| results_report.md | Auto-generated markdown summary |

## License and usage

This repository is intended for educational and research demonstration of differentiable pruning ideas on CIFAR-10.
