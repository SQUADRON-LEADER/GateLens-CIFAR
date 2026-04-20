from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from model import PrunableMLP


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_images(images: Tensor) -> Tensor:
    return images.view(images.size(0), -1)


@torch.no_grad()
def accuracy_from_logits(logits: Tensor, targets: Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item() * 100.0)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss_fn: nn.Module,
    device: torch.device,
    lambda_sparse: float,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_sparse = 0.0
    total_acc = 0.0
    steps = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        images = flatten_images(images)

        logits = model(images)
        ce_loss = ce_loss_fn(logits, targets)

        sparse_loss = torch.tensor(0.0, device=device)
        if isinstance(model, PrunableMLP):
            sparse_loss = model.sparsity_loss()

        loss = ce_loss + lambda_sparse * sparse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_ce += float(ce_loss.item())
        total_sparse += float(sparse_loss.item()) if isinstance(model, PrunableMLP) else 0.0
        total_acc += accuracy_from_logits(logits, targets)
        steps += 1

    return {
        "loss": total_loss / max(steps, 1),
        "ce_loss": total_ce / max(steps, 1),
        "sparse_loss": total_sparse / max(steps, 1),
        "acc": total_acc / max(steps, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    ce_loss_fn: nn.Module,
    device: torch.device,
    lambda_sparse: float,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_sparse = 0.0
    total_acc = 0.0
    steps = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        images = flatten_images(images)

        logits = model(images)
        ce_loss = ce_loss_fn(logits, targets)

        sparse_loss = torch.tensor(0.0, device=device)
        if isinstance(model, PrunableMLP):
            sparse_loss = model.sparsity_loss()

        loss = ce_loss + lambda_sparse * sparse_loss

        total_loss += float(loss.item())
        total_ce += float(ce_loss.item())
        total_sparse += float(sparse_loss.item()) if isinstance(model, PrunableMLP) else 0.0
        total_acc += accuracy_from_logits(logits, targets)
        steps += 1

    return {
        "loss": total_loss / max(steps, 1),
        "ce_loss": total_ce / max(steps, 1),
        "sparse_loss": total_sparse / max(steps, 1),
        "acc": total_acc / max(steps, 1),
    }


def plot_gate_histogram(gates: Tensor, out_path: Path, bins: int = 50) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(gates.cpu().numpy(), bins=bins, color="#006d77", edgecolor="black", alpha=0.85)
    plt.title("Gate Value Distribution (Sigmoid Scores)")
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_sparsity_vs_accuracy(
    lambdas: Iterable[float],
    accuracies: Iterable[float],
    sparsities: Iterable[float],
    out_path: Path,
    baseline_acc: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lambdas_list = list(lambdas)
    accuracies_list = list(accuracies)
    sparsities_list = list(sparsities)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    x = np.arange(len(lambdas_list))
    labels = [f"{lmb:.1e}" for lmb in lambdas_list]

    line1 = ax1.plot(x, accuracies_list, marker="o", color="#1d3557", label="Prunable accuracy")
    ax1.set_ylabel("Accuracy (%)", color="#1d3557")
    ax1.set_xlabel("Lambda")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis="y", labelcolor="#1d3557")

    ax2 = ax1.twinx()
    line2 = ax2.plot(x, sparsities_list, marker="s", color="#e63946", label="Sparsity")
    ax2.set_ylabel("Sparsity (%)", color="#e63946")
    ax2.tick_params(axis="y", labelcolor="#e63946")

    if baseline_acc is not None:
        ax1.axhline(y=baseline_acc, linestyle="--", color="#2a9d8f", label="Baseline accuracy")

    lines = line1 + line2
    labels_plot = [l.get_label() for l in lines]
    if baseline_acc is not None:
        labels_plot.append("Baseline accuracy")
    ax1.legend(lines + ([] if baseline_acc is None else [ax1.lines[-1]]), labels_plot, loc="best")

    plt.title("Sparsity-Accuracy Trade-off Across Lambda Values")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def write_results_markdown(
    out_path: Path,
    rows: List[Tuple[str, float, float]],
    baseline_acc: float | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Self-Pruning Network Report")
    lines.append("")
    lines.append("## Why L1 on gates induces sparsity")
    lines.append(
        "The sparsity term uses the sum of sigmoid(gate_scores). This is an L1-style penalty on effective gates, "
        "which pushes many gate values toward 0 because each nonzero gate incurs a linear cost in the objective. "
        "As lambda grows, optimization increasingly favors turning weak connections off to reduce total loss."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Lambda | Test Accuracy (%) | Sparsity (% gates < 1e-2) |")
    lines.append("|---:|---:|---:|")
    for lmb, acc, sparsity in rows:
        lines.append(f"| {lmb} | {acc:.2f} | {sparsity:.2f} |")
    lines.append("")
    if baseline_acc is not None:
        lines.append(f"Baseline (non-pruned) test accuracy: **{baseline_acc:.2f}%**")
        lines.append("")

    lines.append("## Observations")
    lines.append("- Lower lambda usually preserves accuracy but prunes less.")
    lines.append("- Higher lambda increases sparsity, but can reduce accuracy when too aggressive.")
    lines.append("- The best operating point is task-dependent and should balance model compactness vs performance.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
