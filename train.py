from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import BaselineMLP, PrunableMLP
from utils import (
    evaluate,
    plot_gate_histogram,
    plot_sparsity_vs_accuracy,
    seed_everything,
    train_one_epoch,
    write_results_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-pruning MLP on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to CIFAR-10 data")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory for saved outputs")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="512,256,128",
        help="Comma-separated hidden layer dimensions (2-3 layers recommended)",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="1e-6,1e-5,1e-4",
        help="Comma-separated lambda values to compare",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--threshold", type=float, default=1e-2, help="Gate threshold for sparsity metric")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--run-baseline", action="store_true", help="Train a non-prunable baseline model")
    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    archive_path = Path(args.data_dir) / "cifar-10-python.tar.gz"
    # If a previous interrupted download exists, remove it to force a clean download.
    if archive_path.exists() and archive_path.stat().st_size < 170_000_000:
        archive_path.unlink()

    pin_memory = torch.cuda.is_available()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_full = datasets.CIFAR10(root=args.data_dir, train=True, transform=transform, download=True)
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform, download=True)

    val_size = int(len(train_full) * args.val_split)
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def train_prunable_for_lambda(
    lambda_sparse: float,
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    hidden_dims: List[int],
    output_dir: Path,
) -> tuple[float, float]:
    model = PrunableMLP(input_dim=32 * 32 * 3, hidden_dims=hidden_dims, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_loss_fn = nn.CrossEntropyLoss()

    print(f"\n=== Training prunable model with lambda={lambda_sparse:.1e} ===")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, ce_loss_fn, device, lambda_sparse)
        val_metrics = evaluate(model, val_loader, ce_loss_fn, device, lambda_sparse)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"(CE {train_metrics['ce_loss']:.4f}, Sparse {train_metrics['sparse_loss']:.2f}) | "
            f"Train Acc: {train_metrics['acc']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.2f}%"
        )

    test_metrics = evaluate(model, test_loader, ce_loss_fn, device, lambda_sparse)
    sparsity = model.sparsity_level(threshold=args.threshold)

    print(f"Test Accuracy: {test_metrics['acc']:.2f}% | Sparsity (< {args.threshold}): {sparsity:.2f}%")

    model_path = output_dir / f"prunable_lambda_{lambda_sparse:.1e}.pt"
    torch.save(model.state_dict(), model_path)

    hist_path = output_dir / f"gate_hist_lambda_{lambda_sparse:.1e}.png"
    plot_gate_histogram(model.all_gates().cpu(), hist_path)

    return test_metrics["acc"], sparsity


def train_baseline(
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    hidden_dims: List[int],
    output_dir: Path,
) -> float:
    model = BaselineMLP(input_dim=32 * 32 * 3, hidden_dims=hidden_dims, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_loss_fn = nn.CrossEntropyLoss()

    print("\n=== Training baseline (non-pruned) model ===")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, ce_loss_fn, device, lambda_sparse=0.0)
        val_metrics = evaluate(model, val_loader, ce_loss_fn, device, lambda_sparse=0.0)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['acc']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.2f}%"
        )

    test_metrics = evaluate(model, test_loader, ce_loss_fn, device, lambda_sparse=0.0)
    print(f"Baseline Test Accuracy: {test_metrics['acc']:.2f}%")

    model_path = output_dir / "baseline_mlp.pt"
    torch.save(model.state_dict(), model_path)

    return test_metrics["acc"]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    hidden_dims = [int(v.strip()) for v in args.hidden_dims.split(",") if v.strip()]
    if len(hidden_dims) < 2 or len(hidden_dims) > 3:
        raise ValueError("Please provide 2 or 3 hidden layer sizes via --hidden-dims")

    lambdas = [float(v.strip()) for v in args.lambdas.split(",") if v.strip()]
    if len(lambdas) != 3:
        raise ValueError("Please provide exactly 3 lambda values via --lambdas (low, medium, high)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(args)

    baseline_acc = None
    if args.run_baseline:
        baseline_acc = train_baseline(args, device, train_loader, val_loader, test_loader, hidden_dims, output_dir)

    rows = []
    prunable_accuracies = []
    prunable_sparsities = []

    for lmb in lambdas:
        acc, sparsity = train_prunable_for_lambda(
            lambda_sparse=lmb,
            args=args,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            hidden_dims=hidden_dims,
            output_dir=output_dir,
        )
        rows.append((f"{lmb:.1e}", acc, sparsity))
        prunable_accuracies.append(acc)
        prunable_sparsities.append(sparsity)

    plot_sparsity_vs_accuracy(
        lambdas=lambdas,
        accuracies=prunable_accuracies,
        sparsities=prunable_sparsities,
        out_path=output_dir / "sparsity_vs_accuracy.png",
        baseline_acc=baseline_acc,
    )

    write_results_markdown(output_dir / "results_report.md", rows, baseline_acc)

    print("\nSaved artifacts:")
    print(f"- Models and plots in: {output_dir.resolve()}")
    print(f"- Report: {(output_dir / 'results_report.md').resolve()}")


if __name__ == "__main__":
    main()
