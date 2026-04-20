# Contributing to GateLens-CIFAR

Thanks for your interest in contributing.

## Contribution workflow

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make focused changes with clear commit messages.
4. Run a quick training smoke test.
5. Open a pull request with a concise summary.

## Development setup

```bash
pip install -r requirements.txt
```

Optional smoke test:

```bash
python train.py --epochs 1 --batch-size 256 --lambdas 1e-6,1e-5,1e-4 --run-baseline
```

## Code guidelines

- Keep modules small and focused.
- Avoid breaking command line arguments in `train.py`.
- Keep README and report docs updated with behavior changes.
- Add comments only when logic is non-obvious.

## Pull request checklist

- [ ] Code runs locally
- [ ] No unrelated files changed
- [ ] Documentation updated
- [ ] Commit messages are meaningful
