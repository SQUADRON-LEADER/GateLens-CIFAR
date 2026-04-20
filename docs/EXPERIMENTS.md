# Experiment Playbook

## Baseline experiment

```bash
python train.py --epochs 10 --batch-size 128 --run-baseline --lambdas 1e-6,1e-5,1e-4
```

Goal: establish non-pruned reference accuracy.

## Lambda sweep

```bash
python train.py --epochs 10 --batch-size 128 --lambdas 1e-7,1e-6,1e-5
python train.py --epochs 10 --batch-size 128 --lambdas 1e-6,1e-5,1e-4
python train.py --epochs 10 --batch-size 128 --lambdas 1e-5,1e-4,1e-3
```

Goal: identify where sparsity starts emerging without major accuracy collapse.

## Recommended tracking table

| Run ID | Hidden dims | Lambdas | Epochs | Best val acc | Test acc | Sparsity threshold | Sparsity |
|---|---|---|---:|---:|---:|---|---:|
| run-001 | 512,256,128 | 1e-6,1e-5,1e-4 | 10 | ... | ... | 1e-2 | ... |

## Notes

- Keep only one changed variable at a time for fair comparisons.
- Save the exact command used for each run.
- Use the same seed to reduce variance across sweeps.
