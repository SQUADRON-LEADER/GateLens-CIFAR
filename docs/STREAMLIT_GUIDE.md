# Streamlit Guide

## Local run

```bash
streamlit run app.py
```

## App controls

- Checkpoint folder path
- Hidden dims configuration
- Data folder path
- Checkpoint selection
- Image source selection (upload or CIFAR sample)

## Expected behavior

1. App loads selected model checkpoint.
2. For prunable models, sparsity is shown.
3. Uploaded image is resized to 32x32 and normalized.
4. Top-1 and Top-5 predictions are displayed.

## Streamlit Cloud deployment

1. Push repository to GitHub.
2. Create app in Streamlit Community Cloud.
3. Set `app.py` as entrypoint.
4. Confirm dependencies in `requirements.txt`.

## Common issues

- No checkpoints listed: wrong `outputs` path or missing files.
- Shape mismatch on load: hidden dims differ from training run.
- Low confidence on random uploads: model is trained on CIFAR-10 style images.
