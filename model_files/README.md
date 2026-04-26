# Model Files — Download from Kaggle

Place the following files in this folder:

## Required Files

| File | Source Notebook | How to Get |
|------|----------------|------------|
| `ensemble_best.safetensors` | final-2 (NB2) output | Kaggle → final-2 → Output tab → Download |
| `model_config.yaml` | final-1 (NB1) output | Kaggle → final-1 → Output tab → Download |

## Optional Files (for dashboard)

| File | Source Notebook |
|------|----------------|
| `web_app_bundle.json` | final 3 (NB3) output |
| `model_summary.json` | final-2 (NB2) output |
| `training_history.json` | final-2 (NB2) output |

## Using Kaggle CLI

```bash
pip install kaggle
kaggle kernels output assualttger/final-2 -p ./
kaggle kernels output assualttger/final-1 -p ./
```

## File Sizes
- `ensemble_best.safetensors`: ~700 MB
- All other files: < 1 MB each
