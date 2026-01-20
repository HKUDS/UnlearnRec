# src9_release

This repository contains **raw research code** for GNN pretraining / fine-tuning for unlearning experiments.  
The codebase is currently **not well-organized and may be hard to read**. A cleaner refactor and documentation update will come later.

> Note: I am currently focusing on training a multimodal large model, so maintenance for this repo is temporarily limited. Updates will resume afterward.

## What to read first

- **Start from `examples/`**: there is a small, runnable example showing how to use the code in this folder.
  - Follow that example as the recommended entry point.

## Parameter notes (important)

Different GNN models may require **different hyperparameters** during:
- **pretraining-for-unlearning**, and
- **fine-tuning**.

Please pay special attention to the following parameters and typical values:

| Argument | Suggested values / range |
|---|---|
| `--pretrain_drop_rate` | `0.03`, `0.05`, `0.1`, `0.15`, `0.2`, `0.3` |
| `--batch` | `256`, `512`, `1024`, `2048`, `4096` |
| `--reg` | `1e-8`, `1e-7`, `1e-6` |
| `--unlearn_wei` | `0.1`, `0.2`, `0.5`, `1.0` |
| `--align_wei` | `0.001`, `0.002`, `0.003`, `0.004`, `0.005`, `0.01`, `0.02`, `0.025`, `0.05` |
| `--unlearn_ssl` | `0.001`, `0.0001` |

You can copy parameter configurations from the small example under `examples/` and then adjust per model/dataset.

## Logs / training records

The `logs/` directory contains extensive training records.  
If you need more context on training behavior, hyperparameter choices, or debugging, it is a good place to reference.

## Repo layout (high level)

- `examples/`: minimal example(s) to learn how to use this code
- `datasets/`: dataset files (if provided)
- `checkpoints/`: saved checkpoints
- `logs/`: training logs / records
- `Utils/`: utilities

