# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Action-Based Results Directories ---
RESULTS_DIR = PROJECT_ROOT / "results"

TRAINING_DIR = RESULTS_DIR / "training"
MODELS_DIR = TRAINING_DIR / "checkpoints"
WANDB_DIR = TRAINING_DIR / "wandb_logs"

EVALUATION_DIR = RESULTS_DIR / "evaluation"

# Auto-create them
for d in [MODELS_DIR, WANDB_DIR, EVALUATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)