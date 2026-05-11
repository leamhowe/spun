import os
from pathlib import Path

# --- Dynamic Project Root ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Standardised Output Directories ---
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics_reports"
TUNING_LOGS_DIR = RESULTS_DIR / "tuning_logs"
MODELS_DIR = RESULTS_DIR / "models"
WANDB_DIR = RESULTS_DIR / "wandb_logs"

# Automatically create directories if they don't exist
for d in [FIGURES_DIR, METRICS_DIR, TUNING_LOGS_DIR, MODELS_DIR, WANDB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Server Data Paths ---
# Centralising these here removes ugly hardcoded paths from your training scripts
SERVER_DATA_PATHS = {
    'jasmin': '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/s2_training_scf',
    'pentland': '/home/s2112771/r/1_DETECT/data/s2_training_scf'
}

def get_data_root(server_name: str) -> Path:
    """Retrieve the data root directory based on the server name."""
    server_name = server_name.lower()
    if server_name in SERVER_DATA_PATHS:
        return Path(SERVER_DATA_PATHS[server_name])
    else:
        # Fallback to a local 'data' folder if server isn't recognized
        local_data = PROJECT_ROOT / "data"
        if not local_data.exists():
            print(f"Warning: Server '{server_name}' not recognized. Please provide --data_root manually.")
        return local_data