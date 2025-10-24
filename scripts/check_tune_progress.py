# check_study.py
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend # Correct import path
from optuna.trial import TrialState
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration ---
# !!! Make sure these match your tuning script !!!
study_name = '20251023_tuning_scf_combloss_final' # Use the correct name from logs
output_dir = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/combloss_tuning'
log_path = os.path.join(output_dir, f'{study_name}.log')
# --- End Configuration ---

try:
    storage = JournalStorage(JournalFileBackend(log_path))
    study = optuna.load_study(study_name=study_name, storage=storage)
except FileNotFoundError:
    logging.error(f"Error: Log file not found at {log_path}")
    exit()
except Exception as e:
    logging.error(f"Error loading study '{study_name}': {e}")
    exit()

# Get all trials
all_trials = study.trials

# Filter trials by state
completed_trials = [t for t in all_trials if t.state == TrialState.COMPLETE]
pruned_trials = [t for t in all_trials if t.state == TrialState.PRUNED]
failed_trials = [t for t in all_trials if t.state == TrialState.FAIL]
running_trials = [t for t in all_trials if t.state == TrialState.RUNNING]

# Print summary
print(f"--- Study: {study_name} ---")
print(f"Total trials attempted: {len(all_trials)}")
print(f"  Successfully completed: {len(completed_trials)}")
print(f"  Pruned (skipped/stopped early): {len(pruned_trials)}")
print(f"  Failed: {len(failed_trials)}")
print(f"  Currently running: {len(running_trials)}")


# --- UPDATED: Print best trials WITH parameters ---
if study.best_trials:
    print(f"\nFound {len(study.best_trials)} Pareto optimal trials among completed:")
    # Sort by RMSE (first objective) for easier reading
    sorted_best = sorted(study.best_trials, key=lambda t: t.values[0] if t.values else float('inf'))

    for i, trial in enumerate(sorted_best):
        print(f"\n--- Trial {trial.number} ---") # Add newline for readability
        # Ensure values exist before accessing
        rmse_val = trial.values[0] if trial.values and len(trial.values) > 0 else 'N/A'
        dice_val = trial.values[1] if trial.values and len(trial.values) > 1 else 'N/A'
        print(f"  Values (RMSE_snow, Dice): [{rmse_val:.4f}, {dice_val:.4f}]") # Format values

        # *** ADDED PARAMETER PRINTING ***
        print("  Parameters:")
        if trial.params:
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
        else:
            print("    (Parameters not available)")
        # *** END ADDED BLOCK ***

    # --- Write best trials WITH parameters to file ---
    # best_params_path = os.path.join(output_dir, f'best_hyperparameters_{study_name}.txt')
    # print(f"\nSaving {len(study.best_trials)} Pareto optimal trials to: {best_params_path}")
    # with open(best_params_path, 'w') as f:
    #     f.write(f"Found {len(sorted_best)} Pareto optimal trials:\n\n")
    #     for i, trial in enumerate(sorted_best):
    #         f.write(f"--- Trial {trial.number} ---\n")
    #         rmse_val = trial.values[0] if trial.values and len(trial.values) > 0 else 'N/A'
    #         dice_val = trial.values[1] if trial.values and len(trial.values) > 1 else 'N/A'
    #         f.write(f"  Values (RMSE_snow, Dice): [{rmse_val}, {dice_val}]\n")
    #         f.write("  Parameters:\n")
    #         if trial.params:
    #             for key, value in trial.params.items():
    #                 f.write(f"    {key}: {value}\n")
    #         else:
    #              f.write("    (Parameters not available)\n")
    #         f.write("\n")
    # print("Save complete.")

else:
    print("\nNo successful trials found yet.")
# --- END UPDATED BLOCK ---