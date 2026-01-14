import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import pandas as pd
import altair as alt
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration ---
# study_names = [
#     '20251111_tune_scf_combloss_13bands_AllDino',
#     '20251111_tune_scf_combloss_13bands_decur',
#     '20251111_tune_scf_combloss_13bands_imagenet',
#     '20251111_tune_scf_combloss_13bands_moco',
#     '20251111_tune_scf_combloss_13bands_softcon'
# ]
# study_names = [
#     '20251113_tune_scf_combloss_13bands_NORM_dino',
#     '20251113_tune_scf_combloss_13bands_NORM_decur',
#     '20251113_tune_scf_combloss_13bands_NORM_imagenet',
#     '20251113_tune_scf_combloss_13bands_NORM_moco',
#     '20251113_tune_scf_combloss_13bands_NORM_softcon',
#     '20251113_tune_scf_combloss_13bands_NORM_noweights'
# ]
study_names = [
    '20251118_tune_scf_combloss_13bands_SCALE10000_dino',
    '20251118_tune_scf_combloss_13bands_SCALE10000_decur',
    '20251118_tune_scf_combloss_13bands_SCALE10000_imagenet',
    '20251118_tune_scf_combloss_13bands_SCALE10000_moco',
    '20251118_tune_scf_combloss_13bands_SCALE10000_softcon',
    '20251118_tune_scf_combloss_13bands_SCALE10000_noweights'
]
output_dir = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/combloss_tuning'
# --- End Configuration ---

all_trials_list = []
logging.info(f"Processing {len(study_names)} studies...")

for study_name in study_names:
    log_path = os.path.join(output_dir, f'{study_name}.log')
    logging.info(f"Loading study: {study_name} from {log_path}")
    
    try:
        storage = JournalStorage(JournalFileBackend(log_path))
        study = optuna.load_study(study_name=study_name, storage=storage)
    except FileNotFoundError:
        logging.warning(f"  Warning: Log file not found at {log_path}. Skipping.")
        continue
    except Exception as e:
        logging.warning(f"  Warning: Error loading study '{study_name}': {e}. Skipping.")
        continue

    study_df = study.trials_dataframe()
    try:
        weights_name = study_name.rsplit('_', 1)[-1]
    except:
        weights_name = study_name
    
    study_df['weights'] = weights_name
    all_trials_list.append(study_df)

if not all_trials_list:
    logging.error("No study data was loaded. Exiting.")
    exit()

logging.info("All studies loaded. Concatenating data...")
combined_df = pd.concat(all_trials_list, ignore_index=True)

# --- Clean data for plotting ---
plot_df = combined_df[combined_df['state'] == 'COMPLETE'].copy()

if plot_df.empty:
    logging.error("No COMPLETED trials found in any study. Cannot create plot.")
    exit()

plot_df = plot_df.rename(columns={
    'values_0': 'RMSE_snow',
    'values_1': 'Dice'
})

if 'RMSE_snow' not in plot_df.columns or 'Dice' not in plot_df.columns:
    logging.error("Renaming 'values_0' or 'values_1' failed. Stopping.")
    exit()

# --- *** START OF FIX *** ---
# The 'duration' column is a 'timedelta' object, which Altair can't plot.
# We convert it to total seconds (a number) so Altair can handle it.
if 'duration' in plot_df.columns:
    logging.info("Converting 'duration' column to total seconds for Altair...")
    plot_df['duration_seconds'] = plot_df['duration'].dt.total_seconds()
    # We can now drop the original 'duration' column
    plot_df = plot_df.drop(columns=['duration'])
# --- *** END OF FIX *** ---

# Fill NaNs for cleaner tooltips (now that 'duration' is gone)
plot_df = plot_df.fillna('N/A')

logging.info(f"Data prepared. Plotting {len(plot_df)} total completed trials...")

# --- Create the plot ---
param_columns = [col for col in plot_df.columns if col.startswith('params_')]
# Add our new 'duration_seconds' column to the tooltip list
tooltip_cols = ['weights', 'number', 'RMSE_snow', 'Dice', 'state', 'duration_seconds'] + param_columns

chart = alt.Chart(plot_df).mark_circle(size=80, opacity=0.5).encode(
    x=alt.X('RMSE_snow', 
            scale=alt.Scale(zero=False), 
            title='RMSE Snow (Lower is better)'),
    
    y=alt.Y('Dice', 
            scale=alt.Scale(zero=False), 
            title='Dice Score (Higher is better)'),
    
    color=alt.Color('weights', title="Weights Group"),
    
    # Use the updated tooltip list
    tooltip=tooltip_cols
    
).properties(
    title='All Completed Trials: Dice vs. RMSE by Weights',
    width=800,
    height=600
).interactive()

output_chart_html = "20251118_scale10000_all_trials_tradeoff_plot.html"
chart.save(output_chart_html)

logging.info(f"Successfully saved plot to {output_chart_html}")
logging.info("Use 'scp' to copy this file to your local machine to view it.")