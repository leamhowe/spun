import glob
import re
import pandas as pd
import os

# Find all the text files
directory = "/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/combloss_tuning"
# file_pattern = "best_hyperparameters_20251113_tune_scf_combloss_13bands_*.txt"
file_pattern = "best_hyperparameters_20251118_tune_scf_combloss_13bands_SCALE10000_*.txt"
file_list = glob.glob(os.path.join(directory, file_pattern))

if not file_list:
    print(f"Error: No files found matching the pattern '{file_pattern}'.")
    print("Please make sure the script is in the same directory as your .txt files.")
else:
    print(f"Found {len(file_list)} files to process:")
    for f in file_list:
        print(f"- {f}")

all_trials_data = []

# --- REGEX DEFINITIONS ---
# Values (RMSE_snow, Dice): [0.358..., 0.437...]
values_re = re.compile(r'Values \(RMSE_snow, Dice\): \[([\d\.\we+-]+), ([\d\.\we+-]+)\]')

# *** THIS IS THE CORRECTED REGEX (v5) ***
# We will now strip each line, so the regex just needs
# to match "key: value"
# ([\w_]+) -> captures the key (e.g., "loss_weighting")
# :        -> matches the colon
# \s* -> matches ZERO or MORE whitespace chars after the colon
# (.*)     -> captures the value (e.g., "none" or "0.000182...")
param_re = re.compile(r'([\w_]+):\s*(.*)')
# --- END REGEX DEFINITIONS ---

for filepath in file_list:
    # Extract weights suffix from filename
    try:
        filename = os.path.basename(filepath)
        suffix_with_ext = filename.rsplit('_', 1)[-1]
        weights_suffix = suffix_with_ext.rsplit('.', 1)[0]
    except Exception as e:
        print(f"Warning: Could not parse weights from filename {filename}. Using 'unknown'. Error: {e}")
        weights_suffix = 'unknown'

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    trial_blocks = content.split('--- Trial ')[1:]

    if not trial_blocks:
        print(f"Warning: No '--- Trial ' blocks found in {filepath}.")
        continue

    for i, block_content in enumerate(trial_blocks):
        trial_data = {}
        trial_data['weights'] = weights_suffix

        try:
            trial_num = block_content.split(' ', 1)[0]
            trial_data['trial'] = int(trial_num)
        except:
            trial_data['trial'] = 'unknown'

        values_match = values_re.search(block_content)
        if values_match:
            try:
                trial_data['RMSE_snow'] = float(values_match.group(1))
            except ValueError:
                trial_data['RMSE_snow'] = values_match.group(1)
            try:
                trial_data['Dice'] = float(values_match.group(2))
            except ValueError:
                trial_data['Dice'] = values_match.group(2)
        else:
            trial_data['RMSE_snow'] = None
            trial_data['Dice'] = None

        try:
            params_section = block_content.split('Parameters:')[1]
            
            # *** THIS IS THE CORRECTED LOGIC (v5) ***
            # 1. Split the section into lines
            param_lines = params_section.split('\n')
            
            for line in param_lines:
                # 2. Strip ALL leading/trailing whitespace (incl. \xa0) from EACH line
                line = line.strip()
                
                # 3. Skip any lines that are now empty
                if not line:
                    continue
                
                # 4. Match our new regex against the clean, stripped line
                param_match = param_re.match(line)
                
                if param_match:
                    key = param_match.group(1).strip()
                    value = param_match.group(2).strip()
                    
                    try:
                        value_float = float(value)
                        if value_float.is_integer():
                            value = int(value_float)
                        else:
                            value = value_float
                    except ValueError:
                        pass # It's a string, keep it as is
                    
                    trial_data[key] = value
                else:
                    # Debug line
                    print(f"DEBUG: Line did not match regex: {repr(line)}")
                        
        except IndexError:
            pass # No "Parameters:" section found
        except Exception as e:
            print(f"Error parsing parameters in {filepath} (Trial {trial_data.get('trial')}): {e}")

        all_trials_data.append(trial_data)

# --- Create DataFrame and Save ---
if all_trials_data:
    df = pd.DataFrame(all_trials_data)
    
    # Reorder columns
    all_cols = df.columns
    primary_cols = ['weights', 'trial', 'RMSE_snow', 'Dice']
    param_cols = sorted([c for c in all_cols if c not in primary_cols])
    final_cols = [c for c in (primary_cols + param_cols) if c in df.columns]
    
    df = df[final_cols]
    
    # output_csv = "20251113_hyperparameter_comparison.csv"
    output_csv = "20251118_scale10000_hyperparameter_comparison.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"\nSuccessfully processed {len(file_list)} files.")
    print(f"Found {len(all_trials_data)} total trials.")
    print(f"Data saved to {output_csv}.")
    
    print("\n--- CSV File Head (Top 5 rows) ---")
    print(df.head().to_string())
else:
    print("No data was processed. No CSV file created.")