import json
import glob
import os
import numpy as np
import pandas as pd

# Path to trials - can be configured via command line or use default
import sys
if len(sys.argv) > 1:
    search_dir = sys.argv[1]
else:
    search_dir = "NAS_TEST/SBL_OTFS_NAS"

trial_files = glob.glob(os.path.join(search_dir, "trial_*", "trial.json"))

if not trial_files:
    print(f"No trial files found in {search_dir}")
    print("Please run NAS first or check the directory path.")
    sys.exit(1)

results = []

for tf_path in trial_files:
    with open(tf_path, 'r') as f:
        data = json.load(f)
    
    trial_id = data['trial_id']
    hp = data['hyperparameters']['values']
    
    # metrics might be under metrics -> metrics -> val_loss -> observations -> [0] -> value
    # structure varies by tuner version, usually metrics -> val_loss -> direction/observations
    try:
        observations = data['metrics']['metrics']['val_loss']['observations']
        # Each observation has a 'value' list, we want the min of all of them
        val_losses = [obs['value'][0] for obs in observations]
        val_loss = min(val_losses)
    except KeyError:
        val_loss = float('inf')
        
    results.append({
        'id': trial_id,
        'val_loss': val_loss,
        'hp': hp
    })

# Sort by validation loss (best first)
results.sort(key=lambda x: x['val_loss'])

# Create DataFrame for CSV export
df_data = []
for i, res in enumerate(results):
    h = res['hp']
    df_data.append({
        'Rank': i + 1,
        'Trial_ID': res['id'],
        'Val_Loss': res['val_loss'],
        'Filters_L1': h.get('filters_l1', 'N/A'),
        'Filters_L2': h.get('filters_l2', 'N/A'),
        'Filters_L3': h.get('filters_l3', 'N/A'),
        'Act_L1': h.get('act_l1', 'N/A'),
        'Act_L2': h.get('act_l2', 'N/A'),
        'Act_L3': h.get('act_l3', 'N/A'),
        'Act_L4': h.get('act_l4', 'N/A'),
        'Alpha_Real': h.get('alpha_real', 'N/A'),
        'Alpha_Imag': h.get('alpha_imag', 'N/A'),
        'Pruning_Thrld': h.get('pruning_thrld', 'N/A'),
        'Weight_Update': h.get('wt', 'N/A'),
        'Learning_Rate': h.get('learning_rate', 'N/A')
    })

df = pd.DataFrame(df_data)
csv_path = 'nas_results.csv'
df.to_csv(csv_path, index=False)
print(f"CSV file saved to {csv_path}")

# Print markdown table
print("\n| Rank | Trial ID | Val Loss | Filters (L1, L2, L3) | Activations | Alpha Real/Imag | LR |")
print("|---|---|---|---|---|---|---|")

for i, res in enumerate(results):
    h = res['hp']
    filters = f"[{h.get('filters_l1')}, {h.get('filters_l2')}, {h.get('filters_l3')}, 1]"
    acts = f"[{h.get('act_l1')}, {h.get('act_l2')}, {h.get('act_l3')}, {h.get('act_l4')}]"
    alphas = f"{h.get('alpha_real'):.1e} / {h.get('alpha_imag'):.1e}"
    lr = f"{h.get('learning_rate'):.1e}"
    
    print(f"| {i+1} | {res['id']} | {res['val_loss']:.5f} | {filters} | {acts} | {alphas} | {lr} |")
