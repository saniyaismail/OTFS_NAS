# SBL-OTFS Neural Architecture Search (NAS)

This repository contains code for Neural Architecture Search (NAS) on Sparse Bayesian Learning (SBL) models for OTFS channel estimation.

## Setup

### 1. Create Virtual Environment

```bash
# Create a virtual environment
python3 -m venv otfs_env

# Activate the virtual environment
source otfs_env/bin/activate  # On Linux/Mac
# or
otfs_env\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running NAS

Run Neural Architecture Search with specified number of trials and epochs:

```bash
python sbl_otfs_updated_dataset.py -data_num 1000 -gpu_index 0 --mode nas --nas_trials 10 --nas_epochs 40
```

**Parameters:**
- `-data_num`: Number of data samples to use (default: 1000)
- `-gpu_index`: GPU index to use (use `0` for first GPU, or `""` for CPU)
- `--mode`: Set to `nas` for Neural Architecture Search
- `--nas_trials`: Number of NAS trials to run (default: 15)
- `--nas_epochs`: Number of epochs per trial (default: 40)

**Example:**
```bash
# Run NAS with 10 trials, 40 epochs each
python sbl_otfs_updated_dataset.py -data_num 1000 -gpu_index 0 --mode nas --nas_trials 10 --nas_epochs 40
```

### Running Baseline Training

For baseline training (without NAS):

```bash
python sbl_otfs_updated_dataset.py -data_num 1000 -gpu_index 0 --mode train
```

## Generating Results

### Automatic Analysis Script

After NAS completes, generate all results (CSV, table image, and plots):

```bash
# Make the script executable (first time only)
chmod +x run_analysis.sh

# Run the analysis
./run_analysis.sh
```

This script will:
1. Parse NAS trial results and generate `nas_results.csv`
2. Create a table image `nas_results_table.png` with all trial parameters
3. Generate validation loss plot `val_loss_plot.png`

### Manual Commands

You can also run the analysis steps individually:

```bash
# Parse NAS results and generate CSV
python parse_nas_results.py

# Generate table image from CSV
python generate_table_image.py

# Plot validation loss vs epochs
python plot_loss.py
```

## Output Files

After running NAS and analysis, you will have:

- `nas_results.csv`: CSV file with all trial hyperparameters and validation losses
- `nas_results_table.png`: Visual table showing all NAS trial results
- `val_loss_plot.png`: Plot of validation loss vs epochs
- `NAS_TEST/SBL_OTFS_NAS/`: Directory containing all NAS trial data
- `NAS_TEST/TESTmodel_for_review_comments.mat`: Training history (train_loss, val_loss)
- `NAS_TEST/TESTmodel_for_review_comments.weights.h5`: Best model weights

## Data Requirements

The code expects data files in `data/5db_3kiter/`:
- `save_hDL.mat`: Channel data
- `save_yDL.mat`: Received signal data
- `save_PsiDL.mat`: Dictionary matrix
- `save_sigma2DL.mat`: Noise variance

## Notes

- The NAS process can take several hours depending on the number of trials and epochs
- Monitor progress by checking the log file or NAS_TEST directory
- Best hyperparameters are automatically saved after NAS completes
- The model automatically retrains with best hyperparameters for full epochs


[Add your license here]

