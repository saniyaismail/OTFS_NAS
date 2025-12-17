#!/bin/bash
# Script to generate table and plot after NAS completes

cd /home/saniya/Downloads/model_run
source otfs_env/bin/activate

echo "Checking if NAS has completed..."
if [ -d "NAS_TEST/SBL_OTFS_NAS" ]; then
    TRIAL_COUNT=$(find NAS_TEST/SBL_OTFS_NAS -name "trial_*" -type d | wc -l)
    echo "Found $TRIAL_COUNT trials"
    
    if [ "$TRIAL_COUNT" -ge 15 ]; then
        echo "NAS appears to be complete. Generating results..."
        
        # Parse NAS results and generate CSV
        echo "Parsing NAS results..."
        python parse_nas_results.py
        
        # Generate table image
        echo "Generating table image..."
        python generate_table_image.py
        
        # Plot validation loss
        echo "Plotting validation loss..."
        python plot_loss.py
        
        echo "Done! Check nas_results_table.png and val_loss_plot.png"
    else
        echo "NAS still running... ($TRIAL_COUNT/15 trials completed)"
        echo "Run this script again when NAS completes."
    fi
else
    echo "NAS_TEST/SBL_OTFS_NAS directory not found. NAS may not have started yet."
fi


