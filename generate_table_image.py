import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

# Load data - use current directory CSV file
csv_path = 'nas_results.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found. Please run parse_nas_results.py first.")
    sys.exit(1)

df = pd.read_csv(csv_path)

# Round numeric columns for better display
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].round(5)

# Format scientific notation for small values
for col in ['Alpha_Real', 'Alpha_Imag', 'Pruning_Thrld', 'Learning_Rate']:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: f"{x:.2e}" if isinstance(x, (int, float)) and (x < 0.01 or x > 100) else str(x))

# Create figure - adjust size based on number of rows and columns
num_rows = len(df)
num_cols = len(df.columns)
fig_width = min(24, max(16, num_cols * 1.5))
fig_height = min(12, max(6, num_rows * 0.4 + 2))

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')
ax.axis('tight')

# Convert DataFrame to string for display
df_display = df.copy()
for col in df_display.columns:
    df_display[col] = df_display[col].astype(str)

# Create table
table = ax.table(cellText=df_display.values, colLabels=df_display.columns, loc='center', cellLoc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 2.0) # Scale width and height

# Header styling
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#40466e')
        cell.set_text_props(color='w')
    else:
        cell.set_facecolor('#f5f5f5' if row % 2 else '#ffffff')

plt.title("NAS Trial Results", fontsize=16, pad=20)
plt.tight_layout()

# Save
output_path = 'nas_results_table.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Table image saved to {output_path}")
