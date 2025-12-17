import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import os

# Load data from the saved .mat file
mat_file = './NAS_TEST/TESTmodel_for_review_comments.mat'
if not os.path.exists(mat_file):
    print(f"Error: {mat_file} not found. Please run NAS and retrain first.")
    exit(1)

data = io.loadmat(mat_file)
train_loss = np.squeeze(data['train_loss'])
val_loss = np.squeeze(data['val_loss'])

# Handle different array shapes
if train_loss.ndim == 0:
    train_loss = np.array([train_loss])
if val_loss.ndim == 0:
    val_loss = np.array([val_loss])

# Ensure they are 1D arrays
train_loss = train_loss.flatten()
val_loss = val_loss.flatten()

epochs = np.arange(1, len(train_loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b-o', label='Training Loss', markersize=4)
plt.plot(epochs, val_loss, 'r-s', label='Validation Loss', markersize=4)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Validation Loss vs Epochs', fontsize=14, fontweight='bold')
# plt.yscale('log')
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()
plt.savefig('val_loss_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved to val_loss_plot.png")
