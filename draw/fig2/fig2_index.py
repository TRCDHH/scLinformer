# %% [1] Import Libraries and Set Global Parameters
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------- CUSTOM PARAMETERS (ALL AT THE TOP) ----------------------
OUTPUT_DIR = "/home/draw/output"
FONT_NAME = 'Arial'
FONT_SIZE = 12
METHODS = ['ours', 'scgpt', 'scvi', 'Harmony', 'Scanorama']
COLORS = [
    '#3774b9',
    '#f8dce3',
    '#d0bed1',
    '#92ce91',
    '#e1eee2',
]
METRICS = ['ARI', 'AMI', 'NMI', 'HOM', 'Cell_ASW', 'Batch_ASW', 'Graph_Connectivity']
BAR_WIDTH = 0.15
PLOT_FIGSIZE = (9, 4.5)
Y_AXIS_LIMIT = (0.4, 1.08)
LEGEND_FIGSIZE = (6, 1.2)
DPI_VALUE = 300

# %% [2] Matplotlib Global Settings
plt.rcParams['font.sans-serif'] = [FONT_NAME]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = FONT_SIZE

# %% [3] Create Output Directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [4] Load Data
data = {
    'dataset': ['BMMC'] * 5 + ['PBMC'] * 5 + ['Immune'] * 5 + ['Lung'] * 5 + ['Covid'] * 5,
    'method': ['ours', 'scvi', 'Harmony', 'Scanorama', 'scgpt'] * 5,
    'ARI': [0.753, 0.656, 0.668, 0.592, 0.583, 0.892, 0.872, 0.862, 0.726, 0.668252,
            0.772, 0.717, 0.630, 0.751, 0.631, 0.661, 0.450, 0.549, 0.509, 0.487, 0.361, 0.268, 0.322, 0.340, np.nan],
    'AMI': [0.804, 0.762, 0.755, 0.736, 0.749, 0.859, 0.836, 0.822, 0.823, 0.78445,
            0.812, 0.740, 0.781, 0.817, 0.782, 0.767, 0.688, 0.737, 0.755, 0.741, 0.679, 0.638, 0.641, 0.678, np.nan],
    'NMI': [0.805, 0.763, 0.757, 0.737, 0.750, 0.860, 0.838, 0.823, 0.824, 0.78611,
            0.813, 0.741, 0.782, 0.786, 0.784, 0.769, 0.692, 0.738, 0.758, 0.744, 0.680, 0.639, 0.642, 0.679, np.nan],
    'HOM': [0.777, 0.763, 0.759, 0.713, 0.695, 0.829, 0.806, 0.765, 0.880, 0.825818,
            0.733, 0.652, 0.792, 0.786, 0.791, 0.716, 0.752, 0.688, 0.844, 0.820, 0.693, 0.698, 0.695, 0.720, np.nan],
    'Cell_ASW': [0.562, 0.553, 0.574, 0.558, 0.566, 0.609, 0.586, 0.611, 0.610, 0.744,
                 0.561, 0.538, 0.596, 0.590, 0.613, 0.587, 0.556, 0.602, 0.596, 0.601, 0.489, 0.489, 0.464, 0.476,
                 np.nan],
    'Batch_ASW': [0.926, 0.916, 0.871, 0.883, 0.939, 0.985, 0.991, 0.994, 0.990, 0.988,
                  0.948, 0.935, 0.891, 0.860, 0.924, 0.836, 0.956, 0.842, 0.863, 0.923, 0.872, 0.969, 0.951, 0.937,
                  np.nan],
    'Graph_Connectivity': [1.000, 1.000, 1.000, 0.995, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
                           0.976, 1.000, 0.975, 0.956, 0.973, 0.996, 1.000, 0.973, 0.910, 0.993, 1.000, 1.000, 1.000,
                           1.000, np.nan]
}
df = pd.DataFrame(data)

# %% [5] Prepare Variables
n_methods = len(METHODS)
datasets = df['dataset'].unique()

# %% [6] Plot Figures for All Datasets (Without Legend)
for dset in datasets:
    sub = df[df.dataset == dset]
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    x = np.arange(len(METRICS))

    for i, method in enumerate(METHODS):
        vals = sub[sub.method == method][METRICS].values.flatten()
        ax.bar(x + (i - n_methods / 2 + 0.5) * BAR_WIDTH,
               vals, BAR_WIDTH, color=COLORS[i], lw=0.3, edgecolor='white')

    ax.set_title(f'{dset}', fontsize=FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, rotation=30, ha='right')
    ax.set_ylim(*Y_AXIS_LIMIT)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f'fig_{dset}.png')
    plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
    plt.show()
    plt.close()

# %% [7] Plot and Save Legend Separately
fig, ax = plt.subplots(figsize=LEGEND_FIGSIZE)
ax.axis('off')
for i, method in enumerate(METHODS):
    ax.bar([0], [0], color=COLORS[i], label=method)
ax.legend(loc='center', ncol=5, frameon=False, fontsize=11)
plt.tight_layout()

legend_path = os.path.join(OUTPUT_DIR, 'legend_only.png')
plt.savefig(legend_path, dpi=DPI_VALUE, bbox_inches='tight')
plt.show()
plt.close()

# %% [8] Print Completion Message
print("All figures saved to:", OUTPUT_DIR)
print("Generated files list:")
print("  - fig_BMMC.png")
print("  - fig_PBMC.png")
print("  - fig_Immune.png")
print("  - fig_Lung.png")
print("  - fig_Covid.png")
print("  - legend_only.png")