# %% [1] Import Libraries and Define All Custom Parameters (Top)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------- CUSTOM PARAMETERS ----------------------
OUTPUT_DIR = "/home/draw/output"
FONT_SANS_SERIF = 'Arial'
FONT_FAMILY = 'DejaVu Sans'
FONT_SIZE_GLOBAL = 12
FONT_SIZE_AXIS = 12
FONT_SIZE_TICKS = 10
FONT_SIZE_TITLE = 13

DATASETS = ['BMMC(44321)','PBMC(21445)','Immune(20783)','Lung(7674)']
COLOR_OURS = '#4287D0'
COLOR_SCGPT = '#D993D9'
BAR_WIDTH = 0.35
BAR_GAP = 0
LINE_WIDTH = 3
MARKER_SIZE = 8
PLOT_FIGSIZE = (9, 4.5)
DPI_VALUE = 300
ALPHA_VALUE = 0.8

# %% [2] Matplotlib Global Settings
plt.rcParams['font.sans-serif'] = [FONT_SANS_SERIF]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = FONT_SIZE_GLOBAL
plt.rcParams['font.family'] = FONT_FAMILY

# %% [3] Create Output Directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [4] Data Definition
data = {
    'dataset': ['BMMC(44321)', 'BMMC(44321)',
                'PBMC(21445)', 'PBMC(21445)',
                'Immune(20783)', 'Immune(20783)',
                'Lung(7674)', 'Lung(7674)'],
    'method': ['ours', 'scgpt']*4,
    'memory': [5703,17309, 6145,17355, 5733,17307, 5639,17309],
    'speed': [1600.04,197.86, 4204.90,404.62, 1517.01,190.67, 582.25,72.30]
}
df = pd.DataFrame(data)

# %% [5] Prepare Plot Variables
x = np.arange(len(DATASETS))

# %% [6] Plot Memory and Speed Comparison
fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE)

# Left Y-axis: Memory (Bar)
mem_ours = df[df.method=='ours']['memory'].values
mem_scgpt = df[df.method=='scgpt']['memory'].values

ax1.bar(x - BAR_WIDTH/2 - BAR_GAP/2, mem_ours, BAR_WIDTH, label='ours Memory', color=COLOR_OURS, alpha=ALPHA_VALUE)
ax1.bar(x + BAR_WIDTH/2 + BAR_GAP/2, mem_scgpt, BAR_WIDTH, label='scgpt Memory', color=COLOR_SCGPT, alpha=ALPHA_VALUE)
ax1.set_ylabel('Memory Consumption (MB)', fontsize=FONT_SIZE_AXIS)
ax1.tick_params(axis='y')

# Right Y-axis: Speed (Line + Marker)
ax2 = ax1.twinx()
speed_ours = df[df.method=='ours']['speed'].values
speed_scgpt = df[df.method=='scgpt']['speed'].values

ax2.plot(x, speed_ours, 'o-', color=COLOR_OURS, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='ours Speed')
ax2.plot(x, speed_scgpt, 's-', color=COLOR_SCGPT, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='scgpt Speed')
ax2.set_ylabel('Inference Speed (sample/s)', fontsize=FONT_SIZE_AXIS)
ax2.tick_params(axis='y')

# Global Style
ax1.set_xticks(x)
ax1.set_xticklabels(DATASETS, rotation=15, fontsize=FONT_SIZE_TICKS)
ax1.set_title('Memory & Speed Comparison (batchsize=64)', fontsize=FONT_SIZE_TITLE)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()

# Save and Show
plt.savefig(os.path.join(OUTPUT_DIR, 'speed_memory.png'), dpi=DPI_VALUE, bbox_inches='tight')
plt.show()
plt.close()

# %% [7] Print Completion Message
print("Speed and memory plot saved to:", OUTPUT_DIR)