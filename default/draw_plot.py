#!/usr/bin/env python3

import matplotlib.pyplot as plt

# Task 1b: Pipeline Width Sweep
# X_VALUES = [16, 32, 64, 128, 256]
# Y_VALUES = [0.831878, 0.992742, 1.121239, 1.128770, 1.128770]
# X_LABEL = "ROB size (entries)"
# Y_LABEL = "IPC"
# PLOT_TITLE = "O3CPU IPC vs ROB Size (SEQ_LEN=64)"
# OUTPUT_FILE = "ipc_vs_rob.png"

# Task 1c: Pipeline Width Sweep
X_VALUES = [2, 4, 8]
Y_ORIGINAL = [1.128770, 1.153991, 1.146907]
Y_OPTIMIZED = [1.388050, 1.525176, 1.643418]

X_LABEL = "Pipeline Width"
Y_LABEL = "IPC (Instructions Per Cycle)"
PLOT_TITLE = "Task 1c: IPC vs Pipeline Width (Original vs Optimized)"
OUTPUT_FILE = "ipc_vs_pipeline_width.png"

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(X_VALUES, Y_ORIGINAL, marker='o', linewidth=2, markersize=8, label='Original')
ax.plot(X_VALUES, Y_OPTIMIZED, marker='s', linewidth=2, markersize=8, label='Optimized')
ax.set_xlabel(X_LABEL, fontsize=12)
ax.set_ylabel(Y_LABEL, fontsize=12)
ax.set_title(PLOT_TITLE, fontsize=13, fontweight='bold')
ax.set_xticks(X_VALUES)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches='tight')
plt.show()
