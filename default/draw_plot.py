#!/usr/bin/env python3

import matplotlib.pyplot as plt

# Task 1b: ROB Size Sweep
# X_VALUES = [16, 32, 64, 128, 256]
# Y_VALUES = [0.831878, 0.992742, 1.121239, 1.128770, 1.128770]
# X_LABEL = "ROB size (entries)"
# Y_LABEL = "IPC"
# PLOT_TITLE = "O3CPU IPC vs ROB Size (SEQ_LEN=64)"
# OUTPUT_FILE = "ipc_vs_rob.png"

# Task 1c: Pipeline Width Sweep
# X_VALUES = [2, 4, 8]
# Y_ORIGINAL = [1.128770, 1.153991, 1.146907]
# Y_OPTIMIZED = [1.388050, 1.525176, 1.643418]
# X_LABEL = "Pipeline Width"
# Y_LABEL = "IPC (Instructions Per Cycle)"
# PLOT_TITLE = "Task 1c: IPC vs Pipeline Width (Original vs Optimized)"
# OUTPUT_FILE = "ipc_vs_pipeline_width.png"

# Task 1d: Register File Sweep
X_VALUES = [64, 96, 128]
Y_IPC_ORIGINAL = [1.120506, 1.244335, 1.230802]
Y_IPC_OPTIMIZED = [1.377836, 1.616652, 1.549267]
Y_STALLS_ORIGINAL = [17338, 9583, 145]
Y_STALLS_OPTIMIZED = [2750, 27, 131]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(X_VALUES, Y_IPC_ORIGINAL, marker='o', linewidth=2, markersize=8, label='Original')
ax1.plot(X_VALUES, Y_IPC_OPTIMIZED, marker='s', linewidth=2, markersize=8, label='Optimized')
ax1.set_xlabel("Number of Registers", fontsize=12)
ax1.set_ylabel("IPC (Instructions Per Cycle)", fontsize=12)
ax1.set_title("IPC vs Register Count", fontsize=13, fontweight='bold')
ax1.set_xticks(X_VALUES)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

ax2 = axes[1]
ax2.plot(X_VALUES, Y_STALLS_ORIGINAL, marker='o', linewidth=2, markersize=8, label='Original')
ax2.plot(X_VALUES, Y_STALLS_OPTIMIZED, marker='s', linewidth=2, markersize=8, label='Optimized')
ax2.set_xlabel("Number of Registers", fontsize=12)
ax2.set_ylabel("Register Stalls (fullRegistersEvents)", fontsize=12)
ax2.set_title("Register Stalls vs Register Count", fontsize=13, fontweight='bold')
ax2.set_xticks(X_VALUES)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig("ipc_and_stalls_vs_registers.png", dpi=200, bbox_inches='tight')
plt.show()
