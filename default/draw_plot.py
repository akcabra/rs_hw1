#!/usr/bin/env python3
"""Plot IPC vs ROB size for Task 1b and print a short saturation analysis."""

import matplotlib.pyplot as plt


# Plot configuration
X_VALUES = [16, 32, 64, 128, 256]
Y_VALUES = [0.831878, 0.992742, 1.121239, 1.128770, 1.128770]
X_LABEL = "ROB size (entries)"
Y_LABEL = "IPC"
PLOT_TITLE = "O3CPU IPC vs ROB Size (SEQ_LEN=64)"
OUTPUT_FILE = "ipc_vs_rob.png"


def find_saturation_point(rob, ipc, epsilon=1e-6):
    """Return the first ROB size after which IPC gains are effectively zero."""
    for idx in range(1, len(ipc)):
        gain = ipc[idx] - ipc[idx - 1]
        if abs(gain) <= epsilon:
            return rob[idx - 1], rob[idx]
    return None, None


def main():
    sat_from, sat_to = find_saturation_point(X_VALUES, Y_VALUES)

    plt.figure(figsize=(8, 5))
    plt.plot(X_VALUES, Y_VALUES, marker="o", linewidth=2)
    plt.title(PLOT_TITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.xticks(X_VALUES)
    plt.grid(True, linestyle="--", alpha=0.5)

    if sat_from is not None and sat_to is not None:
        plt.axvline(sat_from, linestyle=":", linewidth=1.5, color="red", label="Saturation begins")
        plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200)
    plt.show()

    print("Task 1b answer:")
    if sat_from is not None and sat_to is not None:
        print(f"- IPC saturates at around ROB={sat_from} entries (no gain from {sat_from} -> {sat_to}).")
    else:
        print("- No clear saturation point found in the provided data.")

    print("- This suggests the workload has limited exploitable ILP beyond roughly 128 in-flight instructions,")
    print("  so increasing ROB further does not expose additional parallelism or improve IPC.")
    print(f"- Plot saved as: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
