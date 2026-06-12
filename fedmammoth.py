"""
parse_ccvr_matrix.py
=====================
Parses one or more CCVR-style FedCL log files and builds a lower-triangular
task-wise accuracy matrix per file, then averages across all files.

Matrix structure:
    matrix[i][j] = accuracy on task j after completing task i
    Upper triangle = 0 (task not yet seen)

Key lines used:
    [DEBUG] Accuracy after task X: Y
    [Global] ... Task accuracies: [a, b, c, ...]  (line just before DEBUG)

CONFIG: Set your log file paths below.
"""

import re
import sys
import numpy as np
import os


# =====================================================================
# CONFIGURATION — edit these
# =====================================================================
LOG_FILES = [
    r"cifar100Results/cifar100seed42_exp1/fedavg/log.txt",
    r"cifar100Results/cifar100seed43_exp1/fedavg/log.txt",
    r"cifar100Results/cifar100seed44_exp1/fedavg/log.txt",
]

# LOG_FILES = [
#     r"CCVR\experiment1\tiny\ccvr_tiny_tasks10_rounds10_alpha10.0_lr0.001_seed70.log",
#     r"CCVR\experiment1\tiny\ccvr_tiny_tasks10_rounds10_alpha10.0_lr0.001_seed123.log",
#     r"CCVR\experiment1\tiny\ccvr_tiny_tasks10_rounds10_alpha10.0_lr0.001.log",
# ]

# =====================================================================


def parse_log(log_path):
    """
    Returns:
        matrix : np.ndarray (T x T) lower-triangular accuracy matrix
    """
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    matrix_rows = {}

    for i, line in enumerate(lines):
        # Match: [DEBUG] Accuracy after task X: Y
        m = re.match(r'\[DEBUG\] Accuracy after task (\d+): ([\d\.]+)', line.strip())
        if m:
            task_id = int(m.group(1))
            # Look back for the [Global] Task accuracies line
            for j in range(i - 1, max(i - 10, 0), -1):
                m2 = re.search(r'\[Global\].*Task accuracies:\s*\[([\d\.,\s]+)\]', lines[j])
                if m2:
                    accs = [float(x.strip()) / 100.0 for x in m2.group(1).split(',')]
                    matrix_rows[task_id] = accs
                    break

    if not matrix_rows:
        print(f"ERROR: No accuracy rows found in {log_path}")
        sys.exit(1)

    n_tasks = max(matrix_rows.keys()) + 1
    matrix = np.zeros((n_tasks, n_tasks))

    for task_i, acc_row in matrix_rows.items():
        for task_j, acc in enumerate(acc_row):
            matrix[task_i][task_j] = acc

    return matrix, n_tasks


def print_matrix(mat, label="Accuracy Matrix"):
    T = mat.shape[0]
    print(f"\n{label}")
    header = f"{'':14s}" + "  ".join(f"{'Task'+str(j):>8s}" for j in range(T))
    print(header)
    print("-" * len(header))
    for i in range(T):
        row = f"After Task {i}   "
        for j in range(T):
            v = mat[i, j]
            row += f"  {v:8.4f}" if v > 0 else f"  {'----':>8s}"
        print(row)
    print()


def compute_metrics(matrix):
    T = matrix.shape[0]

    # Average accuracy = mean of last row
    avg_acc = float(np.mean(matrix[-1]))

    # Average forgetting: max(col j, rows j..T-2) - last_row[j], for j in 0..T-2
    forgetting = []
    for j in range(T - 1):
        peak = float(matrix[j:-1, j].max())
        last = float(matrix[-1, j])
        forgetting.append(peak - last)
    avg_forgetting = float(np.mean(forgetting))

    return avg_acc, avg_forgetting, forgetting


def main():
    print(f"\nFiles : {len(LOG_FILES)}")
    print("=" * 60)

    all_matrices  = []
    all_avg_accs  = []
    all_avg_forgs = []

    # ── Per-file processing ───────────────────────────────────────────────
    for log_path in LOG_FILES:
        print(f"\n>>> File: {os.path.basename(log_path)}")

        matrix, n_tasks = parse_log(log_path)

        print(f"    Tasks detected: {n_tasks}")
        print_matrix(matrix, label=f"    Matrix — {os.path.basename(log_path)}:")

        avg_acc, avg_forg, per_task_f = compute_metrics(matrix)
        print(f"    Avg Accuracy  : {avg_acc:.4f}")
        print(f"    Avg Forgetting: {avg_forg:+.4f}")
        print(f"    Per-task forgetting: " +
              ", ".join(f"T{j}={f:+.4f}" for j, f in enumerate(per_task_f)))

        all_matrices.append(matrix)
        all_avg_accs.append(avg_acc)
        all_avg_forgs.append(avg_forg)

    # ── Average across all files ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AVERAGE ACROSS ALL FILES")
    print("=" * 60)

    avg_matrix = np.mean(all_matrices, axis=0)
    print_matrix(avg_matrix, label="Average Accuracy Matrix:")

    mean_acc  = float(np.mean(all_avg_accs))
    std_acc   = float(np.std(all_avg_accs, ddof=1)) if len(all_avg_accs) > 1 else 0.0
    mean_forg = float(np.mean(all_avg_forgs))
    std_forg  = float(np.std(all_avg_forgs, ddof=1)) if len(all_avg_forgs) > 1 else 0.0

    print(f"Per-file Avg Accuracy  : {[round(a, 4) for a in all_avg_accs]}")
    print(f"Per-file Avg Forgetting: {[round(f, 4) for f in all_avg_forgs]}")
    print()
    print(f"Final Average Accuracy  : {mean_acc:.4f} (+-) {std_acc:.4f}")
    print(f"Final Average Forgetting: {mean_forg:+.4f} (+-) {std_forg:.4f}")

    # ── Save average matrix ───────────────────────────────────────────────
    out_path = "avg_matrix.npy"
    np.save(out_path, avg_matrix)
    print(f"\nAverage matrix saved to: {out_path}")


if __name__ == "__main__":
    main()