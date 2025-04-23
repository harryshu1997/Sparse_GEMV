import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("./benchmark_log.csv")

# Normalize method names
df['Type'] = df['Type'].replace({
    'DenseGPU': 'DenseGPU',
    'DenseCPU': 'DenseBLAS',
    'SparseGPU': 'SparseGPU',
    'SparseCPU': 'SparseBLAS'
})

# Plot styles
plt.style.use("ggplot")
colors = {
    'DenseGPU': 'tab:blue',
    'DenseBLAS': 'tab:orange',
    'SparseGPU': 'tab:green',
    'SparseBLAS': 'tab:red'
}
line_styles = {
    'DenseGPU': '-',
    'DenseBLAS': '--',
    'SparseGPU': '-.',
    'SparseBLAS': ':'
}

# Ensure output directory
os.makedirs("images", exist_ok=True)

# Plot function with optional log scale
def plot_with_fixed_dense_and_selectable_sparse(metric, param, fixed_values, sparse_sparsity, ylabel, title, log_y=False):
    plt.figure(figsize=(10, 6))

    # ---- Dense methods ----
    dense_fixed_values = fixed_values.copy()
    dense_fixed_values["Sparsity"] = 0.0
    dense_df = df.copy()
    for k, v in dense_fixed_values.items():
        if k != param:
            dense_df = dense_df[dense_df[k] == v]
    
    for method in ["DenseGPU", "DenseBLAS"]:
        sub_df = dense_df[dense_df["Type"] == method]
        means = sub_df.groupby(param)[metric].mean()
        if not means.empty:
            fixed_info = ', '.join([f"{k}={v}" for k, v in dense_fixed_values.items() if k != param])
            plt.plot(means.index, means.values,
                     label=f"{method} ({fixed_info})",
                     linestyle=line_styles[method],
                     color=colors[method],
                     marker='o')

    # ---- Sparse methods ----
    sparse_fixed_values = fixed_values.copy()
    sparse_fixed_values["Sparsity"] = sparse_sparsity
    sparse_df = df.copy()
    for k, v in sparse_fixed_values.items():
        if k != param:
            sparse_df = sparse_df[sparse_df[k] == v]
    
    for method in ["SparseGPU", "SparseBLAS"]:
        sub_df = sparse_df[sparse_df["Type"] == method]
        means = sub_df.groupby(param)[metric].mean()
        if not means.empty:
            fixed_info = ', '.join([f"{k}={v}" for k, v in sparse_fixed_values.items() if k != param])
            plt.plot(means.index, means.values,
                     label=f"{method} ({fixed_info})",
                     linestyle=line_styles[method],
                     color=colors[method],
                     marker='o')

    plt.xlabel(param)
    plt.ylabel(ylabel)
    if log_y:
        plt.yscale("log")
        title += " (Log Scale)"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"images/{title}.png")
    plt.close()

# === CONFIGURATION ===
fixed_values = {
    "Rows": 5000,
    "Cols": 512
}
selected_sparse_sparsity = 0.9

# === PLOT CALLS ===
# Latency plots (only linear)
# for axis in ["Rows", "Cols", "Sparsity"]:
#     plot_with_fixed_dense_and_selectable_sparse("Avg Inference (50 times) Time (ms)", axis, fixed_values, selected_sparse_sparsity, "Time (ms)", f"Time (ms) vs {axis}")
#     plot_with_fixed_dense_and_selectable_sparse("Avg Inference (50 times) Time (ms)", axis, fixed_values, selected_sparse_sparsity, "Time (ms)", f"Time (ms) vs {axis}", log_y=True)

# plot_with_fixed_dense_and_selectable_sparse("Avg Inference (50 times) Time (ms)", "Rows", fixed_values, selected_sparse_sparsity, "Time (ms)", "Latency vs Rows")
# plot_with_fixed_dense_and_selectable_sparse("Avg Inference (50 times) Time (ms)", "Cols", fixed_values, selected_sparse_sparsity, "Time (ms)", "Latency vs Cols")
# plot_with_fixed_dense_and_selectable_sparse("Avg Inference (50 times) Time (ms)", "Sparsity", fixed_values, selected_sparse_sparsity, "Time (ms)", "Latency vs Sparsity")

# GFLOPS plots (linear + log scale)
for axis in ["Rows", "Cols", "Sparsity"]:
    plot_with_fixed_dense_and_selectable_sparse("GFLOPS", axis, fixed_values, selected_sparse_sparsity, "GFLOPS", f"GFLOPS vs {axis}")
    plot_with_fixed_dense_and_selectable_sparse("GFLOPS", axis, fixed_values, selected_sparse_sparsity, "GFLOPS", f"GFLOPS vs {axis}", log_y=True)
    plot_with_fixed_dense_and_selectable_sparse("Avg Inference (50 times) Time (ms)", axis, fixed_values, selected_sparse_sparsity, "Time (ms)", f"Time (ms) vs {axis}")
    plot_with_fixed_dense_and_selectable_sparse("Avg Inference (50 times) Time (ms)", axis, fixed_values, selected_sparse_sparsity, "Time (ms)", f"Time (ms) vs {axis}", log_y=True)