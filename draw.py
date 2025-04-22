import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("./benchmark_log.csv")

# Normalize method names for consistency
df['Type'] = df['Type'].replace({
    'DenseGPU': 'DenseGPU',
    'DenseCPU': 'DenseBLAS',
    'SparseGPU': 'SparseGPU',
    'SparseCPU': 'SparseBLAS'
})

# Set up plot style
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

def plot_metric_vs_param(metric, param, ylabel, title):
    plt.figure(figsize=(10, 6))
    for method in df['Type'].unique():
        sub_df = df[df['Type'] == method]
        means = sub_df.groupby(param)[metric].mean()
        plt.plot(means.index, means.values,
                 label=method,
                 linestyle=line_styles[method],
                 color=colors[method],
                 marker='o')
    plt.xlabel(param)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"images/{title}.png")

# 1. Latency vs Rows
plot_metric_vs_param("Avg Inference (50 times) Time (ms)", "Rows", "Time (ms)", "Latency vs Rows")

# 2. GFLOPS vs Rows
plot_metric_vs_param("GFLOPS", "Rows", "GFLOPS", "GFLOPS vs Rows")

# 3. Latency vs Cols
plot_metric_vs_param("Avg Inference (50 times) Time (ms)", "Cols", "Time (ms)", "Latency vs Cols")

# 4. GFLOPS vs Cols
plot_metric_vs_param("GFLOPS", "Cols", "GFLOPS", "GFLOPS vs Cols")

# 5. Latency vs Sparsity
plot_metric_vs_param("Avg Inference (50 times) Time (ms)", "Sparsity", "Time (ms)", "Latency vs Sparsity")

# 6. GFLOPS vs Sparsity
plot_metric_vs_param("GFLOPS", "Sparsity", "GFLOPS", "GFLOPS vs Sparsity")