import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simulate data
timestamps = pd.date_range(start="2023-01-01", periods=100, freq="T")
cpu_usage = np.random.uniform(20, 80, size=(100, 3))  # Three nodes
memory_usage = np.random.uniform(1, 8, size=(100, 3))  # Three nodes (GB)
throughput = np.random.uniform(100, 1000, size=(100, 2))  # Two workers
scaling_actions = np.cumsum(np.random.randint(0, 2, size=100))  # Cumulative pod counts

# Additional data for queue size and autoscaler type
nodes = 20
pods_per_node = 10
queue_size = np.random.randint(0, 500, size=(100, nodes * pods_per_node))  # Queue sizes for 200 pods
autoscaler_types = np.random.choice(["hpa", "kf", "na", "th"], size=(100,))  # Autoscaler type per timestamp

# Create figure with expanded grid for new subplot
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Plot CPU usage
for i in range(cpu_usage.shape[1]):
    axs[0, 0].plot(timestamps, cpu_usage[:, i], label=f"Node {i+1}")
axs[0, 0].set_title("CPU Usage Over Time")
axs[0, 0].set_ylabel("Usage (%)")
axs[0, 0].legend()

# Plot memory usage
for i in range(memory_usage.shape[1]):
    axs[0, 1].plot(timestamps, memory_usage[:, i], label=f"Node {i+1}")
axs[0, 1].set_title("Memory Usage Over Time")
axs[0, 1].set_ylabel("Usage (GB)")
axs[0, 1].legend()

# Plot throughput
for i in range(throughput.shape[1]):
    axs[1, 0].plot(timestamps, throughput[:, i], label=f"Worker {i+1}")
axs[1, 0].set_title("Throughput Over Time")
axs[1, 0].set_ylabel("Bytes/sec")
axs[1, 0].legend()

# Plot scaling actions
axs[1, 1].plot(timestamps, scaling_actions, label="Pod Count")
axs[1, 1].set_title("Scaling Actions Over Time")
axs[1, 1].set_ylabel("Pod Count")
axs[1, 1].legend()

# Plot queue size and autoscaler types
average_queue_size = queue_size.mean(axis=1)
autoscaler_color_map = {"hpa": "blue", "kf": "green", "na": "orange", "th": "red"}
autoscaler_colors = [autoscaler_color_map[atype] for atype in autoscaler_types]

scatter = axs[2, 0].scatter(timestamps, average_queue_size, c=autoscaler_colors, alpha=0.7)
legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
                 for color in autoscaler_color_map.values()]
legend_names = list(autoscaler_color_map.keys())

axs[2, 0].set_title("Queue Size and Autoscaler Type")
axs[2, 0].set_ylabel("Average Queue Size")
axs[2, 0].legend(legend_labels, legend_names, title="Autoscaler Type", loc="upper right")
axs[2, 0].grid()

# Remove empty subplot (bottom-right)
fig.delaxes(axs[2, 1])

# Formatting
for ax in axs.flat:
    ax.set_xlabel("Time")
    ax.grid()

plt.tight_layout()
plt.show()

