import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths (update with your actual folder path)
folder_path = "analysis"

# File mapping for node metrics
node_metrics_files = [
    "data_hpa_2_node_metrics.csv",
    "data_hpa_5_node_metrics.csv",
    "data_kf_2_node_metrics.csv",
    "data_kf_5_node_metrics.csv",
    "data_th_2_node_metrics.csv",
    "data_th_5_node_metrics.csv",
]

# Helper function to load and clean data
def load_csv(file_name):
    df = pd.read_csv(os.path.join(folder_path, file_name))
    # Convert relevant columns to numeric
    for col in df.columns[1:]:  # Skip the "Timestamp" column
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# Function to create box-and-whisker plots
def plot_box_whisker(files):
    fig, axes = plt.subplots(len(files), 1, figsize=(12, len(files) * 3))
    
    if len(files) == 1:  # Ensure axes is iterable if only one subplot
        axes = [axes]
    
    for i, file_name in enumerate(files):
        df = load_csv(file_name)
        metric_label = file_name.split("_")[1]  # Extract the identifier (e.g., "hpa_2")
        
        # Drop the "Timestamp" column
        data = df.drop(columns=["Timestamp"], errors="ignore")
        
        # Drop columns with all NaN values
        data = data.dropna(axis=1, how="all")

        # Generate labels for the remaining columns
        labels = data.columns

        # Create the box plot without circles (set `showfliers=False`)
        box = axes[i].boxplot(data.values, labels=labels, vert=False, showfliers=False, patch_artist=True)
        
        # Add mean ± variance for each metric
        for j, col in enumerate(data.columns):
            mean_val = data[col].mean()
            variance_val = data[col].var()
            axes[i].text(
                mean_val, j + 1, f"{mean_val:.2f} ± {variance_val:.2f}",
                fontsize=6,
                verticalalignment="bottom",
                horizontalalignment="center",
                color="black",
            )

        # Remove the x-axis label
        axes[i].set_xlabel("")
        # Place subplot titles on the right
        axes[i].text(
            1.05, 0.5, f"Node Metrics ({metric_label})",
            transform=axes[i].transAxes,
            fontsize=10,
            rotation=0,
            verticalalignment="center",
        )

        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

# Run the box-and-whisker plot generation
plot_box_whisker(node_metrics_files)

