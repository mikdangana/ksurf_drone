import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example data (replace with your actual data)
data = {
    "algorithm": ["hpa", "kf", "na", "th"],
    "1%": [3000, 2500, 2700, 2600],  # Metric values at 1% threshold
    "2%": [3100, 2600, 2800, 2700],  # Metric values at 2% threshold
    "5%": [3200, 2700, 2900, 2800],  # Metric values at 5% threshold
}

# Create DataFrame
df = pd.DataFrame(data)

# Rank algorithms (lower is better)
for threshold in ["1%", "2%", "5%"]:
    df[f"{threshold}_rank"] = df[threshold].rank(method="min")

# Melt DataFrame for visualization
rank_df = df.melt(id_vars=["algorithm"], value_vars=["1%_rank", "2%_rank", "5%_rank"],
                  var_name="threshold", value_name="rank")

# Plot rankings
plt.figure(figsize=(8, 6))
sns.lineplot(data=rank_df, x="threshold", y="rank", hue="algorithm", marker="o")
plt.title("Algorithm Rankings Across Thresholds")
plt.ylabel("Rank (Lower is Better)")
plt.xlabel("Threshold")
plt.gca().invert_yaxis()  # Invert y-axis so rank 1 is at the top
plt.legend(title="Algorithm")
plt.grid()
plt.tight_layout()
plt.savefig("algorithm_rankings.png")
plt.show()

