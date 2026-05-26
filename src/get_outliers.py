import pandas as pd
import sys

def count_outliers(csv_file, column_name):
    """Return the number of outliers in `column_name` 
    (values more than 2 std devs from mean)."""
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Compute mean and standard deviation
    mean_val = df[column_name].mean()
    std_val = df[column_name].std()
    
    # Define the lower and upper threshold (2 standard deviations)
    lower_bound = mean_val - 2 * std_val
    upper_bound = mean_val + 2 * std_val
    
    # Count how many values lie outside these bounds
    num_outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)].shape[0]
    
    return num_outliers

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <csv_file> <column_name>")
        sys.exit(1)
    
    # Get command-line arguments
    csv_file = sys.argv[1]
    column_name = sys.argv[2]

    # Calculate and print the number of outliers
    outliers_count = count_outliers(csv_file, column_name)
    print(f"Number of outliers in column '{column_name}': {outliers_count}")

