import pandas as pd
from datetime import datetime

def load_and_convert_csv(file_path, output_file):
    # Load CSV into a DataFrame
    df = pd.read_csv(file_path)

    # Replace empty values (hyphens) with zeros
    df.replace('-', 0, inplace=True)

    # Ensure the first column is treated as a string
    first_column = df.columns[0]

    # Convert the first column to numeric timestamps
    df[first_column] = pd.to_datetime(df[first_column], errors='coerce').astype(int) / 10**9

    # Move the specified column to the first position
    column_to_move = 'ksurf-worker2-27aec434 CPU (m)'
    if column_to_move in df.columns:
        df = df[[column_to_move] + [col for col in df.columns if col != column_to_move]]

    # Print the DataFrame to verify
    print(df.head())

    # Save the DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return df


if __name__ == "__main__":
    file_path = "/root/kfpca/data/kf_tune.csv"  # Replace with your actual file path
    output_file = f"/root/kfpca/data/kf_tune_converted.csv"  # Replace with your desired output file path
    load_and_convert_csv(file_path, output_file)

