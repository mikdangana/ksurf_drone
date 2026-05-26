import os, sys, pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error  # For accuracy calculation


PRED_WINDOW = 4


# Function to predict the next value using ARIMA
def predict_next_arima(series, order=(1,1,1)):
    """ Fits an ARIMA model and predicts the next value in the series. """
    try:
        p = int(sys.argv[sys.argv.index('-p')+1]) if '-p' in sys.argv else 1
        d = int(sys.argv[sys.argv.index('-d')+1]) if '-d' in sys.argv else 1
        r = int(sys.argv[sys.argv.index('-r')+1]) if '-r' in sys.argv else 1
        order = (p, d, r)
        series = series.dropna()  # Remove NaN values
        if len(series) < 3:  # Need at least 3 points for ARIMA
            return np.nan
        
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast.iloc[0]  # Return the predicted value
    except Exception as e:
        print(f"Error: {e}")
        return np.nan


def coerce(df, col_name):
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    #print(f"coerce.df = {df[col_name]}")
    df = df.dropna(subset=[col_name])
    return df



def test_arima():
    f = os.path.join(sys.path[0],'..','data','twitter_trace.csv')
    f = sys.argv[sys.argv.index("-f")+1] if "-f" in sys.argv else f
    xcol = sys.argv[sys.argv.index("-x")+1] if "-x" in sys.argv else 'Tweets 09-May-2023'
    ycol = sys.argv[sys.argv.index("-y")+1] if "-y" in sys.argv else 'Tweets 09-May-2023'
 
    # Load CSV file
    csv_file = f  # Change this to your actual file
    column_name = xcol  # Change this to your actual column name

    # Read CSV
    df = pd.read_csv(csv_file)

    # Convert the column to numeric (handling strings)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df = coerce(df, column_name)
    
    # Shift the column to get actual next values for comparison
    df["Actual_Next"] = df[column_name].shift(-1)  # Next row's value as the actual next
    for i in range(PRED_WINDOW):
        df[f"prev{i}"] = df[column_name].shift(i+1, fill_value=0)  # previous 2 row's value 

    #print(f"prev = {df['prev0']}, prev1 = {df['prev1']}, col = {df[column_name]}")
    # Predict next value for each row
    df["Predicted_Next"] = df.apply(lambda row: predict_next_arima(pd.Series([row[column_name]] + [row[f"prev{i}"] for i in range(PRED_WINDOW)])), axis=1)

    # Calculate prediction accuracy (if actual next value is available)
    df["Accuracy"] = df.apply(lambda row: 100 - (abs(row["Predicted_Next"] - row["Actual_Next"]) / row["Actual_Next"] * 100) 
                          if not pd.isna(row["Predicted_Next"]) and not pd.isna(row["Actual_Next"]) and row["Actual_Next"] != 0
                          else np.nan, axis=1)

    # Compute prediction error
    df["Prediction_Error"] = df["Actual_Next"] - df["Predicted_Next"]

    # Compute cumulative accuracy (MAPE-based)
    valid_rows = df.dropna(subset=["Predicted_Next", "Actual_Next"])

    rmse = np.sqrt(mean_squared_error(valid_rows["Actual_Next"], valid_rows["Predicted_Next"]))
    if not valid_rows.empty:
        cumulative_accuracy = 100 - mean_absolute_percentage_error(valid_rows["Actual_Next"], valid_rows["Predicted_Next"]) * 100
        # Compute error variance around RMSE
        error_variance_around_rmse = np.var(valid_rows["Prediction_Error"] - rmse, ddof=1)
    else:
        cumulative_accuracy = np.nan
        error_variance_around_rmse = np.nan


    # Save output to a new CSV
    output_file = "predicted_output.csv"
    df.to_csv(output_file, index=False)

    print(f"Predictions saved to {output_file}")
    print(f"Cumulative Prediction Accuracy: {cumulative_accuracy:.2f}%")
    print(f"RMSE (Mean Error): {rmse:.4f}")
    print(f"Error Variance around RMSE: {error_variance_around_rmse:.4f}")



if __name__ == "__main__":
    test_arima() 
