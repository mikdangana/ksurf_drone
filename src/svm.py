import os, sys, pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


PRED_WINDOW = 4


# Function to predict the next value using SVM regression
def predict_next_svm(series, kernel="rbf", C=0.1, gamma="scale", epsilon=0.1):
    """ Fits an SVR model and predicts the next value in the series. """
    try:
        series = series.dropna().values.reshape(-1, 1)  # Reshape for SVR
        
        if len(series) < 3:  # Need at least 3 points for training
            return np.nan
        
        # Generate feature (X) and target (y)
        X = np.arange(len(series)).reshape(-1, 1)  # Time indices
        y = series.flatten()  # Actual values
        
        # Fit SVR model
        model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        model.fit(X, y)
        
        # Predict next value (extrapolation)
        next_value = model.predict([[len(series)]])[0]
        return next_value
    
    except Exception as e:
        print(f"Error: {e}")
        return np.nan


def coerce(df, col_name):
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    df = df.dropna(subset=[col_name])
    return df


def test_svm():
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
    df = coerce(df, column_name)
    
    # Shift the column to get actual next values for comparison
    df["Actual_Next"] = df[column_name].shift(-1)  # Next row's value as the actual next
    for i in range(PRED_WINDOW):
        df[f"prev{i}"] = df[column_name].shift(i+1, fill_value=0)  # previous 2 row's value 

    # Predict next value for each row using SVM
    #df["Predicted_Next"] = df.apply(lambda row: predict_next_svm(pd.Series(row[column_name])), axis=1)
    df["Predicted_Next"] = df.apply(lambda row: predict_next_svm(pd.Series([row[column_name]] + [row[f"prev{i}"] for i in range(PRED_WINDOW)])), axis=1)

    # Compute prediction error
    df["Prediction_Error"] = df["Actual_Next"] - df["Predicted_Next"]

    # Drop rows where predictions or actual values are NaN
    valid_rows = df.dropna(subset=["Predicted_Next", "Actual_Next"])

    # Compute RMSE (as mean error)
    if not valid_rows.empty:
        rmse = np.sqrt(mean_squared_error(valid_rows["Actual_Next"], valid_rows["Predicted_Next"]))
    
        # Compute error variance around RMSE
        error_variance_around_rmse = np.var(valid_rows["Prediction_Error"] - rmse, ddof=1)
    else:
        rmse = np.nan
        error_variance_around_rmse = np.nan

    # Save output to a new CSV
    output_file = "predicted_output_svm.csv"
    df.to_csv(output_file, index=False)

    print(f"Predictions saved to {output_file}")
    print(f"RMSE (Mean Error): {rmse:.4f}")
    print(f"Error Variance around RMSE: {error_variance_around_rmse:.4f}")



if __name__ == "__main__":
    test_svm() 
