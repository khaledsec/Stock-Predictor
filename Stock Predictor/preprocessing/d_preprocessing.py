import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_clean_data(file_path):
    """
    Load and clean the dataset.
    - Removes duplicate headers
    - Converts columns to float
    - Sorts by date
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f" Dataset not found at {file_path}")

    df = pd.read_csv(file_path)

    # Remove duplicate header rows if any (common issue in merged CSVs)
    df = df[df['Date'] != 'Date']

    # Ensure date format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # Ensure numeric types
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing Close values
    df = df.dropna(subset=['Close'])

    # Sort by date just in case
    df = df.sort_values('Date').reset_index(drop=True)

    return df


def prepare_lstm_data(df, feature_col='Close', time_step=60):
    """
    Prepare scaled LSTM input and output sequences.
    Returns: scaler, x_train, y_train
    """
    if feature_col not in df.columns:
        raise ValueError(f"Column '{feature_col}' not found in dataframe.")

    # Scale selected feature (e.g., 'Close')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[feature_col]].values)

    x_train, y_train = [], []
    for i in range(time_step, len(scaled_data)):
        x_train.append(scaled_data[i-time_step:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape for LSTM input [samples, timesteps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return scaler, x_train, y_train
