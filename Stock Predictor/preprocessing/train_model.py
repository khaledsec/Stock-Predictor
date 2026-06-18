import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import numpy as np
# Assuming the preprocessing file is in a folder named 'preprocessing'
# and this script is run from the root directory.
from preprocessing.d_preprocessing import load_and_clean_data, prepare_lstm_data

# ===================================
# Paths
# ===================================
DATA_PATH = "choosen_ds/AAPL_yfinance.csv"
MODEL_PATH = "aapl_lstm_model.h5"

# ===================================
# Load & Preprocess Data
# ===================================
print("Loading and preprocessing dataset...")
try:
    df = load_and_clean_data(DATA_PATH)
    scaler, X_train, y_train = prepare_lstm_data(df)

    print(f"Dataset loaded with {len(df)} rows")
    print(f"Training samples: {X_train.shape[0]}")

except FileNotFoundError:
    print(f"ERROR: Dataset not found at {DATA_PATH}")
    print("Please check the file path and try again.")
    exit()
except Exception as e:
    print(f"ERROR: An error occurred during data loading: {e}")
    exit()


# ===================================
# Build LSTM Model
# ===================================
print("Building LSTM model...")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# ===================================
# Train Model (Efficient Method)
# ===================================
epochs = 30
batch_size = 32

print(f"\nStarting training for {epochs} epochs...\n")

# Replaced the slow for-loop with a single, efficient .fit() call.
# validation_split=0.2 automatically uses 20% of data for validation.
# verbose=1 will print the epoch log (loss & val_loss) automatically.
history = model.fit(
    X_train, 
    y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=1,
    validation_split=0.2 
)

print("\nTraining complete!\n")

# ===================================
# Save Model
# ===================================
model.save(MODEL_PATH)
print(f"Model saved successfully as '{MODEL_PATH}'")

# ===================================
# Plot Loss (Training vs. Validation)
# ===================================
print("Generating training loss plot...")
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', color='royalblue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orangered')
plt.title('Model Training & Validation Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
print("Displaying plot. Close the plot window to exit.")
plt.show()
