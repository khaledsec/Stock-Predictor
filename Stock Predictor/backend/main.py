import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import uvicorn
import pandas as pd
import numpy as np
import io
import sqlite3
import bcrypt
import joblib
import yfinance as yf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     username TEXT UNIQUE, 
                     password TEXT)''')
    conn.commit()
    conn.close()

init_db()

class UserAuth(BaseModel):
    username: str
    password: str

class TickerRequest(BaseModel):
    ticker: str

app = FastAPI(title="Stock Prediction API with Auth & Live Data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ticker_models = {}
last_trained_ticker = None
MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)


TIME_STEP = 60
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'ma_7', 'ma_21', 'ma_50',
    'ema_12', 'ema_26', 'macd', 'macd_signal',
    'rsi_14', 'volatility_14', 'daily_return',
    'bb_upper', 'bb_lower', 'high_low_range'
]
CLOSE_IDX = 3


def add_features(df):
    df = df.copy()

    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_21'] = df['close'].rolling(window=21).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()

    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50)

    df['daily_return'] = df['close'].pct_change()
    df['volatility_14'] = df['daily_return'].rolling(window=14).std()

    bb_mid = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = bb_mid + (2 * bb_std)
    df['bb_lower'] = bb_mid - (2 * bb_std)

    df['high_low_range'] = (df['high'] - df['low']) / df['close']

    df = df.dropna().reset_index(drop=True)
    return df

def create_dataset(scaled_features, raw_close, time_step=60):
    X, Y = [], []
    for i in range(len(scaled_features) - time_step - 1):
        X.append(scaled_features[i:(i + time_step), :])
        Y.append(raw_close[i + time_step] / raw_close[i + time_step - 1] - 1.0)
    return np.array(X), np.array(Y)

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0: return 0.0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def save_ticker_model(ticker, model, scaler):
    ticker_dir = os.path.join(MODELS_DIR, ticker.upper())
    os.makedirs(ticker_dir, exist_ok=True)
    model.save(os.path.join(ticker_dir, "model.h5"))
    joblib.dump(scaler, os.path.join(ticker_dir, "scaler.joblib"))

def load_ticker_model(ticker):
    ticker_dir = os.path.join(MODELS_DIR, ticker.upper())
    model_path = os.path.join(ticker_dir, "model.h5")
    scaler_path = os.path.join(ticker_dir, "scaler.joblib")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return load_model(model_path), joblib.load(scaler_path)
    return None, None

def train_model_logic(df, ticker="UNKNOWN"):
    global ticker_models, last_trained_ticker

    df = add_features(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    data = df[FEATURE_COLUMNS].values
    num_features = len(FEATURE_COLUMNS)

    raw_split = int(len(data) * 0.8)
    train_data = data[:raw_split]
    test_data = data[raw_split:]

    current_scaler = MinMaxScaler(feature_range=(0, 1))
    current_scaler.fit(data)
    scaled_train = current_scaler.transform(train_data)
    scaled_test = current_scaler.transform(test_data)

    train_close = train_data[:, CLOSE_IDX]
    test_close = test_data[:, CLOSE_IDX]
    X_train, Y_train = create_dataset(scaled_train, train_close, TIME_STEP)
    X_test, Y_test = create_dataset(scaled_test, test_close, TIME_STEP)
    if len(X_train) == 0:
        raise ValueError("Not enough data to train. Need at least 61 rows after feature engineering.")

    current_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(TIME_STEP, num_features)),
        Dropout(0.2),
        LSTM(48, return_sequences=False),
        Dropout(0.2),
        Dense(24, activation='relu'),
        Dense(1)
    ])
    current_model.compile(optimizer='adam', loss='mean_absolute_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

    epochs = 60
    batch_size = 32
    history = current_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                                 validation_data=(X_test, Y_test),
                                 callbacks=[early_stop, reduce_lr])

    predicted_returns = current_model.predict(X_test).flatten()

    n = len(predicted_returns)
    last_closes = test_close[TIME_STEP - 1:TIME_STEP - 1 + n]
    real_prices = test_close[TIME_STEP:TIME_STEP + n]
    test_predictions = last_closes * (1.0 + predicted_returns)

    test_start_idx = raw_split + TIME_STEP
    chart_df = df.iloc[test_start_idx:test_start_idx + len(real_prices)].copy()
    chart_df['predicted'] = test_predictions
    chart_df['real'] = real_prices

    baseline_mae = mean_absolute_error(real_prices, last_closes)

    ticker_key = ticker.upper()
    ticker_models[ticker_key] = {
        "model": current_model,
        "scaler": current_scaler,
        "latest_data_df": df,
        "all_predictions_df": chart_df,
    }

    save_ticker_model(ticker_key, current_model, current_scaler)
    last_trained_ticker = ticker_key

    return real_prices, test_predictions, history.history, baseline_mae

@app.post("/register")
async def register(user: UserAuth):
    conn = get_db_connection()
    salt = bcrypt.gensalt()
    hashed_pwd = bcrypt.hashpw(user.password.encode('utf-8'), salt)
    try:
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user.username, hashed_pwd.decode('utf-8')))
        conn.commit()
        return {"message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        conn.close()

@app.post("/login")
async def login(user: UserAuth):
    conn = get_db_connection()
    db_user = conn.execute("SELECT * FROM users WHERE username = ?", (user.username,)).fetchone()
    conn.close()
    if not db_user or not bcrypt.checkpw(user.password.encode('utf-8'), db_user["password"].encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"message": "Login successful", "username": user.username}

@app.get("/")
def read_root():
    return {"message": "API is running. Please Login to continue."}

@app.post("/upload_and_train")
async def upload_and_train(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.rename(columns=lambda x: x.strip().lower(), inplace=True)

        required = ['date', 'close', 'open', 'high', 'low', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"CSV must have columns: {required}. Missing: {missing}")

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['close'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        ticker_name = file.filename.replace(".csv", "").upper()
        real_prices, predictions, history, baseline_mae = train_model_logic(df, ticker=ticker_name)

        mae = mean_absolute_error(real_prices, predictions)
        rmse = np.sqrt(mean_squared_error(real_prices, predictions))
        mape = calculate_mape(real_prices, predictions)

        return {
            "message": "Model trained successfully from CSV",
            "metrics": {
                "mae": f"${mae:.2f}", "rmse": f"${rmse:.2f}", "mape": f"{mape:.2f}%",
                "baseline_mae": f"${baseline_mae:.2f}",
                "beats_baseline": float(mae) < float(baseline_mae),
            },
            "training_history": {"loss": [float(l) for l in history['loss']], "val_loss": [float(l) for l in history['val_loss']]}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_ticker")
async def train_ticker(req: TickerRequest):
    try:
        ticker = req.ticker.upper().strip()
        print(f"Fetching Live Data for {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y")

        if df.empty:
            raise HTTPException(status_code=404, detail="Ticker not found or no data available.")

        df = df.reset_index()
        df.rename(columns=lambda x: x.strip().lower(), inplace=True)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)

        real_prices, predictions, history, baseline_mae = train_model_logic(df, ticker=ticker)

        mae = mean_absolute_error(real_prices, predictions)
        rmse = np.sqrt(mean_squared_error(real_prices, predictions))
        mape = calculate_mape(real_prices, predictions)

        return {
            "message": f"Model trained successfully on Live {ticker} data!",
            "metrics": {
                "mae": f"${mae:.2f}", "rmse": f"${rmse:.2f}", "mape": f"{mape:.2f}%",
                "baseline_mae": f"${baseline_mae:.2f}",
                "beats_baseline": float(mae) < float(baseline_mae),
            },
            "training_history": {"loss": [float(l) for l in history['loss']], "val_loss": [float(l) for l in history['val_loss']]}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint el News (Updated format for yfinance)
@app.get("/news/{ticker}")
def get_news(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        news_data = stock.news
        cleaned_news = []
        for item in news_data[:4]: # Byrg3 a5er 4 a5bar
            content = item.get("content", item)
            cleaned_news.append({
                "title": content.get("title", "News Article Unavailable"),
                "publisher": content.get("provider", {}).get("displayName", content.get("publisher", "Yahoo Finance")),
                "link": content.get("clickThroughUrl", content.get("link", "#"))
            })
        return {"news": cleaned_news}
    except Exception as e:
        print(f"Error fetching news: {e}")
        return {"news": []}

@app.get("/all_predictions")
def get_all_predictions():
    if last_trained_ticker is None or last_trained_ticker not in ticker_models:
        return []
    entry = ticker_models[last_trained_ticker]
    if entry["all_predictions_df"] is None:
        return []
    df_json = entry["all_predictions_df"].copy()
    df_json['date'] = df_json['date'].dt.strftime('%Y-%m-%d')
    return df_json[['date', 'real', 'predicted']].to_dict('records')

@app.get("/predict_next_day")
def predict_next_day():
    if last_trained_ticker is None or last_trained_ticker not in ticker_models:
        raise HTTPException(status_code=503, detail="Model is not trained. Please upload a dataset and train first.")

    entry = ticker_models[last_trained_ticker]
    current_model = entry["model"]
    current_scaler = entry["scaler"]
    latest_data_df = entry["latest_data_df"]

    if current_model is None or current_scaler is None or latest_data_df is None:
        raise HTTPException(status_code=503, detail="Model is not trained. Please upload a dataset and train first.")

    try:
        data = latest_data_df[FEATURE_COLUMNS].values
        last_60_days = data[-TIME_STEP:]
        last_60_days_scaled = current_scaler.transform(last_60_days)
        X_input = np.array([last_60_days_scaled])

        predicted_return = current_model.predict(X_input)[0][0]

        latest_close = data[-1, CLOSE_IDX]
        predicted_price = latest_close * (1.0 + predicted_return)

        return {"prediction": float(predicted_price)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    print("Starting API server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)