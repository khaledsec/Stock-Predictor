import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import finnhub
from polygon import RESTClient
import pandas as pd
import os
import time
from datetime import date, timedelta

# --- 1. CONFIGURATION & API KEYS ---
print("Starting all downloads...")

# PASTE ALL YOUR API KEYS HERE
ALPHA_KEY = "J9IWAJYMAS6W0S1S"
FINNHUB_KEY = "d3v7g71r01qt2ctoes6gd3v7g71r01qt2ctoes70"
POLYGON_KEY = "ysc_xtZSwHRcEkeVDojKQHJNsHorO38k" # Your key is already in!

# Define the 8 tickers for your project
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ", "BTC-USD"]

# Define output directory
OUTPUT_DIR = "datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 2. YFINANCE DOWNLOAD ---
print("\n--- 1/4: Downloading from Yahoo Finance (yfinance) ---")
# yfinance can handle long date ranges
YF_START = "2010-01-01"
YF_END = date.today().strftime("%Y-%m-%d")

for ticker in TICKERS:
    print(f"Fetching [yfinance] {ticker}...")
    try:
        data = yf.download(ticker, start=YF_START, end=YF_END)
        data.reset_index(inplace=True)
        data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        filename = os.path.join(OUTPUT_DIR, f"{ticker}_yfinance.csv")
        data.to_csv(filename, index=False)
        print(f"  Saved {ticker} to {filename}")
    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")


# --- 3. ALPHA VANTAGE DOWNLOAD ---
print("\n--- 2/4: Downloading from Alpha Vantage ---")
try:
    ts = TimeSeries(key=ALPHA_KEY, output_format='pandas')
    for ticker in TICKERS:
        if ticker == "BTC-USD": # Alpha Vantage uses a different ticker for crypto
            print("  (Skipping BTC-USD for Alpha Vantage)")
            continue
        
        print(f"Fetching [Alpha] {ticker}...")
        try:
            # Get daily data, full history
            data, _ = ts.get_daily(symbol=ticker, outputsize='full')
            data.reset_index(inplace=True)
            # Rename columns to match our standard
            data.rename(columns={
                'index': 'Date',
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }, inplace=True)
            # Keep only standard columns
            data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
            filename = os.path.join(OUTPUT_DIR, f"{ticker}_alpha.csv")
            data.to_csv(filename, index=False)
            print(f" Saved {ticker} to {filename}")
            
            # IMPORTANT: Alpha Vantage has a 5-call-per-minute limit
            print("  ...Waiting 15 seconds to avoid API limit...")
            time.sleep(15)
            
        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")
except Exception as e:
    print(f" Error connecting to Alpha Vantage (check API key?): {e}")


# --- 4. FINNHUB DOWNLOAD ---
print("\n--- 3/4: Downloading from Finnhub ---")
try:
    finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)
    # Finnhub also uses a long date range (as UNIX timestamps)
    FH_START = int(pd.Timestamp("2010-01-01").timestamp())
    FH_END = int(pd.Timestamp.today().timestamp())

    for ticker in TICKERS:
        if ticker == "BTC-USD": # Finnhub uses a different API for crypto
            print("  (Skipping BTC-USD for Finnhub)")
            continue
        
        print(f"Fetching [Finnhub] {ticker}...")
        try:
            res = finnhub_client.stock_candles(ticker, 'D', FH_START, FH_END)
            if res.get('s') != 'ok':
                print(f" Error (from Finnhub API): {res}")
                continue
                
            df = pd.DataFrame(res)
            # Rename columns to match our standard
            df['Date'] = pd.to_datetime(df['t'], unit='s').dt.date
            df.rename(columns={
                'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'
            }, inplace=True)
            
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
            filename = os.path.join(OUTPUT_DIR, f"{ticker}_finnhub.csv")
            df.to_csv(filename, index=False)
            print(f"  Saved {ticker} to {filename}")

        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")
except Exception as e:
    print(f"  Error connecting to Finnhub (check API key?): {e}")


# --- 5. POLYGON.IO DOWNLOAD ---
print("\n--- 4/4: Downloading from Polygon.io ---")
# Polygon free tier is limited to ~2 years
POLY_END = date.today()
POLY_START = POLY_END - timedelta(days=2 * 365)
POLY_START_STR = POLY_START.strftime("%Y-%m-%d")
POLY_END_STR = POLY_END.strftime("%Y-%m-%d")

try:
    client = RESTClient(POLYGON_KEY)
    for ticker in TICKERS:
        # Polygon uses a "X:" prefix for crypto
        poly_ticker = f"X:{ticker}" if ticker == "BTC-USD" else ticker
        
        print(f"Fetching [Polygon] {poly_ticker}...")
        try:
            resp = client.get_aggs(poly_ticker, 1, "day", POLY_START_STR, POLY_END_STR, limit=50000)
            if not resp:
                print(f"  No data returned for {poly_ticker}. Skipping.")
                continue

            df = pd.DataFrame(resp)
            # Rename columns
            df['Date'] = pd.to_datetime(df['t'], unit='ms').dt.date
            df.rename(columns={
                'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'
            }, inplace=True)
            
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            filename = os.path.join(OUTPUT_DIR, f"{ticker}_polygon.csv")
            df.to_csv(filename, index=False)
            print(f"  Saved {len(df)} records to {filename}")
        
        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")
except Exception as e:
    print(f"  Error connecting to Polygon (check API key?): {e}")

print("\nAll download tasks complete")
