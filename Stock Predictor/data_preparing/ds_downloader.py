
import pandas as pd
from polygon import RESTClient
from datetime import date, timedelta
import os

# --- 1. CONFIGURATION ---
# PASTE YOUR POLYGON API KEY HERE
POLYGON_KEY = "ysc_xtZSwHRcEkeVDojKQHJNsHorO38k" 

TICKERS = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "SPY", "QQQ"]
OUTPUT_DIR = "datasets"

# Set date range (Polygon's free tier gives ~2 years of daily data)
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=2 * 365)

# Format for API
START_STR = START_DATE.strftime("%Y-%m-%d")
END_STR = END_DATE.strftime("%Y-%m-%d")

print(f"Connecting to Polygon.io to download data from {START_STR} to {END_STR}...")

# --- 2. CREATE CLIENT ---
try:
    client = RESTClient(POLYGON_KEY)
except Exception as e:
    print(f"Error connecting to Polygon: {e}")
    print("Please make sure your API key is correct in POLYGON_KEY.")
    exit()

# --- 3. DOWNLOAD LOOP ---
for ticker in TICKERS:
    print(f"\nFetching {ticker} from Polygon...")
    try:
        # Get aggregates (daily bars)
        resp = client.get_aggs(
            ticker,
            1,
            "day",
            START_STR,
            END_STR,
            limit=50000
        )

        if not resp:
            print(f"No data returned for {ticker}. Skipping.")
            continue

        # Convert to DataFrame
        df = pd.DataFrame(resp)
        
        # Rename columns to match your standard format
        df['Date'] = pd.to_datetime(df['t'], unit='ms').dt.date
        df.rename(columns={
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        }, inplace=True)
        
        # Keep only the standard columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Save to CSV with the '_polygon.csv' suffix
        filename = os.path.join(OUTPUT_DIR, f"{ticker}_polygon.csv")
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} records to {filename}")

    except Exception as e:
        print(f"Error downloading {ticker}: {e}")

print("\nPolygon download complete.")