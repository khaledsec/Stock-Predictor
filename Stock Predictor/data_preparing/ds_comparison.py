
import sys
import importlib.util

# Ensure pandas is installed before proceeding so the script fails fast with a clear message.
spec = importlib.util.find_spec("pandas")
if spec is None:
    print("Error: required package 'pandas' is not installed. Install it with:")
    print("    python -m pip install pandas")
    sys.exit(1)

import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Define Path ---
# All data is in this single folder
DATASET_DIR = "datasets" 

# --- Column Standardization Helper ---
def standardize_columns(df):
    """
    Renames columns to a standard format (e.g., 'Date', 'Close', 'Ticker')
    This handles variations like 'date', 'close', 'Name', or 'Symbol'.
    """
    df.rename(columns={
        'date': 'Date',
        'close': 'Close',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'volume': 'Volume',
        'Name': 'Ticker',    # For the Kaggle file
        'Symbol': 'Ticker' # Another common name
    }, inplace=True)
    return df

# --- Helper function to load all datasets ---
def load_datasets():
    data = []
    
    print(f"Loading all data from: {DATASET_DIR}")
    
    # Get all CSVs except for the analysis files we create
    files_to_load = [
        f for f in os.listdir(DATASET_DIR) 
        if f.endswith(".csv") 
        and "summary" not in f 
        and "volatility" not in f
    ]
    
    if not files_to_load:
        print(f"Error: No data CSVs found in '{DATASET_DIR}'.")
        print("Please run your download scripts first.")
        return pd.DataFrame()

    for f in files_to_load:
        path = os.path.join(DATASET_DIR, f)
        
        try:
            # --- Handle the Kaggle file specially ---
            if f == "all_stocks_5yr.csv":
                df_kaggle = pd.read_csv(path)
                df_kaggle = standardize_columns(df_kaggle)
                
                if "Ticker" not in df_kaggle.columns:
                    print(f"Error: Kaggle file {f} missing 'Ticker' column. Skipping.")
                    continue
                
                df_kaggle["Source"] = "Kaggle"
                
                # Filter Kaggle file for only the 8 tickers
                tickers_to_compare = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ", "BTC-USD"]
                df_kaggle = df_kaggle[df_kaggle["Ticker"].isin(tickers_to_compare)]
                
                print(f"Loaded {f} (Kaggle) and filtered for {tickers_to_compare}")
                data.append(df_kaggle)

            # --- Handle all other source files (Yahoo, Alpha, etc.) ---
            else:
                source = "Yahoo" if "yfinance" in f else \
                         "Alpha" if "alpha" in f else \
                         "Polygon" if "polygon" in f else \
                         "Finnhub" if "finnhub" in f else "Other"
                
                ticker_base = f.split("_")[0]
                
                df = pd.read_csv(path)
                if df.empty:
                    print(f"Skipping empty file: {f}")
                    continue
                    
                df = standardize_columns(df)
                df["Source"] = source
                
                if 'Ticker' not in df.columns:
                    df["Ticker"] = ticker_base
                    
                data.append(df)
                print(f"Loaded {f} ({source})")

        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {f}")
        except Exception as e:
            print(f"Error loading {f}: {e}")

    # --- 3. Combine All Data ---
    if not data:
        print("\nError: No data was loaded.")
        return pd.DataFrame()
        
    return pd.concat(data, ignore_index=True)

# ====================================================
#  RUN THE ANALYSIS
# ====================================================
combined = load_datasets()

if not combined.empty:
    print(f"\nLoaded {len(combined)} total rows from all sources.\n")

    # --- Ensure correct data types ---
    combined["Date"] = pd.to_datetime(combined["Date"], errors='coerce')
    combined["Close"] = pd.to_numeric(combined["Close"], errors='coerce')
    combined.dropna(subset=["Date", "Close"], inplace=True)
    combined.sort_values(["Ticker", "Date"], inplace=True)

    # ========================
    #  Summary statistics
    # ========================
    summary = combined.groupby(["Source", "Ticker"]).agg(
        Records=("Close", "count"),
        Start_Date=("Date", "min"),
        End_Date=("Date", "max"),
        Mean_Close=("Close", "mean"),
        Std_Close=("Close", "std")
    ).reset_index()

    print(" Summary Table:\n")
    print(summary.to_string()) # .to_string() prints all rows
    summary.to_csv(os.path.join(DATASET_DIR, "source_comparison_summary.csv"), index=False)

    # ========================
    # Missing value analysis
    # ========================
    missing = combined.isna().sum()
    print("\n Missing Data Report:\n", missing)

    # ========================
    # Volatility calculation
    # ========================
    volatility_data = []
    for (src, tick), group in combined.groupby(["Source", "Ticker"]):
        group = group.sort_values("Date")
        group["Daily_Return"] = group["Close"].pct_change(fill_method=None)
        vol = group["Daily_Return"].std() * 100
        volatility_data.append({"Source": src, "Ticker": tick, "Volatility(%)": vol})

    volatility_df = pd.DataFrame(volatility_data)
    print("\nVolatility Summary:\n", volatility_df.to_string()) # .to_string() prints all rows
    volatility_df.to_csv(os.path.join(DATASET_DIR, "volatility_comparison.csv"), index=False)

    # ========================
    # Visualization
    # ========================
    tickers_to_plot = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ", "BTC-USD"]
    sources_to_plot = ["Yahoo", "Alpha", "Kaggle", "Polygon", "Finnhub"] 

    print("\nGenerating charts...")
    for ticker in tickers_to_plot:
        plt.figure(figsize=(12, 6))
        has_data = False
        
        for source in sources_to_plot:
            subset = combined[(combined["Ticker"] == ticker) & (combined["Source"] == source)]
            if len(subset) > 0:
                plt.plot(subset["Date"], subset["Close"], label=f"{source} ({len(subset)} records)")
                has_data = True
        
        if has_data:
            plt.title(f"{ticker} Close Price Comparison (All Sources)")
            plt.xlabel("Date")
            plt.ylabel("Close Price")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            chart_filename = f"{ticker}_comparison_chart.png"
            plt.savefig(os.path.join(DATASET_DIR, chart_filename))
            print(f"  - Saved {chart_filename}")
        else:
            print(f"  - No data found for {ticker}, skipping chart.")
            
        plt.close()

    print("\nAll outputs saved in 'datasets/' folder.")
else:
    print("Script finished: No data was loaded.")