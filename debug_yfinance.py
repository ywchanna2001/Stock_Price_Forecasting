import yfinance as yf
import pandas as pd

def debug_yahoo_finance():
    print("Fetching AAPL data from Yahoo Finance...")
    stock_data = yf.download("AAPL", period="1y")
    
    print("\nData shape:", stock_data.shape)
    print("\nColumn names:", stock_data.columns.tolist())
    print("\nFirst 5 rows:")
    print(stock_data.head())
    print(stock_data.tail())
    
    # Try to access columns
    try:
        if 'Adj Close' in stock_data.columns:
            print("\nAdj Close column exists")
            price_data = stock_data['Adj Close']
        else:
            print("\nAdj Close column does NOT exist")
            if 'Close' in stock_data.columns:
                print("Using Close column instead")
                price_data = stock_data['Close']
            else:
                print("Neither Adj Close nor Close columns exist")
                print("Available columns:", stock_data.columns.tolist())
    except Exception as e:
        print(f"\nError accessing data: {type(e).__name__}: {str(e)}")
        print("Full exception information:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_yahoo_finance()