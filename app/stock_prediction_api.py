import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import pickle
import os
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import uvicorn

# Model class for handling ARIMA predictions
class ARIMAStockPredictor:
    def __init__(self, ticker="AAPL", p=5, d=1, q=0):
        self.ticker = ticker
        self.order = (p, d, q)
        self.model = None
        self.model_fit = None
        self.data = None
        self.last_date = None
    
    def fetch_data(self, period="5y"):
        """Fetch stock data from Yahoo Finance"""
        try:
            print(f"Downloading {self.ticker} data for period {period}")
            stock_data = yf.download(self.ticker, period=period)
            
            print(f"Data columns available: {stock_data.columns.tolist()}")
            
            # Create a price series from Close price
            # Note: We directly extract 'Close' without trying to use nested indexing
            self.data = pd.DataFrame()
            self.data['price'] = stock_data['Close']
            
            print(f"Data shape: {self.data.shape}")
            print(f"First few rows:\n{self.data.head()}")
            
            self.last_date = self.data.index[-1]
            print(f"Last date in dataset: {self.last_date}")
            
            return self.data
            
        except Exception as e:
            print(f"Error in fetch_data: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def check_stationarity(self):
        """Check if the time series is stationary using ADF test"""
        result = adfuller(self.data['price'].dropna())
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        is_stationary = result[1] < 0.05
        print(f'Series is {"stationary" if is_stationary else "not stationary"}')
        return is_stationary
    
    def train_model(self):
        """Train the ARIMA model"""
        # If data is not already loaded, fetch it
        if self.data is None:
            self.fetch_data()
        
        print(f"Training ARIMA model with order {self.order}")
        self.model = ARIMA(self.data['price'], order=self.order)
        self.model_fit = self.model.fit()
        print(self.model_fit.summary())
        return self.model_fit
    
    def predict(self, days=7):
        """Make predictions for specified number of days"""
        if self.model_fit is None:
            raise ValueError("Model must be trained before making predictions")
        
        forecast = self.model_fit.forecast(steps=days)
        forecast_dates = pd.date_range(start=self.last_date + timedelta(days=1), periods=days)
        
        # Create a DataFrame with predictions
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'predicted_price': forecast
        })
        forecast_df.set_index('date', inplace=True)
        
        return forecast_df
    
    def save_model(self, filename="arima_model.pkl"):
        """Save the trained model to a file"""
        if self.model_fit is None:
            raise ValueError("Model must be trained before saving")
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'model_fit': self.model_fit,
                'ticker': self.ticker,
                'order': self.order,
                'last_date': self.last_date,
                'last_price': self.data['price'].iloc[-1]
            }, f)
        
        print(f"Model saved to {filename}")
    
    @classmethod
    def load_model(cls, filename="arima_model.pkl"):
        """Load a trained model from a file"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found")
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls(ticker=model_data['ticker'])
        predictor.order = model_data['order']
        predictor.model_fit = model_data['model_fit']
        predictor.last_date = model_data['last_date']
        
        # Initialize with minimal data
        predictor.data = pd.DataFrame({'price': [model_data['last_price']]}, 
                                     index=[model_data['last_date']])
        
        return predictor

# FastAPI Models
class PredictionRequest(BaseModel):
    days: int = 7

class PredictionResponse(BaseModel):
    ticker: str
    predictions: dict
    last_updated: str
    
######################################################################################################################
######################################################################################################################
######################################################################################################################

# FastAPI app
app = FastAPI(
    title="Stock Price Prediction API",
    description="API for predicting stock prices using ARIMA model.",
    version="1.0.0"
)

# Global predictor object
predictor = None

# Function to initialize model on first request
def initialize_model():
    global predictor
    if predictor is None or predictor.model_fit is None:
        try:
            print("Initializing model...")
            predictor = ARIMAStockPredictor(ticker="AAPL", p=5, d=1, q=0)
            predictor.fetch_data(period="5y")
            predictor.train_model()
            predictor.save_model()
            print("Model initialized successfully")
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Apple Stock Price Prediction API. Use /predict to get predictions."}

@app.get("/predict")
def predict(days: Optional[int] = Query(7, ge=1, le=30)):
    """Get stock price predictions for the specified number of days"""
    global predictor
    
    # Initialize model if not already done
    if predictor is None or predictor.model_fit is None:
        try:
            initialize_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not initialize model: {str(e)}")
    
    # Make predictions
    try:
        forecast = predictor.predict(days=days)
        
        # Convert predictions to a dictionary for the response
        predictions = {
            date.strftime('%Y-%m-%d'): round(price, 2) 
            for date, price in zip(forecast.index, forecast['predicted_price'])
        }
        
        return PredictionResponse(
            ticker=predictor.ticker,
            predictions=predictions,
            last_updated=predictor.last_date.strftime('%Y-%m-%d')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/retrain")
def retrain_model():
    """Retrain the model with fresh data"""
    global predictor
    try:
        initialize_model()
        return {"message": "Model retrained successfully", "ticker": predictor.ticker}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining model: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("stock_prediction_api:app", host="0.0.0.0", port=8001, reload=True)