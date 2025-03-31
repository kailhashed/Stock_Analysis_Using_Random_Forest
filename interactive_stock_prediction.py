import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import warnings
import os
import sys

# Create a directory for saving results
os.makedirs('results', exist_ok=True)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance with error handling.
    """
    try:
        print(f"Fetching data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        # Use a try block to handle potential network errors
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"No data found for {symbol}, using sample data instead")
            # Generate sample data if API fails
            return generate_sample_data(start_date, end_date)
        
        print(f"Successfully fetched {len(df)} days of data")
        return df
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        print("Using sample data instead")
        return generate_sample_data(start_date, end_date)

def generate_sample_data(start_date, end_date):
    """
    Generate sample stock data if the API fails.
    """
    # Create a date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate random data
    np.random.seed(42)  # For reproducibility
    base_price = 150.0
    
    # Generate price data with a trend
    close_prices = base_price + np.cumsum(np.random.normal(0.001, 0.02, len(date_range)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': close_prices * (1 - np.random.uniform(0, 0.01, len(date_range))),
        'High': close_prices * (1 + np.random.uniform(0, 0.02, len(date_range))),
        'Low': close_prices * (1 - np.random.uniform(0, 0.02, len(date_range))),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, len(date_range))
    }, index=date_range)
    
    print(f"Generated sample data with {len(df)} days")
    return df

def calculate_technical_indicators(df):
    """
    Calculate technical indicators with error handling.
    """
    try:
        print("Calculating technical indicators...")
        # Moving Averages
        for window in [5, 10, 20]:
            df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Avoid division by zero
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        rs = rs.fillna(0)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * bb_std
        df['BB_lower'] = df['BB_middle'] - 2 * bb_std
        
        # Volume indicators
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        
        # Additional features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
        
        print("Successfully calculated technical indicators")
        return df
        
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        raise

def prepare_features(df):
    """
    Prepare features for the model with error handling.
    """
    try:
        print("Preparing features...")
        # Create target variable (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        # Select features for the model
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'Signal_Line',
            'BB_middle', 'BB_upper', 'BB_lower',
            'Volume_MA5', 'Volume_MA20',
            'Price_Change', 'High_Low_Range'
        ]
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        print(f"Removed {len(df) - len(df_clean)} rows with NaN values")
        
        if len(df_clean) < 30:
            print("Not enough data points after cleaning. Using forward fill for NaN values.")
            df = df.fillna(method='ffill').fillna(method='bfill')
            df_clean = df
        
        X = df_clean[features]
        y = df_clean['Target']
        
        print(f"Prepared {len(X)} samples with {len(features)} features")
        return X, y, df_clean
        
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        raise

def train_model(X, y, test_size=0.2):
    """
    Train the Random Forest model with error handling.
    """
    try:
        print("Training the model...")
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Initialize and train the model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all available CPU cores
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained successfully. MSE: {mse:.2f}, R²: {r2:.2f}")
        return model, X_test, y_test, y_pred, mse, r2
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise

def predict_next_month(model, df, features):
    """
    Predict stock prices for the next month (21 trading days).
    """
    try:
        print("Predicting stock prices for the next month (21 trading days)...")
        # Get the last row of data
        last_data = df.iloc[-1:].copy()
        # Create future dates (21 trading days ≈ 1 month)
        future_dates = pd.date_range(
            start=df.index[-1] + pd.Timedelta(days=1), 
            periods=21, 
            freq='B'  # Business days
        )
        future_predictions = []
        
        # Lists to store all predictions
        all_dates = list(df.index[-30:]) + list(future_dates)
        all_prices = list(df['Close'].values[-30:])
        
        for i in range(21):
            # For the first prediction, use the last available data
            if i == 0:
                X_pred = last_data[features].values
            else:
                # Update the features for the next day's prediction
                # This is a simplified approach - in reality, you would need to calculate
                # all technical indicators for each future day based on previous predictions
                for feature in features:
                    if feature in ['Open', 'High', 'Low', 'Volume']:
                        # Keep them relatively constant (with small random changes)
                        last_data[feature] = last_data['Close'] * (
                            1 + np.random.normal(0, 0.005)
                        )
                X_pred = last_data[features].values
            
            # Make prediction for the next day
            next_price = model.predict(X_pred)[0]
            
            # Store the prediction
            future_predictions.append(next_price)
            all_prices.append(next_price)
            
            # Update the Close price for the next prediction
            last_data['Close'] = next_price
        
        future_df = pd.DataFrame({
            'Predicted_Price': future_predictions
        }, index=future_dates)
        
        return future_df, all_dates, all_prices
        
    except Exception as e:
        print(f"Error predicting future prices: {str(e)}")
        raise

def plot_predictions(historical_dates, historical_prices, future_dates, future_prices, symbol):
    """
    Plot historical prices and future predictions.
    """
    try:
        # Combine dates and prices
        all_dates = historical_dates + future_dates
        all_prices = historical_prices + future_prices
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(
            historical_dates, 
            historical_prices, 
            'b-', 
            label='Historical Prices'
        )
        
        # Plot predictions
        plt.plot(
            future_dates, 
            future_prices, 
            'r--', 
            label='Predicted Prices'
        )
        
        # Add vertical line to separate historical data from predictions
        plt.axvline(
            x=historical_dates[-1], 
            color='gray', 
            linestyle='--', 
            alpha=0.7
        )
        
        # Add text annotation
        plt.text(
            historical_dates[-1], 
            min(all_prices), 
            'Prediction Start', 
            rotation=90, 
            verticalalignment='bottom'
        )
        
        plt.title(f'{symbol} Stock Price Prediction for Next Month')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout for better display
        plt.tight_layout()
        
        # Save and show the plot
        plt.savefig('results/next_month_prediction.png')
        print(f"Prediction plot saved to 'results/next_month_prediction.png'")
        plt.show()
        
    except Exception as e:
        print(f"Error plotting predictions: {str(e)}")

def interactive_stock_prediction():
    """
    Interactive function to get user input and run prediction.
    """
    try:
        print("\n===== STOCK PRICE PREDICTION USING RANDOM FOREST =====\n")
        
        # Get user input for stock symbol
        symbol = input("Enter stock symbol (e.g., AAPL, MSFT, GOOGL): ").strip().upper()
        
        if not symbol:
            symbol = "AAPL"
            print(f"No symbol entered. Using default: {symbol}")
        
        # Set up dates - use 1 year of historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Fetch historical data
        df = fetch_stock_data(symbol, start_date, end_date)
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Prepare features
        X, y, df_clean = prepare_features(df)
        
        # Train model
        model, X_test, y_test, y_pred, mse, r2 = train_model(X, y)
        
        # Predict next month
        future_df, historical_dates, historical_prices = predict_next_month(model, df_clean, X.columns)
        
        # Print future predictions
        print("\nPredicted Prices for Next 21 Trading Days (≈ 1 month):")
        print(future_df)
        
        # Plot results with both historical and future data
        plot_predictions(
            list(df_clean.index[-30:]), 
            list(df_clean['Close'].values[-30:]),
            list(future_df.index),
            list(future_df['Predicted_Price'].values),
            symbol
        )
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Important Features:")
        print(feature_importance.head(5))
        
        print(f"\nAnalysis for {symbol} completed successfully!")
        print("You can find the prediction plot in the 'results' directory")
        
    except Exception as e:
        print(f"Error in the interactive analysis: {str(e)}")

if __name__ == "__main__":
    interactive_stock_prediction() 