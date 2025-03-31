import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
from typing import Tuple, Dict, List, Any
import pickle
import matplotlib.dates as mdates

# Create a directory for saving results
os.makedirs('results', exist_ok=True)

# Suppress warnings
warnings.filterwarnings('ignore')

def fetch_stock_data(symbol: str, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance with error handling.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date for historical data
        end_date: End date for historical data (default: today)
        
    Returns:
        DataFrame with historical stock data
    """
    try:
        if end_date is None:
            end_date = datetime.now()
            
        print(f"Fetching data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"No data found for {symbol}, using sample data instead")
            return generate_sample_data(start_date, end_date)
        
        print(f"Successfully fetched {len(df)} days of data")
        return df
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        print("Using sample data instead")
        return generate_sample_data(start_date, end_date)

def generate_sample_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Generate sample stock data if the API fails."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    np.random.seed(42)
    base_price = 150.0
    close_prices = base_price + np.cumsum(np.random.normal(0.001, 0.02, len(date_range)))
    
    df = pd.DataFrame({
        'Open': close_prices * (1 - np.random.uniform(0, 0.01, len(date_range))),
        'High': close_prices * (1 + np.random.uniform(0, 0.02, len(date_range))),
        'Low': close_prices * (1 - np.random.uniform(0, 0.02, len(date_range))),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, len(date_range))
    }, index=date_range)
    
    print(f"Generated sample data with {len(df)} days")
    return df

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for feature engineering."""
    try:
        print("Calculating technical indicators...")
        df_copy = df.copy()
        
        # Moving Averages
        for window in [5, 10, 20, 50]:
            df_copy[f'MA{window}'] = df_copy['Close'].rolling(window=window).mean()
        
        # Exponential Moving Averages
        for window in [5, 10, 20, 50]:
            df_copy[f'EMA{window}'] = df_copy['Close'].ewm(span=window, adjust=False).mean()
        
        # Price differences
        df_copy['Price_Diff'] = df_copy['Close'].diff()
        df_copy['Price_Diff_Pct'] = df_copy['Close'].pct_change() * 100
        
        # RSI (Relative Strength Index)
        delta = df_copy['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        rs = rs.fillna(0)
        df_copy['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df_copy['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_copy['Close'].ewm(span=26, adjust=False).mean()
        df_copy['MACD'] = exp1 - exp2
        df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
        df_copy['MACD_Hist'] = df_copy['MACD'] - df_copy['MACD_Signal']
        
        # Bollinger Bands
        df_copy['BB_Middle'] = df_copy['Close'].rolling(window=20).mean()
        bb_std = df_copy['Close'].rolling(window=20).std()
        df_copy['BB_Upper'] = df_copy['BB_Middle'] + 2 * bb_std
        df_copy['BB_Lower'] = df_copy['BB_Middle'] - 2 * bb_std
        df_copy['BB_Width'] = (df_copy['BB_Upper'] - df_copy['BB_Lower']) / df_copy['BB_Middle']
        
        # Volatility
        df_copy['Volatility'] = df_copy['Close'].rolling(window=20).std()
        df_copy['Volatility_Pct'] = df_copy['Volatility'] / df_copy['Close'] * 100
        
        # Volume indicators
        df_copy['Volume_MA5'] = df_copy['Volume'].rolling(window=5).mean()
        df_copy['Volume_MA20'] = df_copy['Volume'].rolling(window=20).mean()
        df_copy['Volume_Ratio'] = df_copy['Volume'] / df_copy['Volume_MA20']
        
        # High-Low Range
        df_copy['HL_Range'] = (df_copy['High'] - df_copy['Low']) / df_copy['Close'] * 100
        df_copy['OC_Range'] = abs(df_copy['Open'] - df_copy['Close']) / df_copy['Close'] * 100
        
        # Trend indicators
        df_copy['Trend_20'] = (df_copy['Close'] / df_copy['Close'].shift(20) - 1) * 100
        df_copy['Trend_50'] = (df_copy['Close'] / df_copy['Close'].shift(50) - 1) * 100
        
        # Moving Average Crossovers
        df_copy['MA_Cross_5_20'] = (df_copy['MA5'] > df_copy['MA20']).astype(int)
        df_copy['MA_Cross_10_50'] = (df_copy['MA10'] > df_copy['MA50']).astype(int)
        
        # Momentum
        df_copy['Momentum_14'] = df_copy['Close'] - df_copy['Close'].shift(14)
        df_copy['Momentum_30'] = df_copy['Close'] - df_copy['Close'].shift(30)
        
        print("Successfully calculated technical indicators")
        return df_copy
        
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        raise

def prepare_features(df: pd.DataFrame, prediction_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare features for the model with targets for specified prediction horizon.
    
    Args:
        df: DataFrame with technical indicators
        prediction_horizon: Number of days to predict ahead (default: 1)
        
    Returns:
        X: Feature matrix
        y: Target vector
        df_clean: Processed DataFrame
    """
    try:
        print(f"Preparing features for {prediction_horizon}-day ahead prediction...")
        
        # Create target variable (future closing price)
        df['Target'] = df['Close'].shift(-prediction_horizon)
        
        # Create percentage change target (can be more stable for prediction)
        df['Target_Pct_Change'] = df['Target'] / df['Close'] - 1
        
        # Remove columns with too many missing values
        missing_threshold = 0.8 * len(df)
        valid_columns = [col for col in df.columns if df[col].count() >= missing_threshold]
        df = df[valid_columns]
        
        # Select features - exclude target and date-related columns
        exclude_cols = ['Target', 'Target_Pct_Change', 'Dividends', 'Stock Splits']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Forward fill remaining NaN values
        df_clean = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining rows with NaN values
        df_clean = df_clean.dropna()
        print(f"Prepared dataset with {len(df_clean)} samples")
        
        X = df_clean[feature_cols]
        y = df_clean['Target']  # Can use 'Target_Pct_Change' for percentage predictions
        
        print(f"Selected {len(feature_cols)} features")
        return X, y, df_clean
        
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        raise

def tune_hyperparameters(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """
    Tune hyperparameters using GridSearchCV with TimeSeriesSplit.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Tuned model and best parameters
    """
    try:
        print("Tuning model hyperparameters...")
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
        }
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize random forest regressor
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            verbose=0,
            n_jobs=-1
        )
        
        # Fit the grid search model
        grid_search.fit(X, y)
        
        # Get best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        print(f"Best parameters found: {best_params}")
        print(f"Best cross-validation score: {-grid_search.best_score_:.2f} MSE")
        
        return best_model, best_params
        
    except Exception as e:
        print(f"Error tuning hyperparameters: {str(e)}")
        # Fallback to default model if tuning fails
        print("Using default model parameters")
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ), {}

def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        model: Trained RandomForestRegressor
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # Percentage of correct direction predictions
        actual_direction = np.sign(y_test.values - X_test['Close'].values)
        pred_direction = np.sign(y_pred - X_test['Close'].values)
        direction_accuracy = np.mean(actual_direction == pred_direction)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape,
            'Direction Accuracy': direction_accuracy
        }
        
        print("Model Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        return metrics
        
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        return {}

def backtest_model(model: RandomForestRegressor, df: pd.DataFrame, 
                   feature_cols: List[str], prediction_horizon: int = 1,
                   start_idx: int = None, windows: int = 5) -> Tuple[List[float], List[float]]:
    """
    Perform backtesting on historical data using expanding window.
    
    Args:
        model: Trained model
        df: DataFrame with features
        feature_cols: List of feature column names
        prediction_horizon: Days ahead to predict
        start_idx: Starting index for backtest
        windows: Number of test windows
        
    Returns:
        Tuple of actual and predicted prices
    """
    try:
        print("Performing backtesting...")
        df_backtest = df.copy()
        
        if start_idx is None:
            start_idx = int(len(df_backtest) * 0.6)  # Default start at 60% of data
            
        window_size = (len(df_backtest) - start_idx) // windows
        
        all_actual = []
        all_predicted = []
        
        for i in range(windows):
            # Define train/test indices for this window
            test_start = start_idx + i * window_size
            test_end = min(test_start + window_size, len(df_backtest))
            
            # Extract train/test sets
            train_df = df_backtest.iloc[:test_start].copy()
            test_df = df_backtest.iloc[test_start:test_end].copy()
            
            # Prepare target variable
            train_df['Target'] = train_df['Close'].shift(-prediction_horizon)
            train_df = train_df.dropna()
            
            # Train model on this window
            X_train = train_df[feature_cols]
            y_train = train_df['Target']
            model.fit(X_train, y_train)
            
            # Make predictions
            X_test = test_df[feature_cols]
            predictions = model.predict(X_test)
            
            # Store actual and predicted values
            actuals = test_df['Close'].values
            
            all_actual.extend(actuals)
            all_predicted.extend(predictions)
            
            print(f"Window {i+1}/{windows}: {len(actuals)} samples, "
                  f"MAPE: {mean_absolute_percentage_error(actuals, predictions):.4f}")
            
        return all_actual, all_predicted
        
    except Exception as e:
        print(f"Error during backtesting: {str(e)}")
        return [], []

def plot_feature_importance(model: RandomForestRegressor, feature_names: List[str], top_n: int = 15) -> None:
    """Plot top N feature importances."""
    try:
        # Get feature importance
        importance = model.feature_importances_
        
        # Create DataFrame
        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_imp.head(top_n))
        plt.title(f'Top {top_n} Feature Importances', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')
        print("Feature importance plot saved to 'results/feature_importance.png'")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting feature importance: {str(e)}")

def plot_predictions_vs_actual(dates: List[datetime], actual: List[float], 
                           predicted: List[float], symbol: str,
                           prediction_dates: List[datetime] = None, 
                           future_predictions: List[float] = None) -> None:
    """
    Plot actual vs predicted prices with optional future predictions.
    
    Args:
        dates: Dates for historical data
        actual: Actual prices
        predicted: Predicted prices
        symbol: Stock symbol
        prediction_dates: Dates for future predictions
        future_predictions: Future predicted prices
    """
    try:
        plt.figure(figsize=(12, 6))
        
        # Set up date formatting
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        
        # Plot historical data
        plt.plot(dates, actual, 'b-', label='Actual Price', linewidth=2)
        plt.plot(dates, predicted, 'g--', label='Model Prediction', linewidth=2, alpha=0.7)
        
        # Calculate and display metrics
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        direction_accuracy = np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100
        
        # Add future predictions if provided
        if prediction_dates is not None and future_predictions is not None:
            plt.plot(prediction_dates, future_predictions, 'r--', label='Future Prediction', linewidth=2)
            
            # Add vertical line separating historical and future
            plt.axvline(x=dates[-1], color='gray', linestyle='--', alpha=0.7)
            plt.annotate('Forecast Start', xy=(dates[-1], min(actual)),
                        xytext=(dates[-1], min(actual) - (max(actual) - min(actual))*0.1),
                        rotation=90, va='top', ha='center')
        
        # Add metrics to plot
        plt.annotate(f'MAPE: {mape:.2f}%\nDirection Accuracy: {direction_accuracy:.2f}%',
                    xy=(0.02, 0.92), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.title(f'{symbol} Stock Price - Actual vs Prediction', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig('results/prediction_vs_actual.png')
        print("Prediction vs Actual plot saved to 'results/prediction_vs_actual.png'")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting predictions: {str(e)}")

def predict_future_prices(model: RandomForestRegressor, latest_data: pd.DataFrame, 
                      feature_cols: List[str], days: int = 21) -> Tuple[List[datetime], List[float]]:
    """
    Predict future stock prices.
    
    Args:
        model: Trained model
        latest_data: DataFrame with latest available data
        feature_cols: List of feature column names
        days: Number of days to predict
        
    Returns:
        Future dates and predicted prices
    """
    try:
        print(f"Predicting prices for next {days} trading days...")
        # Get the last row of data
        last_data = latest_data.iloc[-1:].copy()
        
        # Create future dates (business days)
        future_dates = pd.date_range(
            start=latest_data.index[-1] + pd.Timedelta(days=1), 
            periods=days, 
            freq='B'
        )
        
        future_predictions = []
        
        # Make sequential predictions
        for i in range(days):
            # Ensure all required features are present
            if not all(col in last_data.columns for col in feature_cols):
                missing = [col for col in feature_cols if col not in last_data.columns]
                print(f"Missing features for prediction: {missing}")
                # Initialize missing columns with zeros
                for col in missing:
                    last_data[col] = 0
            
            # Make prediction
            prediction = model.predict(last_data[feature_cols])[0]
            future_predictions.append(prediction)
            
            # Update last_data for next prediction (simplified approach)
            last_data['Close'] = prediction
            last_data['Open'] = prediction * (1 + np.random.normal(0, 0.005))
            last_data['High'] = prediction * (1 + abs(np.random.normal(0, 0.01)))
            last_data['Low'] = prediction * (1 - abs(np.random.normal(0, 0.01)))
            
        # Create future DataFrame
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_predictions
        })
        
        print(f"Future predictions generated for dates: {future_dates[0].date()} to {future_dates[-1].date()}")
        return list(future_dates), future_predictions
        
    except Exception as e:
        print(f"Error predicting future prices: {str(e)}")
        return [], []

def save_model(model: RandomForestRegressor, symbol: str) -> None:
    """Save trained model to disk."""
    try:
        model_path = f'results/{symbol}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def load_model(symbol: str) -> RandomForestRegressor:
    """Load trained model from disk."""
    try:
        model_path = f'results/{symbol}_model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def interactive_stock_prediction() -> None:
    """Interactive function for user to predict stock prices."""
    try:
        print("\n===== ADVANCED STOCK PRICE PREDICTION =====\n")
        
        # Get user input
        symbol = input("Enter stock symbol (e.g., AAPL, MSFT, GOOGL): ").strip().upper()
        if not symbol:
            symbol = "AAPL"
            print(f"No symbol entered. Using default: {symbol}")
            
        # Get prediction horizon
        try:
            horizon = int(input("Enter prediction horizon in days (default: 1): ").strip() or "1")
        except ValueError:
            horizon = 1
            print(f"Invalid input. Using default horizon: {horizon} day")
            
        # Get historical data duration
        try:
            years = int(input("Enter years of historical data (default: 2): ").strip() or "2")
        except ValueError:
            years = 2
            print(f"Invalid input. Using default: {years} years of data")
        
        # Prepare dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        # Fetch historical data
        df = fetch_stock_data(symbol, start_date, end_date)
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Prepare features
        X, y, df_clean = prepare_features(df, prediction_horizon=horizon)
        
        # Check if model exists
        model_exists = os.path.exists(f'results/{symbol}_model.pkl')
        
        if model_exists:
            load_existing = input("Found existing model. Load it? (y/n, default: y): ").strip().lower() != 'n'
            if load_existing:
                model = load_model(symbol)
                if model is None:
                    print("Failed to load model. Training new model...")
                    model, best_params = tune_hyperparameters(X, y)
            else:
                print("Training new model with hyperparameter tuning...")
                model, best_params = tune_hyperparameters(X, y)
        else:
            print("Training new model with hyperparameter tuning...")
            model, best_params = tune_hyperparameters(X, y)
        
        # Split data (use last 20% for testing to maintain time series order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model if not loaded
        if not model_exists or not load_existing:
            print("Training model...")
            model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Perform backtesting
        backtest_actual, backtest_predicted = backtest_model(
            model, 
            df_clean, 
            list(X.columns),
            prediction_horizon=horizon
        )
        
        # Plot feature importance
        plot_feature_importance(model, list(X.columns))
        
        # Plot backtest results
        plot_predictions_vs_actual(
            df_clean.index[-len(backtest_actual):],
            backtest_actual,
            backtest_predicted,
            symbol
        )
        
        # Save model
        save_model(model, symbol)
        
        # Predict future prices
        future_dates, future_predictions = predict_future_prices(
            model,
            df_clean,
            list(X.columns)
        )
        
        # Plot historical and future predictions
        plot_predictions_vs_actual(
            df_clean.index[-30:],
            df_clean['Close'].values[-30:],
            model.predict(X.iloc[-30:]),
            symbol,
            future_dates,
            future_predictions
        )
        
        # Print future predictions
        future_df = pd.DataFrame({
            'Date': [d.date() for d in future_dates],
            'Predicted_Price': future_predictions
        })
        print("\nPredicted Prices for the Next Trading Days:")
        print(future_df)
        
        print(f"\nAnalysis for {symbol} completed successfully!")
        print("You can find the results in the 'results' directory")
        
    except Exception as e:
        print(f"Error in interactive analysis: {str(e)}")

if __name__ == "__main__":
    interactive_stock_prediction() 