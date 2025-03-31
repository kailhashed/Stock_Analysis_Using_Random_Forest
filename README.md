# Stock Price Prediction using Random Forest

This project implements a stock price prediction model using Random Forest algorithm. The model predicts future stock prices based on historical data and technical indicators.

## Features

- Fetches historical stock data using yfinance (with offline fallback)
- Implements technical indicators as features
- Uses Random Forest for price prediction
- Includes model evaluation and visualization
- Interactive input for stock symbol
- One-month future price prediction visualization

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

There are multiple ways to run the code:

### 1. Interactive Stock Prediction Script:

```bash
python interactive_stock_prediction.py
```

This will:
- Prompt you to enter a stock symbol (e.g., AAPL, MSFT, GOOGL)
- Download historical stock data for the selected symbol
- Train the Random Forest model
- Show and save a visualization of the next month's predicted prices
- Display the most important features for prediction

### 2. Run the Standard Python Script:

```bash
python stock_prediction_fixed.py
```

This will:
- Download historical stock data for Apple (AAPL)
- Calculate technical indicators
- Train the Random Forest model
- Save results and visualizations in the 'results' directory

### 3. Run the Jupyter Notebook:

```bash
jupyter notebook stock_prediction_notebook.ipynb
```

The notebook provides an interactive interface to:
- Explore the data and visualizations
- Adjust parameters
- See the results interactively

### 4. Custom Stock Analysis:

To analyze a different stock, modify the symbol in the script or notebook, or use the interactive script.

## Troubleshooting

If you encounter any issues:

1. **Network Errors**: The code includes a fallback to generate sample data if the API fails
2. **Missing Dependencies**: Make sure you've installed all requirements
3. **Plotting Issues**: The Python script uses a non-interactive backend to avoid plotting errors

## Model Features

The model uses the following technical indicators as features:
- Moving Averages (5, 10, 20 days)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators
- Price Change
- High-Low Range

## Files in the Repository

- `interactive_stock_prediction.py`: Interactive script for stock prediction with user input
- `stock_prediction_fixed.py`: Robust Python script for stock prediction
- `stock_prediction_notebook.ipynb`: Interactive Jupyter notebook
- `requirements.txt`: Dependencies for the project 