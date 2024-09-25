import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

def calculate_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    return df

def prepare_features(df):
    df['Target'] = df['Close'].shift(-1)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'Volatility']
    X = df[features]
    y = df['Target']
    
    # Remove rows with NaN values
    valid_data = X.notna().all(axis=1) & y.notna()
    X = X[valid_data]
    y = y[valid_data]
    
    return X, y

def main():
    st.title('Walmart Stock Analysis App')
    
    st.sidebar.header('User Input Parameters')
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
    
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    else:
        st.sidebar.error('Error: End date must fall after start date.')
    
    # Fetch Walmart stock data
    with st.spinner('Fetching stock data...'):
        walmart_df = fetch_stock_data('WMT', start_date, end_date)
    
    # Calculate technical indicators
    walmart_df = calculate_indicators(walmart_df)
    
    # Prepare features and target variable
    X, y = prepare_features(walmart_df)
    
    # Check if we have enough data
    if len(X) < 10:  # You can adjust this threshold
        st.error("Not enough valid data points. Please select a larger date range.")
        return
    
    # Train the model
    with st.spinner('Training the model...'):
        model, X_test, y_test = train_model(X, y)
    
    # Evaluate the model
    rmse, predictions = evaluate_model(model, X_test, y_test)
    st.write(f"Root Mean Squared Error: {rmse:.2f}")
    
    # Plot results
    st.subheader('Actual vs Predicted Stock Prices')
    fig = plot_results(walmart_df, predictions)
    st.pyplot(fig)
    
    # Display recent data
    st.subheader('Recent Stock Data')
    st.dataframe(walmart_df.tail())
    
    # Feature importance
    st.subheader('Feature Importance')
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    st.bar_chart(feature_importance.set_index('feature'))

if __name__ == "__main__":
    main()