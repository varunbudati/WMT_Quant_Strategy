import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

def calculate_indicators(df, volatility_window, rsi_window):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=volatility_window).std() * np.sqrt(252)
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def prepare_features(df, sentiment_score, interest_rate, economic_growth):
    df['Target'] = df['Close'].shift(-1)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'Volatility', 'RSI']
    X = df[features]
    y = df['Target']
    valid_data = X.notna().all(axis=1) & y.notna()
    X = X[valid_data]
    y = y[valid_data]
    
    # Add new factors as features
    X['Sentiment'] = sentiment_score
    X['Interest_Rate'] = interest_rate
    X['Economic_Growth'] = economic_growth
    
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse, predictions

def adjust_predictions(predictions, sentiment_score, interest_rate, economic_growth):
    sentiment_impact = 0.02 * sentiment_score
    interest_rate_impact = -0.01 * (interest_rate - 2)
    growth_impact = 0.005 * economic_growth
    
    total_impact = 1 + sentiment_impact + interest_rate_impact + growth_impact
    adjusted_predictions = predictions * total_impact
    
    return adjusted_predictions

def predict_future(model, last_data, days, sentiment_score, interest_rate, economic_growth):
    future_predictions = []
    current_data = last_data.copy()
    for _ in range(days):
        prediction = model.predict(current_data.reshape(1, -1))[0]
        adjusted_prediction = adjust_predictions([prediction], sentiment_score, interest_rate, economic_growth)[0]
        future_predictions.append(adjusted_prediction)
        current_data = np.roll(current_data, -1)
        current_data[-3:] = [sentiment_score, interest_rate, economic_growth]
        current_data[-4] = adjusted_prediction

    return future_predictions

def plot_results_with_future(df, historical_predictions, future_predictions, future_dates):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(df.index, df['Close'], label='Actual Price', color='blue')
    ax.plot(df.index[-len(historical_predictions):], historical_predictions, label='Historical Predictions', color='green')
    ax.plot(future_dates, future_predictions, label='Future Predictions', color='red', linestyle='--')
    ax.set_title('Walmart Stock Price: Historical and Future Predictions')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.axvline(x=df.index[-1], color='gray', linestyle='--')
    ax.text(df.index[-1], ax.get_ylim()[1], 'Today', ha='right', va='top')
    
    return fig

def main():
    st.title('Walmart Stock Analysis and Prediction for Pitch SEED')
    st.warning("Disclaimer: This is purely Quantitative and only done as an experiment")
    st.sidebar.header('User Input Parameters')
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
    future_days = st.sidebar.slider("Days to predict in the future", 1, 60, 30)
    
    volatility_window = st.sidebar.slider("Volatility Window", 5, 50, 20)
    rsi_window = st.sidebar.slider("RSI Window", 5, 30, 14)
    sentiment_score = st.sidebar.slider("Market Sentiment (-1 to 1)", -1.0, 1.0, 0.0, 0.1)
    interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 10.0, 2.0, 0.1)
    economic_growth = st.sidebar.slider("Economic Growth (%)", -5.0, 10.0, 2.0, 0.1)
    
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    else:
        st.sidebar.error('Error: End date must fall after start date.')
    
    with st.spinner('Fetching stock data...'):
        walmart_df = fetch_stock_data('WMT', start_date, end_date)
    
    walmart_df = calculate_indicators(walmart_df, volatility_window, rsi_window)
    X, y = prepare_features(walmart_df, sentiment_score, interest_rate, economic_growth)
    
    if len(X) < 10:
        st.error("Not enough valid data points. Please select a larger date range.")
        return
    
    with st.spinner('Training the model...'):
        model, X_test, y_test = train_model(X, y)
    
    rmse, historical_predictions = evaluate_model(model, X_test, y_test)
    st.write(f"Root Mean Squared Error: {rmse:.2f}")
    
    last_known_data = X.iloc[-1].values
    future_predictions = predict_future(model, last_known_data, future_days, sentiment_score, interest_rate, economic_growth)
    future_dates = pd.date_range(start=walmart_df.index[-1] + timedelta(days=1), periods=future_days)
    
    st.subheader('Historical Data and Future Predictions')
    fig = plot_results_with_future(walmart_df, historical_predictions, future_predictions, future_dates)
    st.pyplot(fig)
    
    st.subheader('Recent Stock Data and Future Predictions')
    recent_data = walmart_df.tail()
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions})
    future_df.set_index('Date', inplace=True)
    combined_df = pd.concat([recent_data, future_df])
    st.dataframe(combined_df)
    
    st.subheader('Feature Importance')
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    st.bar_chart(feature_importance.set_index('feature'))
    
    st.warning("Disclaimer: These predictions are for educational purposes only. Stock market prediction is inherently uncertain and these results should not be used for actual trading decisions.")

if __name__ == "__main__":
    main()