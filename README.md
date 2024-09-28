# WMT_Quant_Strategy

This is for Educational Purposes only

1. Technical Indicators:
   The code calculates several technical indicators commonly used in stock analysis.

   ```python
   def calculate_indicators(df):
       df['SMA_50'] = df['Close'].rolling(window=50).mean()  # Simple Moving Average (50-day)
       df['SMA_200'] = df['Close'].rolling(window=200).mean()  # Simple Moving Average (200-day)
       df['Daily_Return'] = df['Close'].pct_change()  # Daily returns
       df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)  # Annualized volatility
       return df
   ```

   - Simple Moving Averages (SMA): 50-day and 200-day SMAs are calculated.
   - Daily Returns: Percentage change in closing price from one day to the next.
   - Volatility: 20-day rolling standard deviation of daily returns, annualized.

2. Feature Engineering:
   The code prepares features for the machine learning model.

   ```python
   def prepare_features(df):
       df['Target'] = df['Close'].shift(-1)  # Next day's closing price as the target
       features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'Volatility']
       X = df[features]
       y = df['Target']
       # ... (code to remove NaN values)
       return X, y
   ```

   This function selects relevant features and creates a target variable (next day's closing price).

3. Machine Learning Model:
   The code uses a Random Forest Regressor for prediction.

   ```python
   def train_model(X, y):
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model = RandomForestRegressor(n_estimators=100, random_state=42)
       model.fit(X_train, y_train)
       return model, X_test, y_test
   ```

   - Random Forest: An ensemble learning method that constructs multiple decision trees and outputs the average prediction of the individual trees.
   - Train-Test Split: The data is split into training (80%) and testing (20%) sets.

4. Model Evaluation:
   The model's performance is evaluated using Root Mean Squared Error (RMSE).

   ```python
   def evaluate_model(model, X_test, y_test):
       predictions = model.predict(X_test)
       mse = mean_squared_error(y_test, predictions)
       rmse = np.sqrt(mse)
       return rmse, predictions
   ```

   RMSE measures the standard deviation of the residuals (prediction errors).

5. Future Prediction:
   The code attempts to predict future stock prices using the trained model.

   ```python
   def predict_future(model, last_data, days=30):
       future_predictions = []
       current_data = last_data.copy()
       for _ in range(days):
           prediction = model.predict(current_data.reshape(1, -1))[0]
           future_predictions.append(prediction)
           current_data = np.roll(current_data, -1)
           current_data[-1] = prediction
       return future_predictions
   ```

   This function uses the last known data point to predict the next day, then uses that prediction to predict the following day, and so on.

6. Feature Importance:
   The code analyzes which features have the most impact on the predictions.

   ```python
   feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
   feature_importance = feature_importance.sort_values('importance', ascending=False)
   st.bar_chart(feature_importance.set_index('feature'))
   ```

   This uses the `feature_importances_` attribute of the Random Forest model to rank the importance of each feature.

These quantitative methods combine technical analysis (through the use of technical indicators) with machine learning (Random Forest) to analyze historical stock data and attempt to make predictions. 


