import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns

# Load the data
data = pd.read_csv('coffee_sales.csv')

# Convert date columns to datetime
data['date'] = pd.to_datetime(data['date'])
data['datetime'] = pd.to_datetime(data['datetime'])

# Feature engineering - create daily aggregates
daily_data = data.groupby('date').agg({
    'money': 'sum',  # Daily revenue
    'coffee_name': 'count',  # Daily demand (number of orders)
    'cash_type': lambda x: (x == 'cash').sum()  # Cash transactions count
}).rename(columns={'money': 'revenue', 'coffee_name': 'demand', 'cash_type': 'cash_transactions'})

# Add day of week, month, and other temporal features
daily_data['day_of_week'] = daily_data.index.dayofweek
daily_data['month'] = daily_data.index.month
daily_data['day_of_month'] = daily_data.index.day
daily_data['is_weekend'] = (daily_data['day_of_week'] >= 5).astype(int)

# Add lag features for time series
for lag in [1, 2, 3, 7]:
    daily_data[f'revenue_lag_{lag}'] = daily_data['revenue'].shift(lag)
    daily_data[f'demand_lag_{lag}'] = daily_data['demand'].shift(lag)

# Add rolling features
daily_data['revenue_rolling_7_mean'] = daily_data['revenue'].shift(1).rolling(7).mean()
daily_data['demand_rolling_7_mean'] = daily_data['demand'].shift(1).rolling(7).mean()

# Drop rows with NaN values created by lag features
daily_data = daily_data.dropna()

# Separate features and targets
X = daily_data.drop(['revenue', 'demand'], axis=1)
y_revenue = daily_data['revenue']
y_demand = daily_data['demand']

# Percentage-based train-test split (80-20)
test_size = 0.2
split_idx = int(len(daily_data) * (1 - test_size))

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_revenue_train, y_revenue_test = y_revenue.iloc[:split_idx], y_revenue.iloc[split_idx:]
y_demand_train, y_demand_test = y_demand.iloc[:split_idx], y_demand.iloc[split_idx:]

# Create a pipeline with scaling and model
pipeline_revenue = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline_demand = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the models
pipeline_revenue.fit(X_train, y_revenue_train)
pipeline_demand.fit(X_train, y_demand_train)

# Make predictions
revenue_pred = pipeline_revenue.predict(X_test)
demand_pred = pipeline_demand.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{model_name} Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Mean Actual Value: {y_true.mean():.2f}")
    print(f"Mean Predicted Value: {y_pred.mean():.2f}")
    print("-" * 50)
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true, label='Actual', marker='o')
    plt.plot(y_true.index, y_pred, label='Predicted', marker='x')
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel(model_name)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

evaluate_model(y_revenue_test, revenue_pred, "Daily Revenue")
evaluate_model(y_demand_test, demand_pred, "Daily Demand")

# Feature importance analysis
def plot_feature_importance(model, features, title):
    importance = model.named_steps['model'].feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.bar(range(len(features)), importance[indices], align='center')
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

plot_feature_importance(pipeline_revenue, X.columns, "Revenue Prediction Feature Importance")
plot_feature_importance(pipeline_demand, X.columns, "Demand Prediction Feature Importance")

# Future prediction function
def predict_future(days_to_predict=7):
    # Start from the last date in our data
    last_date = daily_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
    
    # Create a DataFrame for future predictions
    future_data = pd.DataFrame(index=future_dates)
    
    # Initialize with the last known values
    for lag in [1, 2, 3, 7]:
        future_data[f'revenue_lag_{lag}'] = np.nan
        future_data[f'demand_lag_{lag}'] = np.nan
    
    # Fill initial lag values from historical data
    for i, date in enumerate(future_dates):
        for lag in [1, 2, 3, 7]:
            if i < lag:
                # Get lag from historical data
                lag_date = date - pd.Timedelta(days=lag)
                if lag_date in daily_data.index:
                    future_data.loc[date, f'revenue_lag_{lag}'] = daily_data.loc[lag_date, 'revenue']
                    future_data.loc[date, f'demand_lag_{lag}'] = daily_data.loc[lag_date, 'demand']
    
    # Add temporal features
    future_data['day_of_week'] = future_data.index.dayofweek
    future_data['month'] = future_data.index.month
    future_data['day_of_month'] = future_data.index.day
    future_data['is_weekend'] = (future_data['day_of_week'] >= 5).astype(int)
    
    # Initialize cash transactions with historical average
    future_data['cash_transactions'] = daily_data['cash_transactions'].mean()
    
    # Initialize rolling means with historical data
    last_week_revenue = daily_data['revenue'].iloc[-7:].mean()
    last_week_demand = daily_data['demand'].iloc[-7:].mean()
    future_data['revenue_rolling_7_mean'] = last_week_revenue
    future_data['demand_rolling_7_mean'] = last_week_demand
    
    # Make predictions day by day, updating lag features
    predicted_revenue = []
    predicted_demand = []
    
    for i, date in enumerate(future_dates):
        # Prepare features for this day
        features = future_data.loc[[date]].copy()
        
        # Fill any remaining NaN values with historical averages
        for col in features.columns:
            if features[col].isna().any():
                features[col] = daily_data[col].mean()
        
        # Make predictions
        rev_pred = pipeline_revenue.predict(features)[0]
        dem_pred = pipeline_demand.predict(features)[0]
        
        predicted_revenue.append(rev_pred)
        predicted_demand.append(dem_pred)
        
        # Update future lag features for subsequent days
        for j in range(i+1, min(i+8, len(future_dates))):
            lag = j - i
            if lag in [1, 2, 3, 7]:
                future_data.loc[future_dates[j], f'revenue_lag_{lag}'] = rev_pred
                future_data.loc[future_dates[j], f'demand_lag_{lag}'] = dem_pred
    
    future_data['predicted_revenue'] = predicted_revenue
    future_data['predicted_demand'] = predicted_demand
    
    return future_data[['predicted_revenue', 'predicted_demand']]

# Predict next 7 days
future_predictions = predict_future(7)
print("\nFuture Predictions for Next 7 Days:")
print(future_predictions)

# Plot future predictions
plt.figure(figsize=(12, 6))
plt.plot(daily_data.index[-30:], daily_data['revenue'][-30:], label='Historical Revenue', marker='o')
plt.plot(future_predictions.index, future_predictions['predicted_revenue'], label='Predicted Revenue', marker='x', color='red')
plt.title('Revenue - Historical and Future Predictions')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(daily_data.index[-30:], daily_data['demand'][-30:], label='Historical Demand', marker='o')
plt.plot(future_predictions.index, future_predictions['predicted_demand'], label='Predicted Demand', marker='x', color='red')
plt.title('Demand - Historical and Future Predictions')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()