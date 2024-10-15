import json
import streamlit as st
from streamlit_echarts import st_echarts
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime, date
import calendar
from dateutil.relativedelta import relativedelta

cwd = os.path.dirname(__file__)

def load_and_preprocess_data():
    coffee_prices_df = pd.read_csv(cwd + "/data/price/PCOFFOTMUSDM.csv")
    commodity_df = pd.read_csv(cwd + "/data/price/WPU026301.csv")
    
    final_df = pd.merge(coffee_prices_df, commodity_df, how="outer", on="DATE")
    # final_df["PCOFFOTMUSDM"] = final_df["PCOFFOTMUSDM"].fillna(0)
    # final_df["WPU026301"] = final_df["WPU026301"].fillna(0)
    final_df['DATE'] = pd.to_datetime(final_df['DATE'])
    final_df = final_df.sort_values('DATE')
    
    return final_df

def prepare_data_for_model(df):
    df['price_change'] = df['PCOFFOTMUSDM'].diff().apply(lambda x: 1 if x > 0 else 0)
    df['days_since_start'] = (df['DATE'] - df['DATE'].min()).dt.days
    df['price_lag1'] = df['PCOFFOTMUSDM'].shift(1)
    df['price_lag2'] = df['PCOFFOTMUSDM'].shift(2)
    df = df.dropna()
    
    X = df[['days_since_start', 'WPU026301', 'price_lag1', 'price_lag2']]
    y = df['price_change']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, df

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        multi_class='ovr',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

def predict_future_prices_dual(model, scaler, df, end_date):
    last_date = df['DATE'].max()
    months_to_predict = (end_date - last_date).days // 30
    future_dates = pd.date_range(start=last_date +  relativedelta(months=+1), periods=months_to_predict)

    future_df = pd.DataFrame({'DATE': future_dates})
    future_df['days_since_start'] = (future_df['DATE'] - df['DATE'].min()).dt.days

    # Initialize with last known values
    future_df['PCOFFOTMUSDM_lag1'] = df['PCOFFOTMUSDM'].iloc[-1]
    future_df['PCOFFOTMUSDM_lag2'] = df['PCOFFOTMUSDM'].iloc[-2]
    future_df['WPU026301'] = df['WPU026301'].iloc[-1]

    future_pcoffotmusdm_prices = []
    future_wpu_prices = []

    # Predict future values for both PCOFFOTMUSDM and WPU026301
    for _, row in future_df.iterrows():
        input_data = scaler.transform([[row['days_since_start'], row['WPU026301'], row['PCOFFOTMUSDM_lag1'], row['PCOFFOTMUSDM_lag2']]])
        prob = model.predict_proba(input_data)[0][1]  # Probability of price increase

        # Predict price change for PCOFFOTMUSDM
        if prob > 0.5:
            pcoffotmusdm_price_change = row['PCOFFOTMUSDM_lag1'] * 0.01  # Assume 1% increase
        else:
            pcoffotmusdm_price_change = -row['PCOFFOTMUSDM_lag1'] * 0.01  # Assume 1% decrease

        new_pcoffotmusdm_price = row['PCOFFOTMUSDM_lag1'] + pcoffotmusdm_price_change
        future_pcoffotmusdm_prices.append(new_pcoffotmusdm_price)

        # Predict WPU026301 price change (similar logic)
        wpu_price_change = row['WPU026301'] * 0.01  # Assume 1% change for WPU
        new_wpu_price = row['WPU026301'] + wpu_price_change
        future_wpu_prices.append(new_wpu_price)

        # Update lags for the next prediction
        row['PCOFFOTMUSDM_lag2'] = row['PCOFFOTMUSDM_lag1']
        row['PCOFFOTMUSDM_lag1'] = new_pcoffotmusdm_price
        row['WPU026301'] = new_wpu_price

    future_df['PCOFFOTMUSDM'] = future_pcoffotmusdm_prices
    future_df['WPU026301'] = future_wpu_prices

    return future_df


def render_price_chart():
    df = load_and_preprocess_data()
    X, y, scaler, df = prepare_data_for_model(df)
    model, accuracy = train_model(X, y)

    # Predict future prices for both PCOFFOTMUSDM and WPU026301
    future_df = predict_future_prices_dual(model, scaler, df, datetime(2024, 10, 1))

    # Combine actual and future data
    combined_df = pd.concat([df, future_df])

    # Separate actual and future data for plotting
    actual_mask = combined_df['DATE'] <= pd.Timestamp(datetime(2023, 12, 31))
    future_mask = combined_df['DATE'] > pd.Timestamp(datetime(2023, 12, 31))

    # Extract actual and future data for plotting
    actual_dates = combined_df.loc[actual_mask, 'DATE'].dt.strftime('%Y-%m-%d').tolist()
    future_dates = combined_df.loc[future_mask, 'DATE'].dt.strftime('%Y-%m-%d').tolist()

    actual_pcoff_prices = combined_df.loc[actual_mask, 'PCOFFOTMUSDM'].tolist()
    future_pcoff_prices = combined_df.loc[future_mask, 'PCOFFOTMUSDM'].tolist()

    actual_wpu_prices = combined_df.loc[actual_mask, 'WPU026301'].tolist()
    future_wpu_prices = combined_df.loc[future_mask, 'WPU026301'].tolist()

    options = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["PCOFFOTMUSDM (Actual)", "PCOFFOTMUSDM (Predicted)", "WPU026301 (Actual)", "WPU026301 (Predicted)"]},
        "color": [
            "#610C04", "#610C04", "#1C90FF", "#1C90FF"
        ],
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        # "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": actual_dates + future_dates,
        },
        "yAxis": {"type": "value"},
        "series": [
            {
                "name": "PCOFFOTMUSDM (Actual)",
                "type": "line",
                "data": actual_pcoff_prices + [None] * len(future_pcoff_prices),
                "smooth": True,
                "lineStyle": {"color": "#610C04"},  
            },
            {
                "name": "PCOFFOTMUSDM (Predicted)",
                "type": "line",
                "data": [None] * len(actual_pcoff_prices) + future_pcoff_prices,
                "smooth": True,
                "lineStyle": {"color": "#610C04", "type": "dashed"},  
            },
            {
                "name": "WPU026301 (Actual)",
                "type": "line",
                "data": actual_wpu_prices + [None] * len(future_wpu_prices),
                "smooth": True,
                "lineStyle": {"color": "#1C90FF"},  
            },
            {
                "name": "WPU026301 (Predicted)",
                "type": "line",
                "data": [None] * len(actual_wpu_prices) + future_wpu_prices,
                "smooth": True,
                "lineStyle": {"color": "#1C90FF", "type": "dashed"},  
            }
        ],
    }

    left, right = st.columns([4, 1])
    with left:
        st.header("Coffeee Prices")
        st_echarts(options=options, height="400px")
    with right.expander("Learn more", icon="ℹ"):
        st.markdown("""
            - **PCOFFOTMUSDM**: Global price of Coffee, Other Mild Arabica
            - **WPU026301**: Producer Price Index by Commodity: Processed Foods and Feeds: Coffee (Whole Bean, Ground, and Instant)
            - Future prices are projected based on the trained model
        """)


