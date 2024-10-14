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

def predict_future_prices(model, scaler, df, end_date):
    last_date = df['DATE'].max()
    print(type(last_date))
    print(type(end_date))
    months_to_predict = (end_date - last_date).days / 30
    future_dates = pd.date_range(start=last_date, periods=months_to_predict)
    future_df = pd.DataFrame({'DATE': future_dates})
    future_df['days_since_start'] = (future_df['DATE'] - df['DATE'].min()).dt.days
    future_df['WPU026301'] = df['WPU026301'].iloc[-1]  # Assume last known value
    future_df['price_lag1'] = df['PCOFFOTMUSDM'].iloc[-1]
    future_df['price_lag2'] = df['PCOFFOTMUSDM'].iloc[-2]
    future_prices = []
    for _, row in future_df.iterrows():
        input_data = scaler.transform([[row['days_since_start'], row['WPU026301'], row['price_lag1'], row['price_lag2']]])
        prob = model.predict_proba(input_data)[0][1]  # Probability of price increase
        if prob > 0.5:
            price_change = row['price_lag1'] * 0.01  # Assume 1% increase
        else:
            price_change = -row['price_lag1'] * 0.01  # Assume 1% decrease
        new_price = row['price_lag1'] + price_change
        future_prices.append(new_price)
        # Update lags for next prediction
        row['price_lag2'] = row['price_lag1']
        row['price_lag1'] = new_price
    future_df['PCOFFOTMUSDM'] = future_prices
    return future_df


def render_price_chart():
    df = load_and_preprocess_data()
    X, y, scaler, df = prepare_data_for_model(df)
    model, accuracy = train_model(X, y)

    future_df = predict_future_prices(model, scaler, df, datetime(2024, 10, 1))

    combined_df = pd.concat([df, future_df])
    filtered_df = combined_df[combined_df['DATE'] <= pd.Timestamp(datetime(2024, 10, 1))]

    options = {
        "title": {"text": "Coffee Prices"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["PCOFFOTMUSDM", "WPU026301"]},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": list(filtered_df["DATE"].dt.strftime('%Y-%m-%d')),
        },
        "yAxis": {"type": "value"},
        "series": [
            {
                "name": "PCOFFOTMUSDM",
                "type": "line",
                "data": list(filtered_df["PCOFFOTMUSDM"]),
            },
            {
                "name": "WPU026301",
                "type": "line",
                "data": list(filtered_df["WPU026301"]),
            }
        ],
    }

    left, right = st.columns([4, 1])
    with left:
        st_echarts(options=options, height="400px")
    with right.expander("Learn more", icon="ℹ"):
        st.markdown("""
            - **PCOFFOTMUSDM**: Global price of Coffee, Other Mild Arabica
            - **WPU026301**: Producer Price Index by Commodity: Processed Foods and Feeds: Coffee (Whole Bean, Ground, and Instant)
            - Future prices are projected based on the trained model
        """)
