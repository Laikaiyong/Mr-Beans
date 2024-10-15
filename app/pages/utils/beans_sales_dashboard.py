import json
import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px

cwd = os.path.dirname(__file__)

def load_data():
    df1 = pd.read_csv(cwd + "/data/sales/coffeebeansales_0.csv")
    df2 = pd.read_csv(cwd + "/data/sales/coffeebeansales_2.csv")
    coffee_bean_sales_df = pd.merge(df1, df2, on='Product ID', how='outer')
    return coffee_bean_sales_df

df = load_data()
# print(df.head())

def render_coffee_type_chart():
    data = df.groupby("Coffee Type_y")["Quantity"].sum().reset_index()
    fig = px.bar(data, x="Coffee Type_y", y="Quantity", title="Total Quantity Sold by Coffee Type")
    st.plotly_chart(fig)

def roast_type_profit_chart():
    data = df.groupby("Roast Type_y")["Profit"].sum().reset_index()
    fig = px.pie(data, values="Profit", names="Roast Type_y", title="Profit Distribution by Roast Type")
    st.plotly_chart(fig)

def price_profit_scatter():
    fig = px.scatter(df, x="Price per 100g", y="Profit", color="Coffee Type_y", hover_data=["Product ID"])
    fig.update_layout(title="Price vs Profit by Coffee Type")
    st.plotly_chart(fig)

def cumulative_sales_chart():
    df_sorted = df.sort_values("Customer ID")
    df_sorted["CumulativeSales"] = df_sorted["Quantity"].cumsum()
    fig = px.line(df_sorted, x="Customer ID", y="CumulativeSales", title="Cumulative Sales Over Time")
    st.plotly_chart(fig)