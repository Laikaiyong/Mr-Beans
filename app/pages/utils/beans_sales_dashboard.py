import json
import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_echarts import st_echarts


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
    # Prepare data for ECharts
    categories = data["Coffee Type_y"].tolist()
    quantities = data["Quantity"].tolist()
    options = {
        "xAxis": {
            "type": "category",
            "data": categories,
        },
        "yAxis": {"type": "value"},
        "series": [
            {
                "data": quantities,
                "type": "bar",
            }
        ],
        "title": {
            "text": "Total Quantity Sold by Coffee Type",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {
                "type": "shadow"
            }
        }
    }
    left, right = st.columns([4, 1])
    with left:
        st_echarts(options=options, height="500px")
    with right.expander("Learn more", icon="ℹ"):
        st.markdown("""
            - **Ara - Arabica**: Excels in flavor, low bitterness
            - **Exc - Excelsa**: Tart, fruity, and aromatic
            - **Lib - Liberica**: Unique, bold, woody flavor
            - **Rob - Robusta**: Strong, bitter, high caffeine
        """)

def roast_type_profit_chart():
    data = df.groupby("Roast Type_y")["Profit"].sum().reset_index()
    data["Profit"] = data["Profit"].apply(lambda x: float("{:.2f}".format(x)))
    # Prepare data for ECharts
    echarts_data = [{"value": row["Profit"], "name": row["Roast Type_y"]} for _, row in data.iterrows()]
    option = {
        "title": {
            "text": "Profit Distribution by Roast Type",
            "left": "center"
        },
        "tooltip": {
            "trigger": "item",
            "formatter": "{a} <br/>{b}: ${c} ({d}%)"
        },
        "legend": {
            "orient": "vertical",
            "left": "left"
        },
        "toolbox": {
            "show": True,
            "feature": {
                "mark": {"show": True},
                "dataView": {"show": True, "readOnly": False},
                "restore": {"show": True},
                "saveAsImage": {"show": True}
            }
        },
        "series": [
            {
                "name": "Profit",
                "type": "pie",
                "radius": ["30%", "70%"],
                "center": ["50%", "50%"],
                "roseType": "area",
                "itemStyle": {
                    "borderRadius": 8
                },
                "data": echarts_data,
                "label": {
                    "show": True,
                    "formatter": "{b}: ${c}"
                },
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                }
            }
        ]
    }
    left, right = st.columns([4, 1])
    with left:

        st_echarts(options=option, height="500px")
    with right.expander("Learn more", icon="ℹ"):
        st.markdown("""
            - **M**: Medium roast
            - **D**: Dark Roast
            - **L**: Light Roast
        """)

def price_profit_scatter():
    # Get unique coffee types
    coffee_types = df["Coffee Type_y"].unique()

    # Prepare series data
    series_data = []
    for coffee_type in coffee_types:
        data = df[df["Coffee Type_y"] == coffee_type]
        series_data.append({
            "name": coffee_type,
            "type": "scatter",
            "data": data[["Price per 100g", "Profit", "Product ID"]].values.tolist(),
            "symbolSize": 10,
        })

    option = {
        "title": {
            "text": "Price vs Profit by Coffee Type",
            "left": "center"
        },
        "tooltip": {
            "trigger": "item",
            # "formatter": function (params) {
            #     return `
            #             {params.value[0]}<br/>
            #             {params.value[1]}<br/>
            #             {params.value[2]}`;
            # }
        },
        "legend": {
            "data": coffee_types.tolist(),
            "orient": "vertical",
            "right": 10,
            "top": 50
        },
        "xAxis": {
            "type": "value",
            "name": "Price per 100g ($)",
            "nameLocation": "middle",
            "nameGap": 30,
        },
        "yAxis": {
            "type": "value",
            "name": "Profit ($)",
            "nameLocation": "middle",
            "nameGap": 30,
        },
        "series": series_data
    }

    left, right = st.columns([4, 1])
    with left:

        st_echarts(options=option, height="600px")
    with right.expander("Learn more", icon="ℹ"):
        st.markdown("""
            - **Ara - Arabica**: Excels in flavor, low bitterness
            - **Exc - Excelsa**: Tart, fruity, and aromatic
            - **Lib - Liberica**: Unique, bold, woody flavor
            - **Rob - Robusta**: Strong, bitter, high caffeine
        """)

# def cumulative_sales_chart():
#     df_sorted = df.sort_values("Customer ID")
#     df_sorted["CumulativeSales"] = df_sorted["Quantity"].cumsum()
#     fig = px.line(df_sorted, x="Customer ID", y="CumulativeSales", title="Cumulative Sales Over Time")
#     st.plotly_chart(fig)