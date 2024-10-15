import textwrap

import streamlit as st

from charts import ST_DEMOS
import pages.utils.price_dashboard_utils as priceut
import pages.utils.beans_sales_dashboard as beansut

def config():
    st.set_page_config(
        layout="wide",
        page_title="Mr Beans | Home",
        page_icon="🎯"
    )

    # Customize page title
    st.title("Mr Beans 🫘")

    st.info(
        "Understand how your business progress here"
    )


def render_view():
    priceut.render_price_chart()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Coffee Type", "Roast Type", "Price vs Profit", "Cumulative Sales"])
    
    with tab1:
        beansut.render_coffee_type_chart()
    with tab2:
        beansut.roast_type_profit_chart()
    with tab3:
        beansut.price_profit_scatter()
    with tab4:
        beansut.cumulative_sales_chart()
    # for chart in ST_DEMOS:
    #     demo, url = (
    #         ST_DEMOS[chart]
    #     )
        
    #     demo()


if __name__ == "__main__":
    config()
    render_view()