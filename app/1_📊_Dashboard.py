import textwrap

import streamlit as st

from charts import ST_DEMOS
import pages.utils.price_dashboard_utils as priceut

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
    # for chart in ST_DEMOS:
    #     demo, url = (
    #         ST_DEMOS[chart]
    #     )
        
    #     demo()


if __name__ == "__main__":
    config()
    render_view()