import streamlit as st
import pymongo
import random
import time
from pymongo import MongoClient
import pages.utils.rag_utils as rag
from streamlit.runtime.scriptrunner import get_script_run_ctx


# Name of the database -- Change if needed or leave as is
DB_NAME = "mongodb_rag_lab"
# Name of the collection -- Change if needed or leave as is
COLLECTION_NAME = "knowledge_base"
# Name of the vector search index -- Change if needed or leave as is
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

    
ctx = get_script_run_ctx()
session_id = ctx.session_id

def config():
    st.set_page_config(
        layout="wide",
        page_title="Mr Beans | Chat",
        page_icon="💬"
    )

    # Customize page title
    st.title("Beans Guru 💬")
    
    popover = st.popover("Filter topics")
    mcmc = popover.checkbox("MCMC Regulations", True)
    packaging = popover.checkbox("Packaging Coffee Beans", True)
    shipping = popover.checkbox("Shipping Coffee Beans", True)

    st.info(
        "Package your beans better with MCMC Regulations"
    )



@st.cache_resource
def init_connection():
    return pymongo.MongoClient(st.secrets["mongo"]["host"], appname="devrel.workshop.rag")


def render_view():
    mongodb_client = init_connection()

    
    access_key = st.secrets["aws"]["access_key"]
    secret = st.secrets["aws"]["secret_key"]
    
    if "messages" not in st.session_state:
        st.session_state.messages = []


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask Beans Guru"):
        st.session_state.messages.append({"role": "user", "name": "You", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = rag.generate_answer(session_id, prompt)
            # response = st.write_stream(response)
            st.markdown(response)  
        st.session_state.messages.append({"role": "assistant", "name": "AWS Guru", "content": response})


if __name__ == "__main__":
    config()
    render_view()