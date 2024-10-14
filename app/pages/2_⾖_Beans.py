import streamlit as st
from inference_sdk import InferenceHTTPClient

def analyze_image(image, result_area):
    CLIENT = InferenceHTTPClient(
        api_url="https://classify.roboflow.com",
        api_key=st.secrets["roboflow"]["api_key"]
    )
    
    print(image)

    result = CLIENT.infer(image, model_id="coffebeans-5p6py/6")
    result_area.code(result)


def config():
    st.set_page_config(
        layout="wide",
        page_title="Mr Beans | Beans",
        page_icon="☕️"
    )

    # Customize page title
    st.title("Beans Analyzer ☕️")

    st.info(
        "Beans analyzer with Classification & Validity"
    )


def render_view():
    left, right = st.columns(2)
    uploaded = left.file_uploader(
        "Upload beans",
        type = ['png', 'jpg', 'jpeg', 'webp']
    )
    enable = right.checkbox("Enable camera")
    picture = right.camera_input("Take the beans", disabled = not enable)
    
    new_left, new_right = st.columns(2)
    if uploaded:
        new_left.image(uploaded)
        analyze_image(uploaded, new_right)
    elif picture:
        new_left.image(picture)
        result - analyze_image(picture, new_right)



if __name__ == "__main__":
    config()
    render_view()