import streamlit as st
import time

st.set_page_config(
    page_title="JARVIS",
    page_icon="",
    layout="centered")

st.title("Jarvis - interface v 1.0")
st.markdown("---")

st.chat_message(message["role"])