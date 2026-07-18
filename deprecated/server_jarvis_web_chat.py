import streamlit as st
import time

st.set_page_config(
    page_title="JARVIS",
    page_icon="",
    layout="centered")

st.title("Jarvis - interface v 1.0")
st.markdown("---")

st.sidebar.header("Sistem Status")



if "messages" not in st.session_state :
    st.session_state.messages = []

for message in st.session_state.messages :
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Scrivi qui..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Ricevuto, sto analizzando..."
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})