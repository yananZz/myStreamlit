"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import buildchain as buildchain


st.set_page_config(page_title="LawAI Demo", page_icon=":robot:")
st.header("LawAI Demo")

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key",
        key="chatbot_api_key",
    )

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([4, 1])
    user_input = a.text_input(
        label="Your message:", placeholder="请输入", label_visibility="collapsed"
    )
    b.form_submit_button("Send", use_container_width=True)

# user_input = get_text()
if user_input and not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")

if openai_api_key:
    chain = buildchain.load_chain(openai_api_key)

if user_input and openai_api_key:
    output = chain.predict(human_input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
