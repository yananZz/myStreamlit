"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import buildchain as buildchain
from langchain.callbacks import get_openai_callback


st.set_page_config(page_title="FaAI Demo", page_icon=":robot:")
st.header("FaAI Demo")


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


chain = buildchain.load_chain()

if user_input:
    output = chain.run(input=user_input)
    # print(output)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    # print(st.session_state["generated"])
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
