"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import buildchain as buildchain
from langchain.callbacks import get_openai_callback
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

import prompt

class CAMELAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model(messages)
        self.update_messages(output_message)

        return output_message

st.set_page_config(page_title="FaAI Demo", page_icon=":robot:")
st.header("FaAI Demo")
assistant_role_name = "律师"
user_role_name = "咨询客户"
word_limit = 50


def create_task(assistant_role_name,user_role_name,task):
    task_specifier_sys_msg = SystemMessage(content="你可以使任务更加具体。")
    task_specifier_prompt = """以下是 {assistant_role_name} 将帮助 {user_role_name} 完成的任务：{task}。请说得更具体一些要根据实际的法律依据。
    请在 {word_limit} 个字或更少的时间内回复指定的任务。 不要添加任何其他东西。"""
    task_specifier_template = HumanMessagePromptTemplate.from_template(
        template=task_specifier_prompt
    )
    task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(temperature=1.0))
    task_specifier_msg = task_specifier_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
        word_limit=word_limit,
    )[0]
    specified_task_msg = task_specify_agent.step(task_specifier_msg)
    return specified_task_msg.content





def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(
        template=prompt.assistant_inception_prompt
    )
    assistant_sys_msg = assistant_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(
        template=prompt.user_inception_prompt
    )
    user_sys_msg = user_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    return assistant_sys_msg, user_sys_msg





if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if 'task_content' not in st.session_state:
    st.session_state['task_content'] =''


with st.sidebar:
    task= st.text_input("设定任务：", key="task", placeholder="点击输入")
    if task !='' and  st.session_state['task_content'] =='' :
        task_constent=create_task(assistant_role_name=assistant_role_name,user_role_name=user_role_name,task=task)
        st.session_state.generated.append(task_constent)
        st.session_state['task_content'] = task_constent


with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([4, 1])
    user_input = a.text_input(
        label="Your message:", placeholder="请输入", label_visibility="collapsed"
    )
    b.form_submit_button("Send", use_container_width=True)


assistant_sys_msg, user_sys_msg = get_sys_msgs(
    assistant_role_name, user_role_name, st.session_state["task_content"]
)
assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

# Reset agents
assistant_agent.reset()
user_agent.reset()


print(f"原任务prompt:\n{st.session_state['task']}\n")
print(f"指定任务 prompt:\n{st.session_state['task_content']}\n")

if user_input:

    assistant_msg = HumanMessage( content=(
        f"{user_input}. "
    ))
    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content)
    print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")
    assistant_ai_msg = assistant_agent.step(user_msg)
    print(f"AI assistant_ai_msg ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
    assistant_msg = HumanMessage(content=assistant_ai_msg.content)
    print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
    st.session_state.past.append(user_input)
    st.session_state.generated.append(assistant_msg.content)
if st.session_state["generated"]:
    # print(st.session_state["generated"])
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        if len(st.session_state["past"]) >0 :
            message(st.session_state["past"][i-1], is_user=True, key=str(i-1) + "_user")




