from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    CombinedMemory,
    ConversationSummaryMemory,
)
import logging as log
import os

os.environ["OPENAI_API_KEY"]


conv_memory = ConversationBufferMemory(
    memory_key="chat_history_lines", input_key="input"
)
summary_memory = ConversationSummaryMemory(llm=OpenAI(temperature=0), input_key="input")
memory = CombinedMemory(memories=[conv_memory, summary_memory])
template = """
我希望你能作为我的法律顾问。我将描述一个法律情况，你将提供如何处理的建议。
如果你觉的我描述的信息不够全面可以向我提问1-5个问题后生成你的建议。
其中需要根据基本案情、引用法律条款等几个方面生成

    Summary of conversation:
    {history}
    Current conversation:
    {chat_history_lines}
    Conversation:
    Human: {input}
    AI:"""
PROMPT = PromptTemplate(
    input_variables=["history", "input", "chat_history_lines"],
    template=template,
)


def load_chain():
    log.info("开始构建chain")
    llm = OpenAI(temperature=0)
    conversation = ConversationChain(
        llm=llm, verbose=True, memory=memory, prompt=PROMPT
    )
    return conversation



