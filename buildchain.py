from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    CombinedMemory,
    ConversationSummaryMemory,
)

# def load_chain(openai_api_key):
#     llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
#     conv_memory = ConversationBufferMemory(memory_key="chat_history_lines", input_key="input")
#     summary_memory = ConversationSummaryMemory(
#             llm= llm, input_key="input",return_messages=True
#         )
#     memory = CombinedMemory(memories=[conv_memory, summary_memory])
#     # 这里这是当前角色为律师,只能以律师的口吻回答问题
#     template = """
#     LawAI是一个法律类的大型语言模型。
#     LawAI旨在帮助完成各种任务,从回答简单的问题到就各种主题提供深入的解释和讨论。作为一种语言模型,LawAI 能够根据接收到的输入生成类似人类的文本,使其能够进行听起来自然的对话,并提供与当前主题连贯且相关的响应。
#     LawAI 不断学习和改进,其能力也在不断发展。它能够处理和理解大量文本,并可以利用这些知识对各种问题提供准确且内容丰富的答案。
#     此外,LawAI 能够根据收到的输入生成自己的文本,使其能够参与讨论并就各种主题提供解释和描述。
#     总体而言,LawAI 是一款功能强大的工具,可以帮助完成各种法律任务,并提供有关各种主题的宝贵见解和信息。无论您需要解决特定问题的帮助还是只想就特定主题进行对话,助理都会随时为您提供帮助。

#     Summary of conversation:
#     {history}
#     Current conversation:
#     {chat_history_lines}
#     Conversation:
#     Human: {input}
#     AI:"""
#     prompt = PromptTemplate(
#         input_variables=["history", "input", "chat_history_lines"], template=template
#     )
#     chain = ConversationChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
#     return chain


def load_chain(openai_api_key):
    conv_memory = ConversationBufferMemory(
        memory_key="chat_history_lines", input_key="input"
    )

    summary_memory = ConversationSummaryMemory(llm=OpenAI(openai_api_key=openai_api_key), input_key="input")
    # Combined
    memory = CombinedMemory(memories=[conv_memory, summary_memory])
    _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Summary of conversation:
    {history}
    Current conversation:
    {chat_history_lines}
    Human: {input}
    AI:"""
    PROMPT = PromptTemplate(
                input_variables=["history", "input", "chat_history_lines"],
                template=_DEFAULT_TEMPLATE,
            )
    llm = OpenAI(temperature=0,openai_api_key=openai_api_key)
    conversation = ConversationChain(llm=llm, verbose=True, memory=memory, prompt=PROMPT)
    return conversation
