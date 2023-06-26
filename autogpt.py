from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool


from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI


search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    WriteFileTool(),
    ReadFileTool(),
]

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
from langchain.memory.chat_message_histories import FileChatMessageHistory
agent = AutoGPT.from_llm_and_tools(
    ai_name="FaAI",
    ai_role="LawAssistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever(),
    chat_history_memory=FileChatMessageHistory("chat_history.txt")
)
# Set verbose to be true
agent.chain.verbose = True
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    agent.run(["今天北京天气报告,结果请翻译成中文"])
    print(cb)
