from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

import os

os.environ["OPENAI_API_KEY"]

# This is an LLMChain to write a synopsis given a title of a play.
llm = OpenAI(temperature=0)
template = """你的角色是一名专业的律师。我将描述一个案例。
案件描述:
{input}
我需要你根据以下几点关键点分析案情,每个关键点都需要分析。如果哪些关键点不明确的地方可以向我提问补充信息
1、基本信息
2、关键词
3、案情摘要
4、争议焦点
5、裁判要点
6、涉及的法律条款
"""
prompt_template = PromptTemplate(input_variables=["input"], template=template,)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template,output_key="synopsis")

# This is an LLMChain to write a review of a play given a synopsis.
template = """
{synopsis}
根据关键点列表的方式输出结果。最后再进行总结和违反的法律条款
"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)


def buildChain():
    # This is the overall chain where we run these two chains in sequence.
    chain= SimpleSequentialChain(chains=[synopsis_chain, review_chain],input_key="input", verbose=True,strip_outputs=True)
    return chain