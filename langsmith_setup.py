"""
LangSmith stup and Observability
Production monitoring for LangChain/LangGraph
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from dotenv import load_dotenv

load_dotenv()

@traceable(name="basic chaining")
def demo_basic_training():
    
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash"
    )
    
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in one sentence"
    )
    
    chain = prompt | llm | StrOutputParser()
    
    print("Basic Training Demo:\n")
    print("Running chain with Langsmith tracing enabled")
    
    result = chain.invoke(
        {"topic" : "machine learning"}
    )
    
    print(f"Result: {result}")
    
    print("\n Check LangSmith Dashboard for trace details")

@traceable(
    name="named_run_demo",
    tags=["production", "summarization"]
)
def demo_named_run():
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash"
    )
    
    prompt = ChatPromptTemplate.from_template(
        "Summarize: {text}"
    )
    
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke(
        {
            "text" : "LangSmith provides observability for LLM applications"
        }
    )
    
    print("\n Named run demos")
    print(f"Result: {result}")
    
    
@traceable(
    name="trace_with_metadata_demo",
    tags=["metadata", "filtering"]
)
def demo_trace_with_metadata(
    user_id: str,
    request_type: str
):
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash"
    )
    
    result = llm.invoke(
        f"Hello from user {user_id}. "
        f"Request type {request_type}. "
    )
    
    return result.content

if __name__ == "__main__":
    demo_basic_training()
    demo_named_run()
    
    result = demo_trace_with_metadata(
        user_id = "user_123",
        request_type = "greetings"
    )
    
    print("\n MetaData Demo")
    print(result)
