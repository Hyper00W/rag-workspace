# So this is a basic rag pipeline with gemini plus chromadb
# Flow will be like this

# Knowledge Base
#       ↓
# Chunking
#       ↓
# Gemini Embeddings
#       ↓
# ChromaDB
#       ↓
# Retriever / Similarity Search
#       ↓
# Relevant Chunks
#       ↓
# Context + User Question
#       ↓
# Gemini LLM
#       ↓
# Final Answer

#Imports

import tempfile

from dotenv import load_dotenv

# Gemini LLM + Gemini Embeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

# LangChain document object
from langchain_core.documents import Document

# Prompt template
from langchain_core.prompts import ChatPromptTemplate

# Allows the original question to pass through unchanged
from langchain_core.runnables import RunnablePassthrough

# Converts LLM response into a normal string
from langchain_core.output_parsers import StrOutputParser

# Text chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma vector database
from langchain_chroma import Chroma


load_dotenv()

#Creating gemini embedding model

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

#This is not the LLM its job is only to embedd the text into vector form

#Now creating the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature=0.2
)

#Embedding and LLMs models boht have different jobs
#EMbeddding models turn text -> vector , it is used for retrieval and search
#LLM are used after we have the context from the knowledge base and user's questions for generation of answer/result


#Sample Knwoledge base

KNOWLEDGE_BASE = """
# LangChain Framework

LangChain is a framework for developing applications
powered by language models. It was created by Harrison
Chase in October 2022.

## Core Components

1. Models:
LangChain supports various LLM providers including
OpenAI, Anthropic, and local models.

2. Prompts:
Templates for structuring inputs to language models.

3. Chains:
Sequences of calls to models and other components.

4. Agents:
Systems that use LLMs to determine which actions to take.

5. Memory:
Components for persisting state between chain or agent calls.

## LangGraph

LangGraph is a library for building stateful,
multi-actor applications.

Key features:

- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free.

LangSmith is the observability platform and has
free and paid plans.

## Getting Started

Install LangChain using pip.

Create your first chain in under 10 lines of code.
"""

#In a real RAG application it can come from:
#PDF , DOCX, WEBSITE , DATABASES , etc


#=================================================
#Function to create our knowledge Base
#=================================================

def create_kb():
    """
    Create a Chroma vector store from our knowledge base.

    Steps:
    1. Split large document into chunks
    2. Convert chunks into embeddings
    3. Store embeddings + documents in ChromaDB
    """
    
    #---------------------------
    #Create Text Splitter
    #---------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    #---------------------------
    #Create LangChain Document
    #---------------------------
    doc = Document(
        page_content=KNOWLEDGE_BASE,
        metadata={
            "source": "langchain_knowledge_base.md"
        }
    )
    #This document contain the actual text along with the metadata
    
    
    #-----------------------
    #Split Documents into Chunks
    #-----------------------
    chunks = splitter.split_documents([doc])
    
    #Before 1 large document
    #Now
    #Document(chunk 1),
    #Document(chunk 2),
    #Document(chunk 3),
    #Each chunk has its own embedding...
    
    
    #--------------------------
    #Create Chroma Vector Store
    #--------------------------
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=tempfile.mkdtemp(),
    )
    # IMPORTANT:
    #
    # Chroma now takes each chunk and uses the
    # Gemini embedding model to create vectors.
    #
    # Example:
    #
    # Chunk 1
    #    ↓
    # Gemini Embedding Model
    #    ↓
    # Vector 1
    #    ↓
    # Chroma
    #
    # Chunk 2
    #    ↓
    # Gemini Embedding Model
    #    ↓
    # Vector 2
    #    ↓
    # Chroma
    # Now Chroma has a searchable vector database.
    return vector_store

#=======================
#BASIC RAG FUNCTION
#=======================

def demo_basic_rag():
    #Create a vector store
    vector_store = create_kb()
    
    #Create a Retreiver
    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs={"k": 2}
    )
    # Retriever's job:
    #
    # User Question
    #      ↓
    # Create query embedding
    #      ↓
    # Compare with Chroma's stored vectors
    #      ↓
    # Find most similar chunks
    #
    # k=2 means:
    #
    # Return the top 2 most relevant chunks.

    #=====================
    #RAG PROMPT
    #=====================
    
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the following context:

        {context}

        Question: {question}

        Answer:

        Make sure to answer in a concise manner,
        and if you don't know the answer,
        just say "I don't know."
        """   
    )
    # IMPORTANT:
    #
    # {context}
    #     ↓
    # Retrieved chunks will be inserted here.
    #
    # {question}
    #     ↓
    # User's original question will be inserted here.
    #
    # The LLM therefore receives:
    #
    # Context + Question
    #
    # and generates the answer.
    
    #===============================
    #Format Recived DOcumetns
    #================================
    
    def format_docs(docs):
        return "\n\n".join(
            [doc.page_content for doc in docs]
        )
    # Retriever returns Document objects.
    #
    # Example:
    #
    # [
    #     Document(page_content="LangChain is..."),
    #     Document(page_content="LangGraph is...")
    # ]
    #
    # We only want the actual text.
    #
    # So we extract:
    #
    # doc.page_content
    #
    # and combine them into one string.
    #
    # Result:
    #
    # "LangChain is...
    #
    #  LangGraph is..."
    
    # ========================================================
    # 9. CREATE THE RAG CHAIN
    # ========================================================
    
    rag_chain = (
        {
            #Step1 -- Retrieve Context
            
            "context": retriever | format_docs,
            
            #Question directly goes forward without any modification
            "question": RunnablePassthrough()
        }
        
        #INSERT CONTEXT PLUS QUESTION INTO THE PROMPT
        
        | prompt

        #SEND PROMPT TO GEMINI
        
        |llm

        #CONVERT GEMINI RESPONSE TO STIRNG
        
        |StrOutputParser()
    )
    
    #=====================
    #Test Questions
    #======================
    
    questions = [
        "What is Langchain?",
        "Who created Langchain?",
        "What is LangGraph used for?",
    ]
    
    #==========================
    #RUN THE RAG CHAIN
    #-==========================
    
    print("RAG Chain Running")
    
    for q in questions:
        # invoke() runs the ENTIRE RAG pipeline.
        #
        # Question
        #    ↓
        # Retriever
        #    ↓
        # Relevant chunks
        #    ↓
        # Format chunks
        #    ↓
        # Context + Question
        #    ↓
        # Prompt
        #    ↓
        # Gemini
        #    ↓
        # String output
        answer = rag_chain.invoke(q)
        
        print(f"Q: {q}")
        print(f"A: {answer}\n")
        
        #RUN THE PROGRAM
if __name__ == "__main__":
    demo_basic_rag()