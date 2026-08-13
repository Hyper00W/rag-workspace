from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import InMemoryStore

from langchain_chroma import Chroma

load_dotenv()

DOCUMENTS = [
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. It is widely used in web development, data science, artificial intelligence, and automation.",
        metadata={"topic": "programming"},
    ),

    Document(
        page_content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Frameworks like React, Vue, and Angular are commonly used to build interactive web applications.",
        metadata={"topic": "programming"},
    ),

    Document(
        page_content="Machine learning is a subset of artificial intelligence that enables systems to learn from data. Popular frameworks include TensorFlow, PyTorch, and scikit-learn.",
        metadata={"topic": "ai"},
    ),

    Document(
        page_content="LangChain is a framework for building applications powered by large language models. It provides tools for prompts, chains, agents, and memory.",
        metadata={"topic": "llm"},
    ),

    Document(
        page_content="LangGraph is a library for building stateful and complex applications with language models. It supports state management, cycles, loops, human-in-the-loop workflows, and persistence.",
        metadata={"topic": "llm"},
    ),

    Document(
        page_content="Docker is a platform for containerizing applications. Containers package code and dependencies together so applications can run consistently across different environments.",
        metadata={"topic": "devops"},
    ),

    Document(
        page_content="PostgreSQL is an open-source relational database. It supports JSON data, full-text search, and extensions such as pgvector for vector similarity search.",
        metadata={"topic": "database"},
    ),

    Document(
        page_content="Vector databases such as Chroma, Pinecone, and Qdrant are designed for storing and searching embeddings. They are commonly used for semantic search and RAG applications.",
        metadata={"topic": "database"},
    ),
]

embeddings = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-001"
)

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap = 100
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20
)


vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name="parent_document_demo"
)

store = InMemoryStore()


retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(DOCUMENTS)

vector_retriever = vectorstore.as_retriever(
    search_kwargs = {"k" : 2}
)

bm25_retriever = BM25Retriever.from_documents(
    DOCUMENTS
)
bm25_retriever.k = 2

ensemble_retriever = EnsembleRetriever(
    retrievers=[
        vector_retriever,
        bm25_retriever
    ],
    weights=[0.6,0.4]
)

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm = llm
)

compressor = LLMChainExtractor.from_llm(
    llm
)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)

query = "What tools I can use to build AI applications"

results = retriever.invoke(query)

print(f"\nQuery: {query}")

for i, doc in enumerate(results):

    print(f"\n--- Result {i + 1} ---")
    print(f"Topic: {doc.metadata.get('topic')}")
    print(doc.page_content)
    
   