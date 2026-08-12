from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_chroma import Chroma
from langsmith import traceable

load_dotenv()

SAMPLE_TEXT = """
# Machine Learning

Machine learning is a branch of artificial intelligence that
allows computers to learn patterns from data and make predictions
without being explicitly programmed for every task.

## Supervised Learning

Supervised learning uses labeled training data. Each training
example contains an input and a known output. The model learns
the relationship between inputs and outputs and uses that
relationship to make predictions on new data.

### Regression

Regression is used when the target variable is continuous.
Common examples include predicting house prices, temperature,
sales revenue, and vehicle counts.

Popular regression algorithms include Linear Regression,
Polynomial Regression, Decision Tree Regression, and
Random Forest Regression.

### Classification

Classification is used when the target variable belongs to
a specific category or class. Examples include spam detection,
disease classification, and image recognition.

Common classification algorithms include Logistic Regression,
Decision Trees, Random Forests, Support Vector Machines,
and Neural Networks.

## Unsupervised Learning

Unsupervised learning works with data that does not contain
predefined labels. The goal is to discover hidden patterns,
groups, or structures inside the dataset.

### Clustering

Clustering groups similar data points together.

K-Means is one of the most commonly used clustering algorithms.
Other approaches include Hierarchical Clustering and DBSCAN.

### Dimensionality Reduction

Dimensionality reduction reduces the number of features while
trying to preserve important information.

Principal Component Analysis, commonly called PCA, is a popular
dimensionality reduction technique.

## Model Evaluation

Machine learning models must be evaluated before being used
in real applications.

For regression problems, common metrics include Mean Absolute
Error, Mean Squared Error, Root Mean Squared Error, and R-squared.

For classification problems, common metrics include Accuracy,
Precision, Recall, F1 Score, and ROC-AUC.

Choosing the correct evaluation metric depends on the problem
and the consequences of incorrect predictions.

## Applications

Machine learning is used in many real-world applications.

Recommendation systems use machine learning to suggest products,
movies, music, and other content.

Fraud detection systems identify suspicious financial
transactions by learning patterns from historical data.

Healthcare systems can use machine learning for medical image
analysis, risk prediction, and patient monitoring.

Autonomous vehicles use machine learning together with computer
vision and sensor data to understand their environment.

## Important Considerations

A machine learning model can perform well on training data but
poorly on unseen data. This problem is known as overfitting.

Regularization, cross-validation, appropriate feature selection,
and sufficient training data can help reduce overfitting.

The quality of the training data is also extremely important.
Incorrect, biased, or incomplete data can lead to unreliable
predictions even when the underlying algorithm is powerful.
""".strip()

#CONVERT OUR DOCUMENT INTO LANGCHAIN DOCUMENT
DOCUMENTS = [
    Document(
        page_content=SAMPLE_TEXT,
        metadata = {
            "source" : "sample_machine_learning_document",
            "type" : "learning_document",
        },
    )
]

EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

LLM = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)


#====================
# RECURSIVE CHUNKING
#====================

def create_recursive_chunks():
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        separators= [
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )
    
    chunks = splitter.split_documents(DOCUMENTS)
    
    print("\n" + "=" * 60)
    print("RECURSIVE CHUNKING")
    print("=" * 60)

    print(f"Number of chunks: {len(chunks)}")

    print(
        "Chunk sizes:",
        [len(chunk.page_content) for chunk in chunks]
    )

    return chunks

#===================
# SEMANTIC CHUNKING
#===================

def create_semantic_chunks():
    
    splitter = SemanticChunker(
        EMBEDDINGS,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    
    chunks = splitter.split_documents(DOCUMENTS)
    
    print("\n" + "=" * 60)
    print("SEMANTIC CHUNKING")
    print("=" * 60)

    print(f"Number of chunks: {len(chunks)}")

    print(
        "Chunk sizes:",
        [len(chunk.page_content) for chunk in chunks]
    )

    return chunks

def show_chunks(chunks, name):
    print("\n" + "=" * 60)
    print(f"{name.upper()} - CHUNKS")
    print("=" * 60)
    
    for i , chunk in enumerate(chunks):
        
        print(f"\n--- Chunks {i+1} ---")
        print(chunk.page_content[:400])
        


#===========================
#CREATE VECTOR STORES
#===========================

def create_vector_stores(
    recursive_chunks,
    semantic_chunks
):
    recursive_store = Chroma.from_documents(
        documents=recursive_chunks,
        embedding=EMBEDDINGS,
        collection_name="recursive_ml_chunks"
    )
    
    semantic_store = Chroma.from_documents(
        documents=semantic_chunks,
        embedding=EMBEDDINGS,
        collection_name="semantic_ml_chunks"
    )
    
    print("\n" + "=" * 60)
    print("VECTOR STORES")
    print("=" * 60)

    print("Recursive chunks stored in Chroma")
    print("Semantic chunks stored in Chroma")

    return recursive_store, semantic_store
# ============================================================
# TEST RETRIEVAL
# ============================================================

@traceable(name="test_retrieval")
def test_retrieval(
    query,
    vectorstore,
    name,
    k=3
):

    print("\n" + "=" * 60)
    print(f"{name.upper()} RETRIEVAL")
    print("=" * 60)

    print(f"Query: {query}")

    results = vectorstore.similarity_search(
        query,
        k=k
    )

    for i, document in enumerate(results):

        print(f"\n--- Result {i + 1} ---")

        print(
            f"Source: "
            f"{document.metadata.get('source')}"
        )

        print(
            document.page_content[:500]
        )

    return results


# ============================================================
# GENERATE ANSWER
# ============================================================

@traceable(name="generate_answer")
def generate_answer(
    query,
    documents,
    strategy
):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
    Answer the question using ONLY the context below.

    Retrieval strategy:
    {strategy}

    Context:
    {context}

    Question:
    {query}

    If the answer is not present in the context,
    say that the information is not available.

    Answer:
    """

    response = LLM.invoke(prompt)

    return response.content


# ============================================================
# COMPARE BOTH APPROACHES
# ============================================================

def compare(query, recursive_store, semantic_store):

    recursive_results = test_retrieval(
        query,
        recursive_store,
        "Recursive Chunking"
    )

    semantic_results = test_retrieval(
        query,
        semantic_store,
        "Semantic Chunking"
    )

    recursive_answer = generate_answer(
        query,
        recursive_results,
        "Recursive Chunking"
    )

    semantic_answer = generate_answer(
        query,
        semantic_results,
        "Semantic Chunking"
    )

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print("\nQUERY:")
    print(query)

    print("\n--- RECURSIVE ANSWER ---")
    print(recursive_answer)

    print("\n--- SEMANTIC ANSWER ---")
    print(semantic_answer)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    recursive_chunks = create_recursive_chunks()

    semantic_chunks = create_semantic_chunks()

    show_chunks(
        recursive_chunks,
        "Recursive"
    )

    show_chunks(
        semantic_chunks,
        "Semantic"
    )

    recursive_store, semantic_store = create_vector_stores(
        recursive_chunks,
        semantic_chunks
    )

    queries = [
        "What is the difference between regression and classification?",
        "How can overfitting be reduced?",
        "What are some real-world applications of machine learning?"
    ]

    for query in queries:

        compare(
            query,
            recursive_store,
            semantic_store
        )