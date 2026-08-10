from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-001"
)

#================
# BASIC EMBEDDING
#================
 
def basic_embeddings():
    
    #Single Text
    text = "What is Machine Learning?"
    
    #Convert text -> Embedding Vector
    embedding = embedding_model.embed_query(text)
    
    #Check vector dimensions
    print(f"Vector Dimensions: {len(embedding)}")
    
    #Look at the first 5 values of the vector
    print(f"First 5 values of the vector: {embedding[:5]}")
    
    #Check vector magnitude (norm)
    print(
        f"Vector norm: {np.linalg.norm(embedding):.4f}"
    )
    
    
#====================
# BATCH EMBEDDINGS
#=====================

def batch_embeddings():
    
    texts = [
        "What is Machine Learning",
        "Explain the concept of overfitting in ML",
        "How does a neural network work?",
    ]
    
    #Convert Multiple Texts -> Multiple Vectors
    embeddings = embedding_model.embed_documents(texts)
    
    for i , embedding in enumerate(embeddings):
        print(
            f"Text {i+1} - "
            f"Vector dimensions: {len(embedding)}"
        )
        
        print(
            
            f"Text {i+1} - "
            f"First 5 values: {embedding[:5]}"
        )
        
        print(
            f"Text {i + 1} - "
            f"Vector norm: "
            f"{np.linalg.norm(embedding):.4f}"
        )

#=================
#SIMILARITY SEARCH
#=================

def similarity_search():
    
    #Documents we want to search

    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    # User's search query
    query = "What programming languages exist?"
    
    #Embedd all the documents
    
    doc_vectors = embedding_model.embed_documents(docs)
    
    #Embedd the user's query
    
    query_vector = embedding_model.embed_query(query)
    
    #Calculate cosine similarity (Meaning / context kitna same h)
    
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1)
            * np.linalg.norm(vec2)
        )
        
    #Compare the query vector with every document vector
    
    similarities = [
        cosine_similarity(query_vector, doc_vector)
        for doc_vector in doc_vectors
    ]
    
    
    #RANK THE DOCUMENTS BASED ON THE SIMILARITY SCORE
    
    ranked_docs = sorted(
        zip(docs, similarities),
        key = lambda x: x[1],
        reverse = True
    )
    
    #DISPLAY RESULTS
    
    print(f"\nQuery: {query}\n")

    print("Ranked by similarity:")

    for doc, score in ranked_docs:

        print(
            f"  {score:.4f}: {doc}"
        )
        
        
#RUN THE EXPERIMENT
if __name__ == "__main__":

    print("\n===== BASIC EMBEDDINGS =====")
    basic_embeddings()

    print("\n===== BATCH EMBEDDINGS =====")
    batch_embeddings()

    print("\n===== SIMILARITY SEARCH =====")
    similarity_search()