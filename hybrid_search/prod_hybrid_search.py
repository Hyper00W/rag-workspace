from rank_bm25 import BM25Okapi
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()


#================
# Embedding Model
#================

embedding_model = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-001"
)


#================
# BM25
#================

documents = [
    "Python is a programming language",
    "Javascript is used for web development",
    "Machine Learning enables AI applications",
    "Deep Learning uses neural networks",
    "Cats are popular pets",
]

#BM25 works with tokens(words) so we tokenize our documents
tokenized_documents = [
    document.lower().split()
    for document in documents
]

#Create the BM25 index (prepares documents ny checking for various stuff like length , which word comes how many times, etc)
bm25 = BM25Okapi(tokenized_documents)

#User's Query
query = "Programming languages"

#Tokenize the query too
tokenized_query = query.lower().split()

#Search
bm25_scores = bm25.get_scores(tokenized_query)

#Display Scores
for document , score in zip(documents , bm25_scores):
    print(f"{score:.4f} -> {document}")
    
    
    

#====================
# VECTOR SEARCH
#====================

#Convert Documents into Vectors
documents_vector = embedding_model.embed_documents(documents)

#Convert Query into Vector
query_vector = embedding_model.embed_query(query)

#Cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

#Compare query with evry document
vector_scores = [
    cosine_similarity(query_vector, document_vector)
    for document_vector in documents_vector
]


print("\n====Vector Search Results ======")

for document, score in zip(documents, vector_scores):
    print(f"{score:.4f} -> {document}")
    
    
def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    
    #Avoid division by 0 ,,,,, agr saari values same h to sbko 1 dedo
    if max_score == min_score:
        return [1] * len(scores)
    
    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]

normalized_bm25 = normalize_scores(bm25_scores)
normalized_vector = normalize_scores(vector_scores)

alpha = 0.5

#Creating hybrid scores
hybrid_scores = [
    alpha * vector_score + (1 - alpha) * bm25_score
    for vector_score , bm25_score in zip(normalized_vector, normalized_bm25)
]


ranked_results = sorted(
    zip(documents, hybrid_scores),
    key = lambda x: x[1],
    reverse =True
)


print("======HYBIRD SEARCH=========")
for document , score in ranked_results:
    print(f"{score:.4f} -> {document}")