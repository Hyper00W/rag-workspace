import numpy as np

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# SEMANTIC CACHE
# ============================================================

class SemanticCache:

    def __init__(self, similarity_threshold=0.90):

        self.threshold = similarity_threshold

        self.embedder = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )

        self.cache = []

    def cosine_similarity(self, vec1, vec2):

        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) *
            np.linalg.norm(vec2)
        )

    def get(self, query):

        query_embedding = self.embedder.embed_query(query)

        best_score = -1
        best_response = None

        for item in self.cache:

            score = self.cosine_similarity(
                query_embedding,
                item["embedding"]
            )

            print(
                f"Similarity: {score:.4f} -> {item['query']}"
            )

            if score > best_score:
                best_score = score
                best_response = item["response"]

        if best_score >= self.threshold:

            print(f"CACHE HIT: {best_score:.4f}")

            return best_response

        print(f"CACHE MISS: {best_score:.4f}")

        return None

    def set(self, query, response):

        embedding = self.embedder.embed_query(query)

        self.cache.append({
            "query": query,
            "embedding": embedding,
            "response": response
        })


# ============================================================
# TOKEN BUDGET
# ============================================================

class TokenBudget:

    def __init__(self, max_tokens=4000):

        self.max_tokens = max_tokens

        self.usage = {
            "total_input": 0,
            "total_output": 0,
            "requests": 0
        }

    def estimate_tokens(self, text):

        return int(len(text.split()) * 1.3)

    def check_budget(self, text):

        tokens = self.estimate_tokens(text)

        return tokens <= self.max_tokens, tokens

    def record_usage(self, input_tokens, output_tokens):

        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def stats(self):

        total_tokens = (
            self.usage["total_input"] +
            self.usage["total_output"]
        )

        return {
            **self.usage,
            "total_tokens": total_tokens
        }


# ============================================================
# MODEL ROUTER
# ============================================================

class ModelRouter:

    def __init__(self):

        self.cheap_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash"
        )

        self.expensive_model = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash"
        )

        self.classifier = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash"
        )

        self.cache = SemanticCache(
            similarity_threshold=0.90
        )

        self.budget = TokenBudget(
            max_tokens=4000
        )

        self.cache_hits = 0
        self.cache_misses = 0


    def classify_complexity(self, query: str) -> str:

        prompt = f"""
        Classify this query as either 'simple' or 'complex'.

        Simple:
        - Basic facts
        - Short answers
        - Simple calculations

        Complex:
        - Analysis
        - Deep reasoning
        - Multi-step problems
        - Detailed explanations

        Query: {query}

        Respond with ONLY:
        simple
        or
        complex
        """

        response = self.classifier.invoke(prompt)

        print("CLASSIFIER:", response.content)

        return response.content.strip().lower()


    def invoke(self, query: str):

        # 1. Check token budget

        within_budget, tokens = self.budget.check_budget(query)

        if not within_budget:

            raise ValueError(
                f"Query exceeds token budget: "
                f"{tokens} > {self.budget.max_tokens}"
            )


        # 2. Check semantic cache

        cached_response = self.cache.get(query)

        if cached_response is not None:

            self.cache_hits += 1

            return cached_response, "CACHE", "cached"


        self.cache_misses += 1


        # 3. Classify query

        complexity = self.classify_complexity(query)


        # 4. Choose model

        if complexity == "simple":

            model = self.cheap_model

            model_name = "gemini-2.5-flash"

        else:

            model = self.expensive_model

            model_name = "gemini-3.5-flash"


        # 5. Generate answer

        response = model.invoke(query)

        answer = response.content


        # 6. Estimate output tokens

        output_tokens = self.budget.estimate_tokens(
            answer
        )


        # 7. Record usage

        self.budget.record_usage(
            tokens,
            output_tokens
        )


        # 8. Store in semantic cache

        self.cache.set(
            query,
            answer
        )


        return answer, model_name, complexity


    def stats(self):

        total = (
            self.cache_hits +
            self.cache_misses
        )

        hit_rate = (
            self.cache_hits / total
            if total > 0
            else 0
        )

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{hit_rate:.1%}",
            "token_usage": self.budget.stats()
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    router = ModelRouter()

    queries = [

        "What is Python?",

        "Tell me about the Python programming language.",

        "What is 2+2?",

        "Can you explain Python to me?",

        "Explain the extinction of dinosaurs philosophically",
    ]


    for query in queries:

        try:

            answer, model, complexity = router.invoke(query)

            print("\n" + "=" * 60)

            print(f"Query: {query}")

            print(f"Complexity: {complexity}")

            print(f"Model Used: {model}")

            print(f"Answer: {answer[:200]}...")

        except ValueError as e:

            print("\n❌", e)


    print("\n" + "=" * 60)

    print("STATS")

    print(router.stats())