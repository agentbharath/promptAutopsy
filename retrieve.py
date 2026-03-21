"""
PromptAutopsy — Retrieval
==========================
Loads the indexed knowledge base from disk.
Takes a failure mode as input, queries the vector
store, and returns the top-k most relevant chunks
with source citations.

Never rebuilds the index — always loads from disk.
"""

import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

CHROMA_DB_PATH = "./chroma_store"
TOP_K = 3
COLLECTION_NAME = "promptautopsy"
MMR_THRESHOLD   = 0.7
_index_cache = None

load_dotenv()
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
Settings.embed_model = embed_model

def load_index() -> VectorStoreIndex:
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )
    collection = chroma_client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    _index_cache = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    print("✓ Index loaded from disk")
    return _index_cache

def retrieve(query: str, k: int = TOP_K) -> list:
    index = load_index()
    retriever = index.as_retriever(
        similarity_top_k = k*2,
    )
    nodes = retriever.retrieve(query)
    nodes  = mmr_rerank(nodes, k, MMR_THRESHOLD)
    print(f"\n✓ Retrieved {len(nodes)} chunks for query: '{query}'")
    for i, node in enumerate(nodes):
        print(f"  [{i+1}] Source: {node.metadata['source']} | "
            f"Failure mode: {node.metadata['failure_mode']} | "
            f"Score: {node.score:.3f}")
    return nodes

def mmr_rerank(nodes: list, k: int, lambda_mult: float = 0.7) -> list:
    """
    Manual MMR implementation.
    Selects diverse results from retrieved nodes.
    lambda_mult: 0 = max diversity, 1 = max relevance
    """
    if len(nodes) <= k:
        return nodes
    
    selected    = [nodes[0]]  # always take top result
    candidates  = nodes[1:]
    
    while len(selected) < k and candidates:
        mmr_scores = []
        for candidate in candidates:
            relevance  = candidate.score
            similarity = max(
                _text_similarity(candidate.text, s.text)
                for s in selected
            )
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * similarity
            mmr_scores.append(mmr_score)
        
        best_idx = mmr_scores.index(max(mmr_scores))
        selected.append(candidates.pop(best_idx))
    
    return selected

def _text_similarity(text1: str, text2: str) -> float:
    """Simple word overlap similarity."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)

if __name__ == "__main__":
    test_queries = [
        "how to fix a vague instruction in a prompt",
        "why does context placement matter in prompts",
        "how to get structured JSON output from an LLM",
        "when should I use chain of thought prompting",
        "how to write a good system prompt role",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        nodes = retrieve(query)
        for i, node in enumerate(nodes):
            print(f"\n  [{i+1}] Source     : {node.metadata['source']}")
            print(f"       Failure mode: {node.metadata['failure_mode']}")
            print(f"       Rule type   : {node.metadata['rule_type']}")
            print(f"       Score       : {node.score:.3f}")
            print(f"       Preview     : {node.text[:150]}")
