"""
PromptAutopsy — Indexing Pipeline
===================================
Reads raw docs from /raw_docs
Chunks them, embeds them, stores in ChromaDB
Run once. Loads from disk every time after.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import VectorStoreIndex, Document, StorageContext
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key=  os.getenv("ANTHROPIC_API_KEY")

embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
Settings.embed_model = embed_model

RAW_DOCS_PATH   = "./raw_docs"
CHROMA_DB_PATH  = "./chroma_store"
CHUNK_SIZE     = 512
OVERLAP        = 64

def detect_failure_mode(text: str) -> str:
    text = text.lower()
    if any(w in text for w in ["vague", "unclear", "specific", "explicit", "precise"]):
        return "vague_instruction"
    elif any(w in text for w in ["audience", "context", "role", "purpose", "who"]):
        return "missing_context"
    elif any(w in text for w in ["json", "format", "schema", "structure", "xml"]):
        return "wrong_format"
    elif any(w in text for w in ["conflicting", "contradict", "consistent", "opposing"]):
        return "conflicting_instructions"
    elif any(w in text for w in ["example", "few-shot", "demonstration", "sample"]):
        return "missing_examples"
    else:
        return "general"

def detect_rule_type(text: str) -> str:
    text = text.lower()
    if any(w in text for w in ["add", "include", "insert", "provide"]):
        return "additive"
    elif any(w in text for w in ["remove", "avoid", "don't", "never"]):
        return "restrictive"
    elif any(w in text for w in ["replace", "rewrite", "change", "convert"]):
        return "transformative"
    else:
        return "informational"


def load_documents(path: str) -> list:
    documents = []
    for file in Path(path).glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        doc = Document(
                text=text,
                metadata={
                    "source"   : file.stem,
                    "file_path": str(file)
                }
            )
        documents.append(doc)
        print(f"  ✓ {file.stem} — {len(text):,} characters loaded")
    print(f"\n✓ Loaded {len(documents)} documents from {path}")
    return documents

def chunk_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size = CHUNK_SIZE,
        chunk_overlap = OVERLAP
    )

    chunks = []
    for doc in documents:
        split_texts = splitter.split_text(doc.text)
        for i, chunk_text in enumerate(split_texts):
            chunk_doc = Document(
                text=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "failure_mode": detect_failure_mode(chunk_text),
                    "rule_type": detect_rule_type(chunk_text)
                }
            )
            chunks.append(chunk_doc)
    print(f"\n✓ {len(chunks)} chunks created from {len(documents)} documents")
    return chunks
    
def build_index(chunks: list) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,  
    )
    collection = chroma_client.get_or_create_collection("promptautopsy")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
            documents=chunks,
            storage_context=storage_context
        )
    print(f"\n✓ Successfully indexed {len(chunks)} chunks into ChromaDB")
    return index

def main():
    if os.path.exists(CHROMA_DB_PATH):
        print("✓ Index already exists — loading from disk")
        print("  Delete chroma_store/ and rerun to rebuild")
        return

    documents = load_documents(RAW_DOCS_PATH)
    chunks    = chunk_documents(documents)
    index     = build_index(chunks)


if __name__ == "__main__":
    main()
        
