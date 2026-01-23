from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config.settings import settings
import sys
import os

try:
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    HAS_HYBRID = True
except ImportError:
    print("Warning: Hybrid Search libraries not found. Defaulting to pure Vector Search.")
    HAS_HYBRID = False

class RetrieverBuilder:
    def __init__(self):
        """Initialize the retriever builder with Local Ollama embeddings."""
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text" 
        )

    def build_hybrid_retriever(self, docs):
        """Build a retriever (Vector-only fallback if Hybrid fails)."""
    
        if os.path.exists(settings.CHROMA_DB_PATH) and os.listdir(settings.CHROMA_DB_PATH):
            print("✅ Existing database found. Loading from disk...")
            vector_store = Chroma(
                persist_directory=settings.CHROMA_DB_PATH,
                embedding_function=self.embeddings
            )
        else:
            print(f"⚠️ No database found. Creating Vector Store with {len(docs)} documents...")
            try:
                vector_store = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    persist_directory=settings.CHROMA_DB_PATH
                )
            except Exception as e:
                print(f"CRITICAL ERROR: Could not connect to Ollama. Is 'ollama serve' running?")
                print(f"Details: {e}")
                sys.exit(1)
        
        vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": settings.VECTOR_SEARCH_K}
        )

        # hybrid search
        if HAS_HYBRID:
            try:
                print("Building Hybrid Retriever (BM25 + Vector)...")
                bm25_retriever = BM25Retriever.from_documents(docs)
                hybrid_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=settings.HYBRID_RETRIEVER_WEIGHTS
                )
                return hybrid_retriever
            except Exception as e:
                print(f"Hybrid build failed ({e}). Fallback to Vector Search.")
                return vector_retriever

        print("🔹 Using Standard Vector Retriever.")
        return vector_retriever