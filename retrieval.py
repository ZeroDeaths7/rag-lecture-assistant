from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config.settings import settings


class RetrieverBuilder:
    def __init__(self):
        """Initialize the retriever builder with Ollama BGE-M3 embeddings."""
        self.embeddings = OllamaEmbeddings(model="bge-m3")

    def build_hybrid_retriever(self, docs):
        """Build a hybrid retriever using BM25 and vector-based retrieval."""
    
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=settings.CHROMA_DB_PATH
        )
        
        bm25_retriever = BM25Retriever.from_documents(docs)
        
        vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": settings.VECTOR_SEARCH_K}
        )

        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=settings.HYBRID_RETRIEVER_WEIGHTS
        )
        
        return hybrid_retriever
