from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    CHROMA_DB_PATH: str = "./chroma_db"
    VECTOR_SEARCH_K: int = 4
    HYBRID_RETRIEVER_WEIGHTS: list[float] = [0.5, 0.5]
