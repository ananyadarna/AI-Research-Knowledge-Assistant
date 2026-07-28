from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8000
    database_url: str = "sqlite:///./data/assistant.db"
    vector_db_dir: str = "./data/vector_db"
    upload_dir: str = "./data/raw_documents"
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model_name: str = "gpt-4o-mini"
    embedding_provider: str = "local"  # options: local, openai

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# Ensure necessary folders are created
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.vector_db_dir, exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./data/dataset", exist_ok=True)
