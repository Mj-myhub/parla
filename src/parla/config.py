"""Typed configuration via pydantic-settings (reads from .env)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "groq"          # groq | openai | anthropic
    planner_model: str = "llama-3.3-70b-versatile"
    synth_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = ".chroma"
    langsmith_tracing: bool = False


settings = Settings()
