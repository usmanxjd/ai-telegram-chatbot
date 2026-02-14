from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    telegram_bot_token: str = Field(alias="TELEGRAM_BOT_TOKEN")
    huggingfacehub_api_token: str = Field(alias="HUGGINGFACEHUB_API_TOKEN")
    hf_llm_repo_id: str = Field(default="meta-llama/Meta-Llama-3-8B-Instruct", alias="HF_LLM_REPO_ID")
    enable_rag: bool = Field(default=True, alias="ENABLE_RAG")
    knowledge_base_dir: Path = Field(default=Path("data/knowledge_base"), alias="KNOWLEDGE_BASE_DIR")
    vectorstore_dir: Path = Field(default=Path("vectorstore"), alias="VECTORSTORE_DIR")
    retrieval_top_k: int = Field(default=4, alias="RETRIEVAL_TOP_K")

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)


def get_settings() -> Settings:
    load_dotenv()
    return Settings()
