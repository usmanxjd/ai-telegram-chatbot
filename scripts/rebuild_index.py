from support_bot.config import get_settings
from support_bot.knowledge_base import KnowledgeBaseManager


def main() -> None:
    settings = get_settings()
    manager = KnowledgeBaseManager(settings.knowledge_base_dir, settings.vectorstore_dir)
    db = manager.build_or_load()
    if db is None:
        print("No knowledge base documents found. Add .md/.txt files first.")
    else:
        print(f"Vector store ready at: {settings.vectorstore_dir}")


if __name__ == "__main__":
    main()
