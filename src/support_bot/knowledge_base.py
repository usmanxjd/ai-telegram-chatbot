from __future__ import annotations

from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_SUFFIXES = {".md", ".txt"}


class KnowledgeBaseManager:
    def __init__(self, source_dir: Path, vectorstore_dir: Path) -> None:
        self.source_dir = source_dir
        self.vectorstore_dir = vectorstore_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def _load_documents(self) -> list[Document]:
        documents: list[Document] = []
        for file in self.source_dir.glob("**/*"):
            if not file.is_file() or file.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            text = file.read_text(encoding="utf-8")
            documents.append(Document(page_content=text, metadata={"source": str(file)}))
        return documents

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        return splitter.split_documents(documents)

    def build_or_load(self) -> FAISS | None:
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)

        index_path = self.vectorstore_dir / "index.faiss"
        store_path = self.vectorstore_dir / "index.pkl"
        if index_path.exists() and store_path.exists():
            return FAISS.load_local(
                folder_path=str(self.vectorstore_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )

        docs = self._load_documents()
        if not docs:
            return None

        chunks = self._split_documents(docs)
        db = FAISS.from_documents(chunks, self.embeddings)
        db.save_local(str(self.vectorstore_dir))
        return db
