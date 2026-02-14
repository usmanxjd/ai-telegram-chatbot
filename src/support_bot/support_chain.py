from __future__ import annotations

from dataclasses import dataclass

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint


@dataclass
class SupportChainFactory:
    hf_api_token: str
    repo_id: str
    retrieval_top_k: int = 4

    def _build_llm(self) -> HuggingFaceEndpoint:
        return HuggingFaceEndpoint(
            repo_id=self.repo_id,
            huggingfacehub_api_token=self.hf_api_token,
            max_new_tokens=350,
            temperature=0.2,
            repetition_penalty=1.05,
        )

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs)

    def create(self, vectorstore: FAISS | None = None):
        llm = self._build_llm()

        if vectorstore is None:
            prompt = ChatPromptTemplate.from_template(
                """
You are an AI customer support specialist.
Provide clear, polite, actionable responses.
If the user asks for escalation, suggest contacting a human agent.

Customer query: {question}

Response:
""".strip()
            )
            return prompt | llm

        retriever = vectorstore.as_retriever(search_kwargs={"k": self.retrieval_top_k})
        prompt = ChatPromptTemplate.from_template(
            """
You are an AI customer support specialist.
Use the context to answer accurately. If the context is insufficient, state that and provide a safe next step.
Always be concise and empathetic.

Context:
{context}

Customer query:
{question}

Helpful response:
""".strip()
        )

        chain = (
            {
                "context": retriever | RunnableLambda(self._format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        return chain
