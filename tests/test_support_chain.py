from langchain.schema.runnable import RunnableLambda

from support_bot.support_chain import SupportChainFactory


class DummyDoc:
    def __init__(self, page_content: str, source: str) -> None:
        self.page_content = page_content
        self.metadata = {"source": source}


def test_format_docs_combines_sources() -> None:
    docs = [DummyDoc("Reset password from settings.", "faq.md")]
    result = SupportChainFactory._format_docs(docs)
    assert "Source: faq.md" in result
    assert "Reset password" in result


def test_chain_without_vectorstore_uses_prompt_and_llm(monkeypatch) -> None:
    factory = SupportChainFactory(hf_api_token="token", repo_id="repo")

    monkeypatch.setattr(
        factory,
        "_build_llm",
        lambda: RunnableLambda(lambda prompt_value: f"ok::{prompt_value.to_string()[:20]}"),
    )

    chain = factory.create(vectorstore=None)
    output = chain.invoke({"question": "Where is my order?"})
    assert str(output).startswith("ok::")
