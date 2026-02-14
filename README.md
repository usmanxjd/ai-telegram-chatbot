# AI Customer Support Telegram Chatbot (LangChain + Hugging Face)

A production-style starter project for an **AI customer support Telegram chatbot** built with:

- **python-telegram-bot** for Telegram integration
- **LangChain** for prompt orchestration and retrieval
- **Hugging Face Inference API** for LLM responses
- **FAISS + sentence-transformers** for local knowledge-base retrieval (RAG)

## Features

- Telegram `/start` and `/help` commands
- AI-generated support replies from Hugging Face-hosted models
- Optional RAG over local `.md` / `.txt` support docs
- Persistent FAISS vector index for fast startup
- Clean project structure with config and tests

## Project structure

```text
.
├── data/knowledge_base/faq.md
├── src/support_bot/
│   ├── bot.py
│   ├── config.py
│   ├── knowledge_base.py
│   └── support_chain.py
├── tests/test_support_chain.py
├── .env.example
├── pyproject.toml
└── README.md
```

## 1) Prerequisites

- Python 3.10+
- Telegram bot token (create via [@BotFather](https://t.me/BotFather))
- Hugging Face API token with inference access

## 2) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## 3) Configure environment

```bash
cp .env.example .env
```

Update `.env` values:

- `TELEGRAM_BOT_TOKEN`
- `HUGGINGFACEHUB_API_TOKEN`
- (optional) `HF_LLM_REPO_ID`, `ENABLE_RAG`, `KNOWLEDGE_BASE_DIR`

## 4) Add your support content

Put support docs into `data/knowledge_base/` as `.md` or `.txt`.

## 5) Run the bot

```bash
support-bot
```

or

```bash
python -m support_bot.bot
```

Then open Telegram and chat with your bot.

## 6) Test and lint

```bash
pytest
ruff check .
```

## Deployment notes

- For production, run as a long-lived service (systemd, Docker, Kubernetes).
- Consider Telegram webhooks instead of polling for large-scale usage.
- Restrict model output with additional moderation/guardrails for compliance-sensitive domains.

## Customization ideas

- Add conversation memory using LangChain memory modules.
- Route intents (refund, technical issue, billing) to specialized chains.
- Integrate CRM/ticketing tools (Zendesk/Freshdesk) for handoff.
