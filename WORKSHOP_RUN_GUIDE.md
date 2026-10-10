# Workshop run and submission guide

1. Open this project folder in VS Code.
2. Install `uv` if needed. Run `uv sync` (the first run downloads dependencies).
3. Copy `.env.example` to `.env` and set `LLM_PROVIDER=groq`, `GROQ_MODEL=openai/gpt-oss-20b`, `GROQ_API_KEY=YOUR_PRIVATE_KEY`, and `EMBEDDING_MODEL=all-MiniLM-L6-v2`. Never commit `.env`.
4. Run `uv run streamlit run src/rag_agent/ui/app.py`.
5. Upload `lstm_intermediate.md` using the sidebar and click **Ingest Documents**.
6. Ask **What does the forget gate do?** and confirm the answer and source are shown.
7. Re-ingest the same file and verify duplicates are skipped.
8. Save a genuine screenshot showing your question, the generated answer, and sources as `screenshots/rag_demo.png`.
9. Run `uv run pytest tests/ -v` and inspect any environment-specific failures.
10. Create your own empty GitHub repository. Set its URL as `origin`, commit, and push. Verify `.env` is not staged.

**Important:** The supplied project includes optional advanced LangGraph node stubs that are not used by this Markdown-only Streamlit demo. No API key or live Groq run is included in this archive. The screenshot must be captured from your own working application.
