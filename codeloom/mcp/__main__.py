"""
Entry point for CodeLoom MCP server.
Run: python -m codeloom.mcp

IMPORTANT: All logging must go to stderr. stdout is the MCP stdio channel.
"""
import asyncio
import logging
import os
import sys

# Redirect ALL logging to stderr before any imports that might log
logging.basicConfig(
    stream=sys.stderr,
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
# Suppress noisy libraries
for _noisy in [
    "httpx", "httpcore", "llama_index", "openai", "anthropic",
    "transformers", "sqlalchemy", "sqlalchemy.engine", "uvicorn",
]:
    logging.getLogger(_noisy).setLevel(logging.ERROR)


async def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    from codeloom.core.db.db import DatabaseManager
    from codeloom.mcp.server import CodeLoomMCPServer
    from mcp.server.stdio import stdio_server

    # Initialize DB manager
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://codeloom:codeloom@localhost:5432/codeloom_dev",
    )
    db_manager = DatabaseManager(db_url)

    # Optionally initialize the RAG pipeline for semantic search
    pipeline = None
    try:
        # LocalRAGPipeline takes host + database_url; no "lightweight" flag exists
        from codeloom.pipeline import LocalRAGPipeline

        pipeline = LocalRAGPipeline(
            host=os.getenv("OLLAMA_HOST", "localhost"),
            database_url=db_url,
        )
        # Only initialize the vector store / embedding layer — skip RAPTOR worker
        # pipeline.initialize() is not a public method; the constructor suffices
        print("[codeloom-mcp] RAG pipeline initialized", file=sys.stderr)
    except Exception as exc:
        print(f"[codeloom-mcp] RAG pipeline unavailable: {exc}", file=sys.stderr)

    mcp_server = CodeLoomMCPServer(db_manager=db_manager, pipeline=pipeline)

    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.server.run(
            read_stream,
            write_stream,
            mcp_server.server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
