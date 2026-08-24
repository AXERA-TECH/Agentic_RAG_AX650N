"""CLI entry point for Agentic RAG."""

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

app = typer.Typer(help="Agentic RAG — Multi-modal ReAct-powered RAG System")
console = Console()


@app.command()
def chat(
    message: str = typer.Argument(..., help="Your message/question"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream the response"),
    provider: str = typer.Option("", help="LLM provider to use"),
):
    """Send a message to the agent."""
    from agentic_rag.data.models import AgentInput
    from agentic_rag.runtime.orchestrator import get_orchestrator

    async def _run():
        orchestrator = get_orchestrator()
        input_data = AgentInput(query=message)

        if stream:
            async for event in orchestrator.process_stream(
                query=message,
                provider=provider,
                input_data=input_data,
            ):
                if event.event_type.value == "text_delta":
                    console.print(event.data.get("content", ""), end="")
                elif event.event_type.value == "tool_call_start":
                    console.print(f"\n[dim]🔧 {event.data['tool']}...[/dim]")
                elif event.event_type.value == "tool_call_result":
                    status = "✓" if event.data.get("success") else "✗"
                    console.print(f"[dim]   {status} Done[/dim]")
            console.print()
        else:
            with console.status("[bold green]Thinking..."):
                output = await orchestrator.process(
                    query=message,
                    provider=provider,
                    input_data=input_data,
                )

            console.print(Panel(Markdown(output.final_answer), title="Answer"))
            if output.tool_calls_made:
                console.print(f"[dim]Tools used: {len(output.tool_calls_made)}, Iterations: {output.iterations}[/dim]")

    asyncio.run(_run())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the API server."""
    import uvicorn
    console.print(f"[bold green]Starting Agentic RAG server on {host}:{port}[/bold green]")
    uvicorn.run(
        "agentic_rag.entrypoints.rest.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command()
def ingest(
    file: str = typer.Option(..., "--file", "-f", help="File to ingest"),
    source: str = typer.Option("cli", help="Source identifier"),
):
    """Ingest a document into the knowledge base."""
    from agentic_rag.orchestration.l1_tools.rag_tools import RAGIngestTool

    async def _run():
        path = Path(file)
        if not path.exists():
            console.print(f"[red]File not found: {file}[/red]")
            sys.exit(1)

        content = path.read_text()
        tool = RAGIngestTool()
        result = await tool.execute(content=content, source=source)
        console.print(f"[green]{result}[/green]")

    asyncio.run(_run())


@app.command()
def info():
    """Show system information."""
    from agentic_rag import __version__
    from agentic_rag.config.settings import get_settings

    settings = get_settings()

    console.print(Panel(f"Agentic RAG v{__version__}", title="System Info"))
    console.print(f"Default LLM Provider: {settings.default_provider}")
    for name, cfg in settings.llm_providers.items():
        console.print(f"  {name}: {cfg.model} @ {cfg.api_base}")
    console.print(f"Milvus: {settings.milvus.host}:{settings.milvus.port}")
    console.print(f"Embedding: {settings.embedding.model} (dim={settings.embedding.dim})")


if __name__ == "__main__":
    app()
