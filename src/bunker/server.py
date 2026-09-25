"""Server CLI entry point."""

import click
import uvicorn


@click.group()
def cli():
    """Bunker API Server - Biological Foundation Models."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--workers", default=1, type=int, help="Number of worker processes")
def serve(host: str, port: int, reload: bool, workers: int):
    """Start the Bunker API server."""
    click.echo(f"Starting Bunker API server on {host}:{port}")
    click.echo(f"OpenAPI docs available at http://{host}:{port}/docs")

    uvicorn.run(
        "bunker.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
    )


@cli.command()
def models():
    """List available models."""
    from bunker.registry import get_all_metadata

    click.echo("Available models:\n")

    metadata = get_all_metadata()
    for name, meta in sorted(metadata.items()):
        click.echo(f"  {name}")
        click.echo(f"    Task: {meta.task}")
        click.echo(f"    Extra: {meta.extra}")
        click.echo(f"    Description: {meta.description}")
        if meta.memory_gb:
            click.echo(f"    GPU Memory: ~{meta.memory_gb}GB")
        click.echo()


if __name__ == "__main__":
    cli()
