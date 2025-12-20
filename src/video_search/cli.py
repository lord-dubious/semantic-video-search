"""CLI for Semantic Video Search."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from video_search.models import VideoSearchConfig
from video_search.search import create_search_engine

app = typer.Typer(
    name="video-search",
    help="Semantic video search using LanceDB and Gemini AI",
)
console = Console()


@app.command()
def index(
    video_path: Path = typer.Argument(..., help="Path to video file"),
    frames_dir: Optional[str] = typer.Option(
        "./data/frames", "--frames-dir", "-f", help="Directory for extracted frames"
    ),
    interval: float = typer.Option(
        2.0, "--interval", "-i", help="Frame extraction interval in seconds"
    ),
    db_path: Optional[str] = typer.Option(
        "./data/lancedb", "--db", "-d", help="Path to LanceDB database"
    ),
) -> None:
    """Index a video for semantic search."""
    if not video_path.exists():
        console.print(f"[red]Error:[/red] Video not found: {video_path}")
        raise typer.Exit(1)

    config = VideoSearchConfig(
        frames_dir=frames_dir or "./data/frames",
    )
    config.extraction.interval_seconds = interval
    config.vector_store.db_path = db_path or "./data/lancedb"

    engine = create_search_engine(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing video...", total=None)

        def on_progress(p):
            progress.update(
                task,
                description=f"[{p.status.value}] Frames: {p.embedded_frames}/{p.total_frames}",
            )

        engine.on_progress(on_progress)

        try:
            metadata = engine.index_video_sync(video_path)
            console.print(f"\n[green]Success![/green] Indexed video: {metadata.video_id}")
            console.print(f"  Duration: {metadata.duration:.1f}s")
            console.print(f"  Resolution: {metadata.width}x{metadata.height}")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of results to return"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Similarity threshold"),
    verify: bool = typer.Option(False, "--verify", "-v", help="Verify results with LLM"),
    db_path: Optional[str] = typer.Option(
        "./data/lancedb", "--db", "-d", help="Path to LanceDB database"
    ),
) -> None:
    """Search for video moments matching a query."""
    config = VideoSearchConfig()
    config.vector_store.db_path = db_path or "./data/lancedb"

    engine = create_search_engine(config)

    with console.status("Searching..."):
        results = engine.search_sync(
            query=query,
            top_k=top_k,
            threshold=threshold,
            verify=verify,
        )

    if not results.results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Search Results for: '{query}'")
    table.add_column("Video ID", style="cyan")
    table.add_column("Timestamp", style="green")
    table.add_column("Score", style="yellow")
    table.add_column("Description", style="white")

    for result in results.results:
        mins = int(result.timestamp // 60)
        secs = int(result.timestamp % 60)
        table.add_row(
            result.video_id,
            f"{mins:02d}:{secs:02d}",
            f"{result.similarity:.2f}",
            result.description[:60] + "..." if len(result.description) > 60 else result.description,
        )

    console.print(table)
    console.print(f"\nFound {results.total_found} results in {results.search_time_ms:.1f}ms")


@app.command()
def list_videos(
    db_path: Optional[str] = typer.Option(
        "./data/lancedb", "--db", "-d", help="Path to LanceDB database"
    ),
) -> None:
    """List all indexed videos."""
    config = VideoSearchConfig()
    config.vector_store.db_path = db_path or "./data/lancedb"

    engine = create_search_engine(config)

    videos = engine.list_indexed_videos()

    if not videos:
        console.print("[yellow]No videos indexed.[/yellow]")
        return

    table = Table(title="Indexed Videos")
    table.add_column("Video ID", style="cyan")
    table.add_column("Frames", style="green")

    for video_id in videos:
        info = engine.get_video_info(video_id)
        table.add_row(video_id, str(info["frame_count"]))

    console.print(table)


@app.command()
def stats(
    db_path: Optional[str] = typer.Option(
        "./data/lancedb", "--db", "-d", help="Path to LanceDB database"
    ),
) -> None:
    """Show search engine statistics."""
    config = VideoSearchConfig()
    config.vector_store.db_path = db_path or "./data/lancedb"

    engine = create_search_engine(config)

    stats_data = engine.get_stats()

    console.print("\n[bold]Search Engine Statistics[/bold]")
    console.print(f"  Total Frames: {stats_data['total_frames']}")
    console.print(f"  Indexed Videos: {stats_data['indexed_videos']}")
    console.print(f"\n[bold]Configuration[/bold]")
    for key, value in stats_data["config"].items():
        console.print(f"  {key}: {value}")


@app.command()
def delete(
    video_id: str = typer.Argument(..., help="Video ID to delete"),
    db_path: Optional[str] = typer.Option(
        "./data/lancedb", "--db", "-d", help="Path to LanceDB database"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a video from the index."""
    config = VideoSearchConfig()
    config.vector_store.db_path = db_path or "./data/lancedb"

    engine = create_search_engine(config)

    if not force:
        confirm = typer.confirm(f"Delete video {video_id}?")
        if not confirm:
            console.print("Cancelled.")
            return

    deleted = engine.delete_video(video_id)
    console.print(f"[green]Deleted[/green] {deleted} frames from video {video_id}")


@app.command()
def demo() -> None:
    """Run a demo of the video search system."""
    console.print("\n[bold]Semantic Video Search Demo[/bold]")
    console.print("=" * 40)

    console.print("\n[cyan]1. Creating search engine...[/cyan]")
    engine = create_search_engine()

    console.print("\n[cyan]2. Getting statistics...[/cyan]")
    stats_data = engine.get_stats()
    console.print(f"   Total indexed frames: {stats_data['total_frames']}")
    console.print(f"   Indexed videos: {stats_data['indexed_videos']}")

    console.print("\n[cyan]3. Example usage:[/cyan]")
    console.print("   video-search index /path/to/video.mp4")
    console.print('   video-search search "person walking in the park"')
    console.print("   video-search list-videos")
    console.print("   video-search stats")

    console.print("\n[green]Demo complete![/green]")


if __name__ == "__main__":
    app()
