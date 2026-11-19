"""DeepGuard AI Command-Line Interface"""
import click
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from app.core.orchestrator import DetectionOrchestrator
from app.core.reporter import ReportGenerator

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🛡️ DeepGuard AI - Phishing & Deepfake Detection CLI"""
    pass


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--type', '-t', 'analysis_type', 
              type=click.Choice(['email', 'image', 'audio', 'video']),
              help='Override file type detection')
@click.option('--output', '-o', type=click.Path(), 
              help='Output report path')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'html', 'pdf']),
              default='json', help='Report format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analyze(file_path, analysis_type, output, output_format, verbose):
    """Analyze a file for phishing or deepfake indicators."""
    
    console.print(Panel.fit(
        f"[bold blue]🛡️ DeepGuard AI Analysis[/bold blue]\n"
        f"File: [cyan]{file_path}[/cyan]",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Initializing detectors...", total=None)
        
        orchestrator = DetectionOrchestrator()
        
        progress.update(task, description="Analyzing file...")
        result = orchestrator.analyze(file_path, analysis_type)
    
    if not result["success"]:
        console.print(f"[bold red]❌ Error:[/bold red] {result.get('error')}")
        sys.exit(1)
    
    # Display results
    overall = result.get("overall_risk", {})
    risk_level = overall.get("level", "UNKNOWN")
    risk_score = overall.get("score", 0)
    
    risk_colors = {
        "CRITICAL": "bold red",
        "HIGH": "red",
        "MEDIUM": "yellow",
        "LOW": "green",
        "MINIMAL": "bold green",
        "UNKNOWN": "white"
    }
    color = risk_colors.get(risk_level, "white")
    
    console.print()
    console.print(Panel.fit(
        f"[{color}]Risk Level: {risk_level}[/{color}]\n"
        f"Risk Score: {risk_score:.1%}\n"
        f"Threat Detected: {'⚠️ YES' if overall.get('is_threat') else '✅ NO'}\n"
        f"Confidence: {overall.get('confidence', 0):.1%}",
        title="Analysis Result",
        border_style=color.split()[-1]
    ))
    
    # Details table
    table = Table(title="Detection Details")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    file_info = result.get("file_info", {})
    perf = result.get("performance", {})
    
    table.add_row("File Name", file_info.get("name", "N/A"))
    table.add_row("File Type", file_info.get("type", "N/A"))
    table.add_row("File Size", f"{file_info.get('size_mb', 0)} MB")
    table.add_row("SHA-256", file_info.get("sha256", "N/A")[:16] + "...")
    table.add_row("Processing Time", f"{perf.get('processing_time_seconds', 0)}s")
    table.add_row("Integrity Preserved", 
                  "✅ Yes" if result.get("integrity", {}).get("integrity_preserved") else "⚠️ No")
    
    console.print(table)
    
    # Findings
    detection = result.get("detection_result", {})
    findings = detection.get("findings", [])
    
    if findings:
        console.print("\n[bold yellow]⚠️ Indicators Found:[/bold yellow]")
        for f in findings:
            console.print(f"  • {f}")
    
    # Recommendations
    recommendations = detection.get("recommendations", [])
    if recommendations:
        console.print("\n[bold green]✅ Recommendations:[/bold green]")
        for r in recommendations:
            console.print(f"  • {r}")
    
    # Save report
    if output:
        reporter = ReportGenerator()
        if output_format == "json":
            reporter.to_json(result, output)
        elif output_format == "html":
            html = reporter.to_html(result)
            Path(output).write_text(html)
        elif output_format == "pdf":
            reporter.to_pdf(result, output)
        
        console.print(f"\n[green]📄 Report saved to: {output}[/green]")


@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--pattern', '-p', default='*', help='File pattern to match')
def batch(directory, pattern):
    """Analyze all files in a directory."""
    from pathlib import Path
    
    files = list(Path(directory).glob(pattern))
    console.print(f"Found [cyan]{len(files)}[/cyan] files to analyze\n")
    
    orchestrator = DetectionOrchestrator()
    results = orchestrator.analyze_batch([str(f) for f in files])
    
    # Summary table
    table = Table(title="Batch Analysis Results")
    table.add_column("File", style="cyan")
    table.add_column("Type", style="white")
    table.add_column("Risk Level", style="white")
    table.add_column("Risk Score", style="white")
    table.add_column("Status", style="white")
    
    for r in results["results"]:
        if r["success"]:
            risk = r.get("overall_risk", {})
            level = risk.get("level", "UNKNOWN")
            color = {"CRITICAL": "red", "HIGH": "red", "MEDIUM": "yellow",
                    "LOW": "green", "MINIMAL": "green"}.get(level, "white")
            table.add_row(
                r["file_info"]["name"][:30],
                r["file_info"]["type"],
                f"[{color}]{level}[/{color}]",
                f"{risk.get('score', 0):.1%}",
                "✅"
            )
        else:
            table.add_row(r.get("file_hash", "N/A")[:30], "?", "?", "?", "❌")
    
    console.print(table)
    console.print(f"\n[green]Successful: {results['successful']}[/green] | "
                  f"[red]Failed: {results['failed']}[/red]")


@cli.command()
def serve():
    """Start the API server."""
    import uvicorn
    console.print("[bold green]🚀 Starting DeepGuard AI API server...[/bold green]")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


@cli.command()
def ui():
    """Start the web interface."""
    import subprocess
    console.print("[bold green]🌐 Starting DeepGuard AI Web UI...[/bold green]")
    subprocess.run(["streamlit", "run", "ui/streamlit_app.py"])


@cli.command()
def check():
    """Check if all models are loaded correctly."""
    console.print("[bold]Checking model status...[/bold]\n")
    
    orchestrator = DetectionOrchestrator()
    
    models = {
        "Text Detector (DistilBERT)": orchestrator.text_detector.model is not None,
        "Image Detector (EfficientNet)": orchestrator.image_detector.model is not None,
        "Audio Detector (Wav2Vec2)": orchestrator.audio_detector.feature_extractor is not None,
        "Video Detector": orchestrator.video_detector.image_detector.model is not None,
        "Logo Detector (YOLOv8)": orchestrator.logo_detector.model is not None,
    }
    
    for name, loaded in models.items():
        icon = "✅" if loaded else "❌"
        console.print(f"{icon} {name}")


if __name__ == "__main__":
    cli()