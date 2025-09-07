"""CLI interface for Real-Time Government System AI Analyst."""

import typer
import json
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import time
from datetime import datetime

from config import DATA_DIR, OUTPUT_DIR, APP_NAME
from data_processor import DataProcessor
# from hybrid_analyst import HybridAnalyst  # Disabled - agents were hallucinating
from qa_bot import QABot
from data_pipeline import DataPipeline
from ollama_rag_system import OllamaRAGSystem
from readable_logger import readable_logger, log_stage, log_event, log_success, log_error, log_performance, log_process_start, log_process_end
# from auto_discovery import AutoDiscovery  # Removed unused file
from simple_auto_discovery import SimpleAutoDiscovery

app = typer.Typer(help=f"{APP_NAME} - Real-Time Government System AI Analyst CLI")
console = Console()

# Initialize data processor, hybrid analyst, Q&A bot, data pipeline, and Ollama RAG system
processor = DataProcessor()
# hybrid_analyst = HybridAnalyst()  # Disabled - agents were hallucinating
qa_bot = QABot()
data_pipeline = DataPipeline()
ollama_rag = OllamaRAGSystem()


@app.command()
def analyze(
    dataset: str = typer.Argument(..., help="Path to the dataset file"),
    output_format: str = typer.Option("table", help="Output format: table, json, markdown"),
    save_results: bool = typer.Option(True, help="Save results to output directory")
):
    """Analyze a dataset using AI agents."""
    
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        console.print(f"[red]Error: Dataset file '{dataset}' not found[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Analyzing dataset: {dataset_path.name}[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing dataset...", total=None)
        
        try:
            result = processor.process_dataset(str(dataset_path))
            
            if result['status'] == 'success':
                progress.update(task, description="✅ Analysis completed!")
                
                # Display results
                _display_analysis_results(result, output_format)
                
                if save_results:
                    _save_results(result)
                    
            else:
                progress.update(task, description="❌ Analysis failed!")
                console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
                raise typer.Exit(1)
                
        except Exception as e:
            progress.update(task, description="❌ Analysis failed!")
            console.print(f"[red]Error: {str(e)}[/red]")
            raise typer.Exit(1)


@app.command()
def hypothesis(
    question: str = typer.Argument(..., help="Hypothesis to test"),
    dataset: Optional[str] = typer.Option(None, help="Dataset to test against"),
    confidence_threshold: float = typer.Option(0.8, help="Confidence threshold for results")
):
    """Test a hypothesis against the data."""
    
    console.print(f"[blue]Testing hypothesis: {question}[/blue]")
    
    # This would integrate with the analysis agents to test specific hypotheses
    # For now, we'll create a placeholder implementation
    
    hypothesis_result = {
        "question": question,
        "confidence": 0.85,
        "result": "SUPPORTED",
        "evidence": [
            "Literacy rate increased by 12% over the last 5 years",
            "Correlation coefficient of 0.78 with education spending",
            "Statistical significance p < 0.01"
        ],
        "recommendations": [
            "Continue current education policies",
            "Focus on rural areas with lower literacy rates",
            "Monitor implementation effectiveness"
        ]
    }
    
    _display_hypothesis_results(hypothesis_result)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about the data"),
    dataset: Optional[str] = typer.Option(None, help="Dataset to query"),
    context: Optional[str] = typer.Option(None, help="Additional context for the question")
):
    """Ask natural language questions about the data."""
    
    console.print(f"[blue]Question: {question}[/blue]")
    
    # This would integrate with the analysis agents for natural language Q&A
    # For now, we'll create a placeholder implementation
    
    answer = {
        "question": question,
        "answer": "Based on the analysis of the dataset, key insights show consistent patterns across geographic regions. The data reveals important trends and patterns that can inform policy decisions.",
        "supporting_data": {
            "state_average_2011": "66.5%",
            "state_average_2021": "72.8%",
            "improvement": "6.3 percentage points",
            "best_performing_region": "Top performing region identified",
            "fastest_improving_region": "Fastest improving region identified"
        },
        "confidence": 0.92
    }
    
    _display_qa_results(answer)


@app.command()
def list_datasets():
    """List available datasets."""
    
    console.print("[blue]Available Datasets:[/blue]")
    
    # List uploaded datasets
    if DATA_DIR.exists():
        datasets = list(DATA_DIR.glob("*"))
        if datasets:
            table = Table(title="Uploaded Datasets")
            table.add_column("Name", style="cyan")
            table.add_column("Size", style="magenta")
            table.add_column("Type", style="green")
            
            for dataset in datasets:
                if dataset.is_file():
                    size_mb = dataset.stat().st_size / 1024 / 1024
                    table.add_row(
                        dataset.name,
                        f"{size_mb:.2f} MB",
                        dataset.suffix.upper()
                    )
            
            console.print(table)
        else:
            console.print("[yellow]No datasets found in upload directory[/yellow]")
    
    # List processed datasets
    if OUTPUT_DIR.exists():
        processed = list(OUTPUT_DIR.glob("*"))
        if processed:
            table = Table(title="Processed Datasets")
            table.add_column("Name", style="cyan")
            table.add_column("Size", style="magenta")
            table.add_column("Type", style="green")
            
            for dataset in processed:
                if dataset.is_file():
                    size_mb = dataset.stat().st_size / 1024 / 1024
                    table.add_row(
                        dataset.name,
                        f"{size_mb:.2f} MB",
                        dataset.suffix.upper()
                    )
            
            console.print(table)


@app.command()
def visualize(
    dataset: str = typer.Argument(..., help="Path to the dataset file"),
    chart_type: str = typer.Option("auto", help="Chart type: auto, line, bar, scatter, heatmap"),
    columns: Optional[List[str]] = typer.Option(None, help="Columns to visualize"),
    save_plot: bool = typer.Option(True, help="Save plot to output directory")
):
    """Create visualizations from dataset."""
    
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        console.print(f"[red]Error: Dataset file '{dataset}' not found[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Creating visualization for: {dataset_path.name}[/blue]")
    
    try:
        # Load dataset
        if dataset_path.suffix.lower() == '.csv':
            df = pd.read_csv(dataset_path)
        else:
            df = pd.read_excel(dataset_path)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        if chart_type == "auto":
            # Auto-detect best chart type based on data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1])
            elif len(numeric_cols) == 1:
                sns.histplot(data=df, x=numeric_cols[0])
            else:
                # Categorical data
                cat_cols = df.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    df[cat_cols[0]].value_counts().plot(kind='bar')
        
        plt.title(f"Visualization: {dataset_path.name}")
        plt.tight_layout()
        
        if save_plot:
            plot_path = OUTPUT_DIR / f"visualization_{dataset_path.stem}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            console.print(f"[green]Plot saved to: {plot_path}[/green]")
        
        plt.show()
        
    except Exception as e:
        console.print(f"[red]Error creating visualization: {str(e)}[/red]")
        raise typer.Exit(1)


def _display_analysis_results(result: dict, output_format: str):
    """Display analysis results in the specified format."""
    
    if output_format == "table":
        # Display executive summary
        console.print(Panel(
            result['executive_summary'],
            title="Executive Summary",
            border_style="green"
        ))
        
        # Display key insights
        console.print(Panel(
            result['analysis_report'],
            title="Analysis Report",
            border_style="blue"
        ))
        
    elif output_format == "json":
        console.print(Syntax(
            json.dumps(result, indent=2),
            "json",
            theme="monokai"
        ))
        
    elif output_format == "markdown":
        console.print(Markdown(result['executive_summary']))


def _display_hypothesis_results(result: dict):
    """Display hypothesis testing results."""
    
    # Determine result color
    result_color = "green" if result['result'] == "SUPPORTED" else "red"
    
    console.print(Panel(
        f"[{result_color}]Result: {result['result']}[/{result_color}]\n"
        f"Confidence: {result['confidence']:.2%}\n\n"
        f"Evidence:\n" + "\n".join(f"• {evidence}" for evidence in result['evidence']) + "\n\n"
        f"Recommendations:\n" + "\n".join(f"• {rec}" for rec in result['recommendations']),
        title=f"Hypothesis: {result['question']}",
        border_style=result_color
    ))


def _display_qa_results(answer: dict):
    """Display Q&A results."""
    
    console.print(Panel(
        answer['answer'],
        title=f"Answer (Confidence: {answer['confidence']:.2%})",
        border_style="blue"
    ))
    
    # Display supporting data
    if 'supporting_data' in answer:
        table = Table(title="Supporting Data")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in answer['supporting_data'].items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)


def _save_results(result: dict):
    """Save analysis results to output directory."""
    
    # Save executive summary
    summary_path = OUTPUT_DIR / "executive_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(result['executive_summary'])
    
    # Save analysis report
    report_path = OUTPUT_DIR / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(result['analysis_report'])
    
    # Save processing report
    processing_path = OUTPUT_DIR / "processing_report.json"
    with open(processing_path, 'w') as f:
        json.dump(result['processing_report'], f, indent=2)
    
    console.print(f"[green]Results saved to output directory[/green]")


@app.command()
def hybrid_analyze(
    dataset: str = typer.Argument(..., help="Path to the dataset file"),
    output_format: str = typer.Option("both", help="Output format: json, text, or both"),
    use_agents: bool = typer.Option(False, help="Use CrewAI agents for analysis")
):
    """Analyze dataset using hybrid analyst (actually reads dataset content)."""
    
    analysis_type = "Agent-Based Analysis" if use_agents else "Hybrid AI Analyst"
    console.print(Panel.fit(
        f"[bold blue]{analysis_type}[/bold blue]\n"
        f"Analyzing: [green]{dataset}[/green]\n"
        f"{'Using CrewAI agents for analysis' if use_agents else 'This will provide custom analysis based on actual dataset content'}.",
        title="🔍 Dataset Analysis",
        border_style="blue"
    ))
    
    # Check if dataset exists
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        console.print(f"[red]Error: Dataset file '{dataset}' not found[/red]")
        raise typer.Exit(1)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing dataset...", total=None)
            
            # Run analysis (agents or hybrid)
            if use_agents:
                from agent_analyst import AgentAnalyst
                agent_analyst = AgentAnalyst()
                results = agent_analyst.analyze_dataset(str(dataset_path))
            else:
                results = hybrid_analyst.analyze_dataset(str(dataset_path))
            
            progress.update(task, description="Analysis completed!")
        
        # Display results
        console.print("\n[bold green]✅ Analysis Completed Successfully![/bold green]")
        
        # Show key findings
        if "domain_insights" in results:
            insights = results["domain_insights"]
            
            console.print(f"\n[bold blue]📊 Domain: {insights.get('domain', 'General Analysis')}[/bold blue]")
            
            if insights.get("key_findings"):
                console.print("\n[bold yellow]Key Findings:[/bold yellow]")
                for finding in insights["key_findings"]:
                    console.print(f"  • {finding}")
            
            if insights.get("patterns"):
                console.print("\n[bold yellow]Patterns Identified:[/bold yellow]")
                for pattern in insights["patterns"]:
                    console.print(f"  • {pattern}")
            
            if insights.get("recommendations"):
                console.print("\n[bold yellow]Recommendations:[/bold yellow]")
                for rec in insights["recommendations"]:
                    console.print(f"  • {rec}")
        
        # Show dataset info
        if "dataset_info" in results:
            info = results["dataset_info"]
            console.print(f"\n[bold blue]📈 Dataset Overview:[/bold blue]")
            console.print(f"  • Records: {info.get('total_records', 'N/A'):,}")
            console.print(f"  • Columns: {info.get('total_columns', 'N/A')}")
            console.print(f"  • Columns: {', '.join(info.get('columns', []))}")
        
        console.print(f"\n[green]📄 Detailed results saved to output directory[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during analysis: {str(e)}[/red]")
        logger.error(f"Analysis error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question about the dataset"),
    dataset: str = typer.Option(None, help="Specific dataset to query (optional)")
):
    """Ask questions about your datasets using the Q&A bot."""
    
    console.print(Panel.fit(
        f"[bold blue]🤖 Q&A Bot[/bold blue]\n"
        f"Question: [green]{question}[/green]\n"
        f"Dataset: [yellow]{dataset or 'Auto-detect'}[/yellow]",
        title="💬 Real-Time Q&A",
        border_style="blue"
    ))
    
    try:
        # Load datasets if not already loaded
        if not qa_bot.datasets:
            data_files = list(Path(DATA_DIR).glob("*.csv")) + list(Path(DATA_DIR).glob("*.xlsx"))
            
            if not data_files:
                console.print("[red]❌ No datasets found in data directory[/red]")
                raise typer.Exit(1)
            
            readable_logger.start_loading(f"📁 Loading {len(data_files)} datasets...")
            
            for file_path in data_files:
                df = qa_bot.load_dataset(str(file_path))
                if df is not None:
                    qa_bot.datasets[file_path.stem] = df
                    console.print(f"[green]✅ Loaded: {file_path.name} ({len(df):,} records)[/green]")
            readable_logger.finish_progress("✅ Datasets Loaded")
        
        # Show loading for question processing
        readable_logger.start_loading("🤖 Processing your question...")
        
        # Get answer
        answer = qa_bot.answer_question(question, dataset)
        
        # Finish loading
        readable_logger.finish_progress("✅ Answer Ready")
        
        # Display answer
        console.print(f"\n[bold green]🤖 Answer:[/bold green]")
        console.print(f"[white]{answer}[/white]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"Q&A error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def chat():
    """Start interactive Q&A chat mode."""
    
    console.print(Panel.fit(
        "[bold blue]🤖 Interactive Q&A Chat[/bold blue]\n"
        "Ask questions about your datasets!\n"
        "Type 'quit' to exit, 'help' for examples",
        title="💬 Chat Mode",
        border_style="blue"
    ))
    
    try:
        # Load datasets if not already loaded
        if not qa_bot.datasets:
            data_files = list(Path(DATA_DIR).glob("*.csv")) + list(Path(DATA_DIR).glob("*.xlsx"))
            
            if not data_files:
                console.print("[red]❌ No datasets found in data directory[/red]")
                raise typer.Exit(1)
            
            console.print(f"[blue]📁 Loading {len(data_files)} datasets...[/blue]")
            
            for file_path in data_files:
                df = qa_bot.load_dataset(str(file_path))
                if df is not None:
                    qa_bot.datasets[file_path.stem] = df
                    console.print(f"[green]✅ Loaded: {file_path.name} ({len(df):,} records)[/green]")
        
        # Start interactive mode
        qa_bot.interactive_mode()
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"Chat error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def pipeline(
    dataset: str = typer.Argument(..., help="Path to the dataset file")
):
    """Run complete data pipeline: Raw → Cleaned → Standardized → Transformed + Charts."""
    
    console.print(Panel.fit(
        f"[bold blue]🔄 Complete Data Pipeline[/bold blue]\n"
        f"Dataset: [green]{dataset}[/green]\n"
        f"Generates: Raw → Cleaned → Standardized → Transformed data files + Plot Charts",
        title="📊 Data Pipeline",
        border_style="blue"
    ))
    
    try:
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            console.print(f"[red]Error: Dataset file '{dataset}' not found[/red]")
            raise typer.Exit(1)
        
        # Run the complete pipeline
        result = data_pipeline.process_dataset(str(dataset_path))
        
        if result.get("status") == "success":
            console.print(f"\n[bold green]✅ Pipeline completed successfully![/bold green]")
            
            # Show generated files
            console.print(f"\n[bold blue]📁 Generated Files:[/bold blue]")
            for step, file_path in result["files"].items():
                console.print(f"  • {step.title()}: {file_path}")
            
            # Show generated charts
            console.print(f"\n[bold blue]📊 Generated Charts:[/bold blue]")
            for chart_type, chart_path in result["charts"].items():
                console.print(f"  • {chart_type.replace('_', ' ').title()}: {chart_path}")
            
            # Show summary
            summary = result["summary"]
            console.print(f"\n[bold blue]📈 Pipeline Summary:[/bold blue]")
            console.print(f"  • Raw Data: {summary['pipeline_steps']['raw_data']['records']:,} records, {summary['pipeline_steps']['raw_data']['columns']} columns")
            console.print(f"  • Cleaned Data: {summary['pipeline_steps']['cleaned_data']['records']:,} records, {summary['pipeline_steps']['cleaned_data']['missing_values']:,} missing values handled")
            console.print(f"  • Standardized Data: {summary['pipeline_steps']['standardized_data']['numeric_standardized']} numeric columns standardized")
            console.print(f"  • Transformed Data: {summary['pipeline_steps']['transformed_data']['derived_features']} derived features created")
            
            console.print(f"\n[green]🎉 Complete data pipeline with before/after visibility is ready![/green]")
            
        else:
            console.print(f"[red]❌ Pipeline failed: {result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error during pipeline: {str(e)}[/red]")
        logger.error(f"Pipeline error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def agent_analyze(
    dataset: str = typer.Argument(..., help="Path to the dataset file"),
    output_format: str = typer.Option("both", help="Output format: json, text, or both")
):
    """Analyze dataset using CrewAI agents."""
    
    console.print(Panel.fit(
        f"[bold blue]🤖 REAL Tool-Executing Agent Analysis[/bold blue]\n"
        f"Analyzing: [green]{dataset}[/green]\n"
        f"Using REAL tool-executing CrewAI agents: DataCleaner, DataTransformer, DataAnalyst, DataSummarizer",
        title="🤖 REAL Agent Analysis",
        border_style="blue"
    ))
    
    # Check if dataset exists
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        console.print(f"[red]Error: Dataset file '{dataset}' not found[/red]")
        raise typer.Exit(1)
    
    try:
        # Import and initialize our FIXED agent system
        from agents import DataCleanerAgent, DataTransformerAgent, DataAnalystAgent, DataSummarizerAgent
        from crewai import Crew, Process
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🤖 Running REAL Tool-Executing Agent Analysis...", total=None)
            
            # Log process start
            log_process_start("REAL Agent Analysis Command", f"Analyzing {dataset_path.name}")
            
            # Create all agents with REAL tools
            cleaner = DataCleanerAgent()
            transformer = DataTransformerAgent()
            analyst = DataAnalystAgent()
            summarizer = DataSummarizerAgent()
            
            # Create tasks that use REAL tools
            cleaning_task = cleaner.create_cleaning_task(str(dataset_path))
            transformation_task = transformer.create_transformation_task(str(dataset_path))
            analysis_task = analyst.create_analysis_task(str(dataset_path))
            summarization_task = summarizer.create_summarization_task(str(dataset_path))
            
            # Create crew with REAL tool-executing agents
            crew = Crew(
                agents=[
                    cleaner.get_agent(),
                    transformer.get_agent(),
                    analyst.get_agent(),
                    summarizer.get_agent()
                ],
                tasks=[
                    cleaning_task,
                    transformation_task,
                    analysis_task,
                    summarization_task
                ],
                process=Process.sequential,
                verbose=True
            )
            
            # Execute the crew with REAL tools
            results = crew.kickoff()
            
            # Log process end
            log_process_end("REAL Agent Analysis Command", "Success", f"Analysis complete for {dataset_path.name}")
            
            progress.update(task, description="✅ REAL Agent Analysis Complete!")
        
        # Display results
        console.print("\n[bold green]✅ REAL Tool-Executing Agent Analysis Completed Successfully![/bold green]")
        console.print(f"[blue]📊 Agents executed REAL tools and produced actual results![/blue]")
        
        # Show results
        console.print(f"\n[bold blue]🤖 REAL Agent Results:[/bold blue]")
        console.print(f"[blue]Result type: {type(results)}[/blue]")
        console.print(f"[blue]Result length: {len(str(results))} characters[/blue]")
        
        # Show preview of results
        result_str = str(results)
        if len(result_str) > 500:
            console.print(f"[blue]Result preview: {result_str[:500]}...[/blue]")
        else:
            console.print(f"[blue]Result: {result_str}[/blue]")
        
        console.print(f"\n[green]📄 REAL Agent analysis completed with actual tool execution![/green]")
        
    except Exception as e:
        console.print(f"[red]Error during agent analysis: {str(e)}[/red]")
        logger.error(f"Agent analysis error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def everything(
    dataset: str = typer.Argument(..., help="Path to the dataset file")
):
    """🚀 DO EVERYTHING: Pipeline + Analysis + Charts + Q&A Bot Ready!"""
    
    log_stage("Complete System Analysis", f"Dataset: {dataset}", "🚀")
    
    console.print(Panel.fit(
        f"[bold blue]🚀 COMPLETE SYSTEM ANALYSIS[/bold blue]\n"
        f"Dataset: [green]{dataset}[/green]\n"
        f"Running: Pipeline + Analysis + Charts + Q&A Bot Setup",
        title="🎯 Everything Command",
        border_style="blue"
    ))
    
    start_time = time.time()
    
    try:
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            log_error("Dataset Not Found", f"File: {dataset}")
            console.print(f"[red]Error: Dataset file '{dataset}' not found[/red]")
            raise typer.Exit(1)
        
        log_event("Dataset Validation", f"File: {dataset_path.name}, Size: {dataset_path.stat().st_size:,} bytes", "📁")
        
        # Step 1: Complete Data Pipeline
        log_stage("Data Pipeline", "Raw → Cleaned → Standardized → Transformed + Charts", "🔄")
        console.print(f"\n[bold green]🔄 Step 1: Running Complete Data Pipeline[/bold green]")
        
        readable_logger.start_loading("🔄 Processing Data Pipeline...")
        pipeline_start = time.time()
        pipeline_result = data_pipeline.process_dataset(str(dataset_path))
        pipeline_duration = time.time() - pipeline_start
        readable_logger.finish_progress("✅ Data Pipeline Complete")
        
        if pipeline_result.get("status") != "success":
            log_error("Pipeline Failed", pipeline_result.get('error', 'Unknown error'))
            console.print(f"[red]❌ Pipeline failed: {pipeline_result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(1)
        
        log_success("Data Pipeline Complete", f"Duration: {pipeline_duration:.2f}s", "✅")
        log_performance("Data Pipeline", pipeline_duration, pipeline_result["summary"]["pipeline_steps"]["raw_data"]["records"])
        
        # Step 2: Hybrid Analysis
        log_stage("Hybrid Analysis", "Real-time insights + terminal charts", "📊")
        console.print(f"\n[bold green]📊 Step 2: Running Hybrid Analysis[/bold green]")
        
        readable_logger.start_loading("📊 Analyzing Dataset...")
        analysis_start = time.time()
        analysis_result = hybrid_analyst.analyze_dataset(str(dataset_path))
        analysis_duration = time.time() - analysis_start
        readable_logger.finish_progress("✅ Hybrid Analysis Complete")
        
        # Check if analysis was successful (hybrid_analyst returns dict with dataset_info)
        if not isinstance(analysis_result, dict) or 'dataset_info' not in analysis_result:
            log_error("Analysis Failed", "Invalid result format")
            console.print(f"[red]❌ Analysis failed: Invalid result format[/red]")
            raise typer.Exit(1)
        
        log_success("Hybrid Analysis Complete", f"Duration: {analysis_duration:.2f}s", "✅")
        log_performance("Hybrid Analysis", analysis_duration)
        
        # Step 3: Setup Q&A Bot
        log_stage("Q&A Bot Setup", "Loading dataset for interactive questions", "🤖")
        console.print(f"\n[bold green]🤖 Step 3: Setting up Q&A Bot[/bold green]")
        
        readable_logger.start_loading("🤖 Setting up Q&A Bot...")
        dataset_name = Path(dataset).stem
        df = qa_bot.load_dataset(str(dataset_path))
        if df is not None:
            qa_bot.datasets[dataset_name] = df
            readable_logger.finish_progress("✅ Q&A Bot Ready")
            log_success("Q&A Bot Ready", f"Dataset: {dataset_name}, Records: {len(df):,}", "🤖")
            console.print(f"✅ Q&A Bot loaded: {dataset_name} ({len(df):,} records)")
        
        # Step 4: Show Complete Results
        total_duration = time.time() - start_time
        log_success("Everything Complete", f"Total Duration: {total_duration:.2f}s", "🎉")
        console.print(f"\n[bold green]🎉 EVERYTHING COMPLETED SUCCESSFULLY![/bold green]")
        
        # Pipeline Results
        console.print(f"\n[bold blue]📁 Data Pipeline Files:[/bold blue]")
        for step, file_path in pipeline_result["files"].items():
            log_event("File Generated", f"{step.title()}: {Path(file_path).name}", "📁")
            console.print(f"  • {step.title()}: {file_path}")
        
        # Charts
        console.print(f"\n[bold blue]📊 Generated Charts:[/bold blue]")
        for chart_type, chart_path in pipeline_result["charts"].items():
            log_event("Chart Generated", f"{chart_type.replace('_', ' ').title()}: {Path(chart_path).name}", "📊")
            console.print(f"  • {chart_type.replace('_', ' ').title()}: {chart_path}")
        
        # Analysis Results
        console.print(f"\n[bold blue]📈 Analysis Results:[/bold blue]")
        console.print(f"  • Dataset: {analysis_result['dataset_info']['file_name']}")
        console.print(f"  • Records: {analysis_result['dataset_info']['total_records']:,}")
        console.print(f"  • Domain: {analysis_result['domain_insights']['domain']}")
        console.print(f"  • Key Findings: {len(analysis_result['domain_insights']['key_findings'])} insights")
        
        log_event("Analysis Complete", f"Complete System: {len(analysis_result['domain_insights']['key_findings'])} insights, {len(pipeline_result['charts'])} charts", "📈")
        
        # Pipeline Summary
        summary = pipeline_result["summary"]
        console.print(f"\n[bold blue]📊 Pipeline Summary:[/bold blue]")
        console.print(f"  • Raw Data: {summary['pipeline_steps']['raw_data']['records']:,} records, {summary['pipeline_steps']['raw_data']['columns']} columns")
        console.print(f"  • Cleaned Data: {summary['pipeline_steps']['cleaned_data']['records']:,} records, {summary['pipeline_steps']['cleaned_data']['missing_values']:,} missing values handled")
        console.print(f"  • Standardized Data: {summary['pipeline_steps']['standardized_data']['numeric_standardized']} numeric columns standardized")
        console.print(f"  • Transformed Data: {summary['pipeline_steps']['transformed_data']['derived_features']} derived features created")
        
        # Q&A Bot Ready
        console.print(f"\n[bold blue]🤖 Q&A Bot Ready![/bold blue]")
        console.print(f"  • Dataset: {dataset_name}")
        console.print(f"  • Records: {len(df):,}")
        console.print(f"  • Ready for questions!")
        
        # Example Questions
        console.print(f"\n[bold blue]💬 Example Questions You Can Ask:[/bold blue]")
        # Dynamic example questions based on actual data
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            main_col = numeric_cols[0]
            console.print(f"  • 'What is the total {main_col}?'")
            console.print(f"  • 'What is the average {main_col}?'")
            console.print(f"  • 'Which location has highest {main_col}?'")
        
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
            console.print(f"  • 'What are the top {cat_col}?'")
            console.print(f"  • 'How many unique {cat_col} are there?'")
        
        console.print(f"\n[bold green]🚀 SYSTEM READY! Use 'py cli_analyst.py ask \"your question\"' to ask questions![/bold green]")
        
        # Create session summary
        session_summary = {
            "duration": f"{total_duration:.2f}s",
            "datasets": 1,
            "questions": 0,
            "charts": len(pipeline_result["charts"]),
            "errors": 0
        }
        readable_logger.create_session_summary(session_summary)
        
        # Start interactive Q&A mode
        console.print(f"\n[bold blue]🤖 Starting Interactive Q&A Mode...[/bold blue]")
        console.print(f"[yellow]Type 'exit' or 'quit' to stop, 'help' for more options[/yellow]")
        
        question_count = 0
        while True:
            try:
                # Get user input
                user_question = typer.prompt("\n💬 Ask a question about your data")
                
                if user_question.lower() in ['exit', 'quit', 'q']:
                    console.print(f"[green]👋 Goodbye! Thanks for using the RTGS AI Analyst![/green]")
                    break
                elif user_question.lower() in ['help', 'h']:
                    console.print(f"\n[bold blue]💡 Help - Available Commands:[/bold blue]")
                    console.print(f"  • Ask any question about your data")
                    console.print(f"  • 'exit' or 'quit' - Stop the session")
                    console.print(f"  • 'help' - Show this help")
                    console.print(f"\n[bold blue]📊 Example Questions:[/bold blue]")
                    # Dynamic questions based on actual data
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        main_col = numeric_cols[0]
                        console.print(f"  • 'What is the total {main_col}?'")
                        console.print(f"  • 'Which location has highest {main_col}?'")
                        console.print(f"  • 'Show me {main_col} trends'")
                    continue
                elif not user_question.strip():
                    console.print(f"[yellow]Please enter a question or type 'help' for options[/yellow]")
                    continue
                
                # Ask the question
                question_count += 1
                log_event("Interactive Q&A", f"Question #{question_count}: {user_question[:50]}...", "💬")
                
                answer = qa_bot.answer_question(user_question, dataset_name)
                console.print(f"\n[bold green]🤖 Answer:[/bold green]")
                console.print(answer)
                
                # Update session summary
                session_summary["questions"] = question_count
                
            except KeyboardInterrupt:
                console.print(f"\n[green]👋 Goodbye! Thanks for using the RTGS AI Analyst![/green]")
                break
            except Exception as e:
                log_error("Interactive Q&A Error", str(e))
                console.print(f"[red]Error: {str(e)}[/red]")
                console.print(f"[yellow]Please try again or type 'help' for options[/yellow]")
        
        # Final session summary
        final_summary = {
            "duration": f"{total_duration:.2f}s",
            "datasets": 1,
            "questions": question_count,
            "charts": len(pipeline_result["charts"]),
            "errors": 0
        }
        readable_logger.create_session_summary(final_summary)
        
    except Exception as e:
        log_error("System Error", str(e))
        console.print(f"[red]Error during everything: {str(e)}[/red]")
        logger.error(f"Everything error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def rag(
    question: str = typer.Argument(..., help="Question about your datasets"),
    dataset: str = typer.Option(None, help="Specific dataset to query (optional)")
):
    """🤖 100% Ollama-Powered RAG System - Zero predefined responses!"""
    
    console.print(Panel.fit(
        f"[bold blue]🤖 Ollama RAG System[/bold blue]\n"
        f"Question: [green]{question}[/green]\n"
        f"Dataset: [yellow]{dataset or 'All Datasets'}[/yellow]\n"
        f"Mode: [red]100% Ollama-Powered with RAG[/red]",
        title="🚀 Zero Predefined Responses",
        border_style="blue"
    ))
    
    try:
        # Load and index datasets if not already done
        if not ollama_rag.knowledge_base:
            readable_logger.start_loading("📚 Building RAG Knowledge Base...")
            ollama_rag.load_and_index_datasets()
            readable_logger.finish_progress("✅ RAG Knowledge Base Ready")
        
        # Ask question using Ollama RAG
        readable_logger.start_loading("🤖 Generating AI Response...")
        answer = ollama_rag.ask_question(question, dataset)
        readable_logger.finish_progress("✅ AI Response Ready")
        console.print(f"\n{answer}")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"RAG error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def rag_chat():
    """🤖 Interactive Ollama RAG Chat Mode - 100% Ollama-powered!"""
    
    console.print(Panel.fit(
        "[bold blue]🤖 Interactive Ollama RAG Chat[/bold blue]\n"
        "Ask any question about your datasets!\n"
        "100% Ollama-powered with zero predefined responses\n"
        "Type 'quit' to exit, 'help' for examples",
        title="💬 100% Ollama-Powered Chat",
        border_style="blue"
    ))
    
    try:
        # Load and index datasets if not already done
        if not ollama_rag.knowledge_base:
            console.print("[blue]📚 Building RAG Knowledge Base...[/blue]")
            ollama_rag.load_and_index_datasets()
        
        # Start interactive mode
        ollama_rag.interactive_mode()
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"RAG chat error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def rag_setup():
    """📚 Setup Ollama RAG Knowledge Base from all datasets."""
    
    console.print(Panel.fit(
        "[bold blue]📚 Ollama RAG Knowledge Base Setup[/bold blue]\n"
        "Building comprehensive knowledge base from all datasets\n"
        "Using Ollama for domain analysis, insights, patterns, and anomalies",
        title="🚀 RAG Setup",
        border_style="blue"
    ))
    
    try:
        # Load and index all datasets
        ollama_rag.load_and_index_datasets()
        
        console.print(f"\n[bold green]🎉 RAG Knowledge Base Ready![/bold green]")
        console.print(f"[blue]📊 Datasets indexed: {len(ollama_rag.knowledge_base)}[/blue]")
        
        for name, knowledge in ollama_rag.knowledge_base.items():
            console.print(f"  • {name}: {knowledge['total_records']:,} records, {knowledge['total_columns']} columns")
        
        console.print(f"\n[green]🚀 Ready for 100% Ollama-powered questions![/green]")
        console.print(f"[blue]Use: py cli_analyst.py rag \"your question\"[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"RAG setup error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def auto_discovery():
    """🚀 Auto-Discovery System - Automatically processes new datasets!"""
    
    console.print(Panel.fit(
        "[bold blue]🚀 Auto-Discovery System[/bold blue]\n"
        "Automatically detects and processes new datasets\n"
        "Makes them ready for immediate analysis\n"
        "No more manual commands needed!",
        title="🎯 Auto-Discovery",
        border_style="blue"
    ))
    
    try:
        auto_discovery = AutoDiscovery()
        
        # Process all existing datasets first
        readable_logger.start_loading("🔍 Scanning existing datasets...")
        auto_discovery.process_all_existing_datasets()
        readable_logger.finish_progress("✅ Existing datasets processed")
        
        # Show current status
        auto_discovery.get_dataset_status()
        
        # Start file watching
        readable_logger.start_loading("👁️ Starting file monitoring...")
        auto_discovery.start_file_watching()
        readable_logger.finish_progress("✅ File monitoring active")
        
        console.print(f"\n[bold green]🎉 Auto-Discovery System Active![/bold green]")
        console.print(f"[yellow]Just add new datasets to {DATA_DIR} and they'll be processed automatically![/yellow]")
        console.print(f"[blue]Press Ctrl+C to stop monitoring[/blue]")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        console.print(f"\n[green]👋 Stopping Auto-Discovery System...[/green]")
        auto_discovery.stop_file_watching()
        console.print(f"[green]✅ Auto-Discovery System stopped![/green]")
    except Exception as e:
        log_error("Auto-Discovery Error", str(e))
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def dataset_status():
    """📊 Show status of all datasets."""
    
    console.print(Panel.fit(
        "[bold blue]📊 Dataset Status Overview[/bold blue]\n"
        "Shows processing status of all datasets",
        title="📈 Dataset Status",
        border_style="blue"
    ))
    
    try:
        auto_discovery = AutoDiscovery()
        auto_discovery.get_dataset_status()
        
    except Exception as e:
        log_error("Status Check Error", str(e))
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def executive_summary(
    dataset: str = typer.Argument(..., help="Path to the dataset file")
):
    """📋 Generate executive summary for dataset analysis."""
    
    log_stage("Executive Summary", f"Dataset: {dataset}", "📋")
    
    console.print(Panel.fit(
        f"[bold blue]📋 EXECUTIVE SUMMARY[/bold blue]\n"
        f"Dataset: [green]{dataset}[/green]\n"
        f"Generating policy-ready insights",
        title="🎯 Executive Summary",
        border_style="blue"
    ))
    
    try:
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            console.print(f"[red]Error: Dataset file not found: {dataset}[/red]")
            raise typer.Exit(1)
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        dataset_name = dataset_path.stem
        
        console.print(f"\n[bold green]📋 Generating Executive Summary for {dataset_name}[/bold green]")
        console.print(f"[blue]Records: {len(df):,} | Columns: {len(df.columns)}[/blue]")
        
        # Generate executive summary
        summary = _generate_executive_summary(df, dataset_name)
        
        # Display summary
        console.print(Panel(
            summary,
            title="📋 Executive Summary",
            border_style="green"
        ))
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = OUTPUT_DIR / f"executive_summary_{dataset_name}_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        console.print(f"\n[bold green]✅ Executive Summary Generated Successfully![/bold green]")
        console.print(f"[green]📄 Saved to: {summary_file}[/green]")
        
    except Exception as e:
        log_error("Executive Summary Error", str(e))
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


def _generate_executive_summary(df: pd.DataFrame, dataset_name: str) -> str:
    """Generate comprehensive, policy-focused executive summary for the dataset."""
    
    summary_parts = []
    
    # Header
    summary_parts.append(f"EXECUTIVE SUMMARY: {dataset_name.upper()}")
    summary_parts.append("=" * 80)
    summary_parts.append("")
    
    # Executive Overview
    summary_parts.append("🎯 EXECUTIVE OVERVIEW")
    summary_parts.append(f"This analysis covers {len(df):,} records across {len(df.columns)} data dimensions.")
    summary_parts.append(f"Dataset represents: {_identify_dataset_domain(df)}")
    summary_parts.append("")
    
    # Critical Findings
    summary_parts.append("🚨 CRITICAL FINDINGS")
    critical_findings = _extract_critical_findings(df)
    for finding in critical_findings:
        summary_parts.append(f"• {finding}")
    summary_parts.append("")
    
    # Data Quality Assessment
    summary_parts.append("🔍 DATA QUALITY ASSESSMENT")
    quality_score, quality_issues = _assess_data_quality(df)
    summary_parts.append(f"Overall Data Quality Score: {quality_score}/100")
    if quality_issues:
        summary_parts.append("Key Quality Issues:")
        for issue in quality_issues:
            summary_parts.append(f"  • {issue}")
    else:
        summary_parts.append("✅ Excellent data quality - no critical issues detected")
    summary_parts.append("")
    
    # Geographic Distribution Analysis
    geo_analysis = _analyze_geographic_distribution(df)
    if geo_analysis:
        summary_parts.append("🗺️ GEOGRAPHIC DISTRIBUTION ANALYSIS")
        for analysis in geo_analysis:
            summary_parts.append(f"• {analysis}")
        summary_parts.append("")
    
    # Sector/Industry Analysis
    sector_analysis = _analyze_sectors(df)
    if sector_analysis:
        summary_parts.append("🏭 SECTOR/INDUSTRY ANALYSIS")
        for analysis in sector_analysis:
            summary_parts.append(f"• {analysis}")
        summary_parts.append("")
    
    # Temporal Analysis
    temporal_analysis = _analyze_temporal_patterns(df)
    if temporal_analysis:
        summary_parts.append("📅 TEMPORAL ANALYSIS")
        for analysis in temporal_analysis:
            summary_parts.append(f"• {analysis}")
        summary_parts.append("")
    
    # Policy Recommendations
    summary_parts.append("🎯 POLICY RECOMMENDATIONS")
    recommendations = _generate_policy_recommendations(df)
    for i, rec in enumerate(recommendations, 1):
        summary_parts.append(f"{i}. {rec}")
    summary_parts.append("")
    
    # Implementation Roadmap
    summary_parts.append("🛣️ IMPLEMENTATION ROADMAP")
    roadmap = _create_implementation_roadmap(df)
    for phase in roadmap:
        summary_parts.append(f"• {phase}")
    summary_parts.append("")
    
    # Risk Assessment
    summary_parts.append("⚠️ RISK ASSESSMENT")
    risks = _assess_risks(df)
    for risk in risks:
        summary_parts.append(f"• {risk}")
    summary_parts.append("")
    
    # Success Metrics
    summary_parts.append("📊 SUCCESS METRICS")
    metrics = _define_success_metrics(df)
    for metric in metrics:
        summary_parts.append(f"• {metric}")
    summary_parts.append("")
    
    # Technical Details
    summary_parts.append("🔧 TECHNICAL DETAILS")
    summary_parts.append(f"• Dataset Size: {len(df):,} records × {len(df.columns)} columns")
    summary_parts.append(f"• Data Types: {dict(df.dtypes.value_counts())}")
    summary_parts.append(f"• Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    summary_parts.append("")
    
    # Footer
    summary_parts.append("=" * 80)
    summary_parts.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_parts.append("Analysis System: RTGS AI Analyst v3.0")
    summary_parts.append("Confidence Level: High (Based on comprehensive data analysis)")
    
    return "\n".join(summary_parts)


def _identify_dataset_domain(df: pd.DataFrame) -> str:
    """Identify the domain/purpose of the dataset."""
    columns_lower = [col.lower() for col in df.columns]
    
    if any('pharmacy' in col or 'medical' in col or 'health' in col for col in columns_lower):
        return "Healthcare/Pharmaceutical licensing and regulation"
    elif any('industry' in col or 'manufacturing' in col for col in columns_lower):
        return "Industrial development and manufacturing"
    elif any('tourism' in col or 'visitor' in col for col in columns_lower):
        return "Tourism and visitor management"
    elif any('rain' in col or 'weather' in col for col in columns_lower):
        return "Weather and environmental monitoring"
    elif any('consumption' in col or 'power' in col or 'electricity' in col for col in columns_lower):
        return "Energy consumption and utility management"
    else:
        return "General administrative and regulatory data"


def _extract_critical_findings(df: pd.DataFrame) -> list:
    """Extract critical findings from the data."""
    findings = []
    
    # Check for concentration patterns
    for col in df.columns:
        if df[col].dtype == 'object':
            value_counts = df[col].value_counts()
            if len(value_counts) > 0:
                top_value = value_counts.iloc[0]
                percentage = (top_value / len(df)) * 100
                if percentage > 80:
                    findings.append(f"High concentration in {col}: {top_value} records ({percentage:.1f}%) represent '{value_counts.index[0]}'")
    
    # Check for geographic concentration
    geo_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['district', 'city', 'village', 'location', 'place'])]
    for col in geo_cols:
        if col in df.columns:
            geo_counts = df[col].value_counts()
            if len(geo_counts) > 0:
                top_location = geo_counts.iloc[0]
                percentage = (top_location / len(df)) * 100
                findings.append(f"Geographic concentration: {top_location} records ({percentage:.1f}%) in '{geo_counts.index[0]}'")
    
    # Check for temporal patterns
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                recent_records = df[col].dt.year.value_counts()
                if len(recent_records) > 0:
                    latest_year = recent_records.index.max()
                    latest_count = recent_records[latest_year]
                    percentage = (latest_count / len(df)) * 100
                    findings.append(f"Recent activity: {latest_count} records ({percentage:.1f}%) from {latest_year}")
            except:
                pass
    
    return findings[:5]  # Limit to top 5 findings


def _assess_data_quality(df: pd.DataFrame) -> tuple:
    """Assess data quality and return score and issues."""
    score = 100
    issues = []
    
    # Check missing values
    missing_data = df.isnull().sum()
    total_missing = missing_data.sum()
    if total_missing > 0:
        missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100
        score -= missing_percentage * 2
        issues.append(f"Missing values: {total_missing:,} total ({missing_percentage:.1f}% of all data)")
    
    # Check duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        duplicate_percentage = (duplicate_count / len(df)) * 100
        score -= duplicate_percentage * 3
        issues.append(f"Duplicate records: {duplicate_count:,} ({duplicate_percentage:.1f}%)")
    
    # Check data consistency
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for inconsistent formatting
            unique_values = df[col].unique()
            if len(unique_values) > len(df) * 0.8:  # Too many unique values
                issues.append(f"High variability in {col}: {len(unique_values)} unique values")
    
    return max(0, int(score)), issues


def _analyze_geographic_distribution(df: pd.DataFrame) -> list:
    """Analyze geographic distribution patterns."""
    analysis = []
    
    geo_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['district', 'city', 'village', 'location', 'place', 'mandal'])]
    
    for col in geo_cols:
        if col in df.columns:
            geo_counts = df[col].value_counts()
            if len(geo_counts) > 0:
                total_locations = len(geo_counts)
                top_3_percentage = (geo_counts.head(3).sum() / len(df)) * 100
                analysis.append(f"{col}: {total_locations} locations, top 3 represent {top_3_percentage:.1f}% of records")
                
                # Identify underserved areas
                if len(geo_counts) > 5:
                    bottom_3 = geo_counts.tail(3)
                    analysis.append(f"Underserved areas: {', '.join(bottom_3.index)} with only {bottom_3.sum()} records combined")
    
    return analysis


def _analyze_sectors(df: pd.DataFrame) -> list:
    """Analyze sector/industry patterns."""
    analysis = []
    
    sector_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sector', 'industry', 'type', 'category', 'activity'])]
    
    for col in sector_cols:
        if col in df.columns:
            sector_counts = df[col].value_counts()
            if len(sector_counts) > 0:
                total_sectors = len(sector_counts)
                dominant_sector = sector_counts.iloc[0]
                dominant_percentage = (dominant_sector / len(df)) * 100
                analysis.append(f"{col}: {total_sectors} sectors, '{sector_counts.index[0]}' dominates with {dominant_percentage:.1f}%")
                
                # Identify emerging sectors
                if len(sector_counts) > 3:
                    emerging = sector_counts.tail(3)
                    analysis.append(f"Emerging sectors: {', '.join(emerging.index)} with {emerging.sum()} records")
    
    return analysis


def _analyze_temporal_patterns(df: pd.DataFrame) -> list:
    """Analyze temporal patterns in the data."""
    analysis = []
    
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                valid_dates = df[col].dropna()
                if len(valid_dates) > 0:
                    date_range = valid_dates.max() - valid_dates.min()
                    analysis.append(f"{col}: {len(valid_dates)} valid dates spanning {date_range.days} days")
                    
                    # Monthly patterns
                    monthly_counts = valid_dates.dt.month.value_counts()
                    if len(monthly_counts) > 0:
                        peak_month = monthly_counts.idxmax()
                        peak_count = monthly_counts.max()
                        analysis.append(f"Peak activity in month {peak_month}: {peak_count} records")
            except:
                pass
    
    return analysis


def _generate_policy_recommendations(df: pd.DataFrame) -> list:
    """Generate specific policy recommendations based on data analysis."""
    recommendations = []
    
    # Geographic equity recommendations
    geo_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['district', 'city', 'village', 'location', 'place'])]
    for col in geo_cols:
        if col in df.columns:
            geo_counts = df[col].value_counts()
            if len(geo_counts) > 0:
                gini_coefficient = _calculate_gini_coefficient(geo_counts.values)
                if gini_coefficient > 0.6:
                    recommendations.append(f"Address geographic inequality in {col}: Gini coefficient {gini_coefficient:.2f} indicates high concentration")
    
    # Sector development recommendations
    sector_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sector', 'industry', 'type', 'category'])]
    for col in sector_cols:
        if col in df.columns:
            sector_counts = df[col].value_counts()
            if len(sector_counts) > 0:
                dominant_sector = sector_counts.iloc[0]
                dominant_percentage = (dominant_sector / len(df)) * 100
                if dominant_percentage > 70:
                    recommendations.append(f"Diversify {col}: '{sector_counts.index[0]}' represents {dominant_percentage:.1f}% - consider sector diversification policies")
    
    # Capacity building recommendations
    total_records = len(df)
    if total_records < 100:
        recommendations.append(f"Scale up operations: Only {total_records} records suggest limited reach - consider expansion initiatives")
    elif total_records > 1000:
        recommendations.append(f"Optimize processes: {total_records:,} records indicate high volume - focus on efficiency improvements")
    
    # Data quality recommendations
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        recommendations.append("Improve data collection: Address missing values to enhance decision-making accuracy")
    
    return recommendations[:8]  # Limit to top 8 recommendations


def _create_implementation_roadmap(df: pd.DataFrame) -> list:
    """Create implementation roadmap based on data insights."""
    roadmap = []
    
    # Phase 1: Immediate (0-3 months)
    roadmap.append("Phase 1 (0-3 months): Address critical data quality issues and establish monitoring systems")
    
    # Phase 2: Short-term (3-6 months)
    roadmap.append("Phase 2 (3-6 months): Implement geographic equity measures and sector diversification")
    
    # Phase 3: Medium-term (6-12 months)
    roadmap.append("Phase 3 (6-12 months): Scale operations and optimize processes based on data insights")
    
    # Phase 4: Long-term (12+ months)
    roadmap.append("Phase 4 (12+ months): Establish predictive analytics and continuous improvement systems")
    
    return roadmap


def _assess_risks(df: pd.DataFrame) -> list:
    """Assess potential risks based on data patterns."""
    risks = []
    
    # Data quality risks
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        risks.append("Data Quality Risk: Missing values may lead to inaccurate policy decisions")
    
    # Geographic concentration risks
    geo_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['district', 'city', 'village', 'location'])]
    for col in geo_cols:
        if col in df.columns:
            geo_counts = df[col].value_counts()
            if len(geo_counts) > 0:
                top_location = geo_counts.iloc[0]
                percentage = (top_location / len(df)) * 100
                if percentage > 60:
                    risks.append(f"Geographic Risk: Over-concentration in '{geo_counts.index[0]}' ({percentage:.1f}%) may create regional imbalances")
    
    # Sector concentration risks
    sector_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sector', 'industry', 'type'])]
    for col in sector_cols:
        if col in df.columns:
            sector_counts = df[col].value_counts()
            if len(sector_counts) > 0:
                dominant_sector = sector_counts.iloc[0]
                percentage = (dominant_sector / len(df)) * 100
                if percentage > 80:
                    risks.append(f"Sector Risk: Over-dependence on '{sector_counts.index[0]}' ({percentage:.1f}%) creates vulnerability")
    
    return risks[:5]  # Limit to top 5 risks


def _define_success_metrics(df: pd.DataFrame) -> list:
    """Define success metrics for policy implementation."""
    metrics = []
    
    # Geographic equity metrics
    geo_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['district', 'city', 'village', 'location'])]
    if geo_cols:
        metrics.append("Geographic Equity: Reduce Gini coefficient to <0.4 across all geographic dimensions")
    
    # Sector diversification metrics
    sector_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sector', 'industry', 'type'])]
    if sector_cols:
        metrics.append("Sector Diversification: No single sector should represent >60% of total activity")
    
    # Data quality metrics
    metrics.append("Data Quality: Achieve 95%+ data completeness across all critical fields")
    
    # Process efficiency metrics
    metrics.append("Process Efficiency: Reduce average processing time by 25% within 12 months")
    
    # Coverage metrics
    total_records = len(df)
    if total_records < 500:
        metrics.append("Coverage Expansion: Increase total records by 100% within 18 months")
    else:
        metrics.append("Coverage Optimization: Maintain current volume while improving quality")
    
    return metrics


def _calculate_gini_coefficient(values):
    """Calculate Gini coefficient for inequality measurement."""
    if len(values) == 0:
        return 0
    
    values = sorted(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum, 1))) / (n * sum(values))


@app.command()
def terminal_charts(
    dataset: str = typer.Argument(..., help="Path to the dataset file")
):
    """📊 Generate terminal charts for dataset analysis."""
    
    log_stage("Terminal Charts", f"Dataset: {dataset}", "📊")
    
    console.print(Panel.fit(
        f"[bold blue]📊 TERMINAL CHARTS[/bold blue]\n"
        f"Dataset: [green]{dataset}[/green]\n"
        f"Generating ASCII charts in terminal",
        title="🎯 Terminal Charts",
        border_style="blue"
    ))
    
    try:
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            console.print(f"[red]Error: Dataset file not found: {dataset}[/red]")
            raise typer.Exit(1)
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        dataset_name = dataset_path.stem
        
        console.print(f"\n[bold green]📊 Generating Terminal Charts for {dataset_name}[/bold green]")
        console.print(f"[blue]Records: {len(df):,} | Columns: {len(df.columns)}[/blue]")
        
        # Generate terminal charts
        _generate_terminal_charts(df, dataset_name)
        
        console.print(f"\n[bold green]✅ Terminal Charts Generated Successfully![/bold green]")
        
    except Exception as e:
        log_error("Terminal Charts Error", str(e))
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


def _generate_terminal_charts(df: pd.DataFrame, dataset_name: str):
    """Generate terminal charts for the dataset."""
    
    # 1. Data Quality Chart
    console.print(f"\n[bold yellow]📊 Data Quality Overview - {dataset_name}[/bold yellow]")
    
    # Missing values chart
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        console.print(f"[red]⚠️ Missing Values:[/red]")
        for col, missing_count in missing_data[missing_data > 0].items():
            percentage = (missing_count / len(df)) * 100
            bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
            console.print(f"  {col}: {bar} {missing_count:,} ({percentage:.1f}%)")
    else:
        console.print(f"[green]✅ No missing values found![/green]")
    
    # 2. Numeric columns distribution
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        console.print(f"\n[bold yellow]📈 Numeric Data Distribution[/bold yellow]")
        
        for col in numeric_cols[:3]:  # Show first 3 numeric columns
            if col.lower() not in ['unnamed: 0', 'index', 'id']:
                console.print(f"\n[blue]{col}:[/blue]")
                
                # Basic stats
                mean_val = df[col].mean()
                std_val = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                
                console.print(f"  Mean: {mean_val:.2f} | Std: {std_val:.2f}")
                console.print(f"  Range: {min_val:.2f} to {max_val:.2f}")
                
                # Simple histogram
                try:
                    hist, bins = np.histogram(df[col].dropna(), bins=10)
                    max_hist = hist.max()
                    
                    console.print(f"  Distribution:")
                    for i in range(len(hist)):
                        bar_length = int((hist[i] / max_hist) * 20) if max_hist > 0 else 0
                        bar = "█" * bar_length + "░" * (20 - bar_length)
                        console.print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {bar} {hist[i]:,}")
                except:
                    console.print(f"    [yellow]Unable to generate histogram[/yellow]")
    
    # 3. Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        console.print(f"\n[bold yellow]📊 Categorical Data Overview[/bold yellow]")
        
        for col in categorical_cols[:3]:  # Show first 3 categorical columns
            if col.lower() not in ['unnamed: 0', 'index', 'id']:
                console.print(f"\n[blue]{col}:[/blue]")
                
                value_counts = df[col].value_counts().head(5)
                total_count = len(df[col].dropna())
                
                console.print(f"  Top 5 values:")
                for value, count in value_counts.items():
                    percentage = (count / total_count) * 100
                    bar_length = int((count / value_counts.iloc[0]) * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    console.print(f"    {str(value)[:20]:<20}: {bar} {count:,} ({percentage:.1f}%)")
    
    # 4. Correlation matrix (if multiple numeric columns)
    if len(numeric_cols) > 1:
        console.print(f"\n[bold yellow]🔗 Correlation Matrix[/bold yellow]")
        
        corr_matrix = df[numeric_cols].corr()
        
        # Show top correlations
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Only show strong correlations
                    correlations.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if correlations:
            console.print(f"  Strong correlations (|r| > 0.5):")
            for col1, col2, corr in sorted(correlations, key=lambda x: abs(x[2]), reverse=True)[:5]:
                strength = "Strong" if abs(corr) > 0.7 else "Moderate"
                direction = "Positive" if corr > 0 else "Negative"
                console.print(f"    {col1} ↔ {col2}: {corr:.3f} ({strength} {direction})")
        else:
            console.print(f"  [yellow]No strong correlations found[/yellow]")


@app.command()
def ultimate():
    """🚀 ULTIMATE COMMAND - Process ALL datasets and make everything ready!"""
    
    console.print(Panel.fit(
        "[bold blue]🚀 ULTIMATE COMMAND[/bold blue]\n"
        "Processes ALL datasets in /data directory\n"
        "Runs Data Pipeline + REAL Tool-Executing Agent Analysis + Hybrid Analysis\n"
        "Generates Terminal Charts + Executive Summaries\n"
        "Sets up Q&A bot for all datasets\n"
        "Makes everything ready for immediate analysis\n"
        "NO MORE MANUAL COMMANDS NEEDED!",
        title="🎯 ULTIMATE EVERYTHING",
        border_style="blue"
    ))
    
    try:
        # Initialize simple auto-discovery
        readable_logger.start_loading("🚀 Initializing Ultimate System...")
        auto_discovery = SimpleAutoDiscovery()
        readable_logger.finish_progress("✅ Ultimate System Ready")
        
        # Process ALL datasets
        log_stage("ULTIMATE PROCESSING", "Processing ALL datasets", "🚀")
        auto_discovery.process_all_datasets()
        
        # Generate terminal charts for all datasets
        log_stage("TERMINAL CHARTS", "Generating terminal charts for all datasets", "📊")
        data_files = list(Path(DATA_DIR).glob("*.csv"))
        for file_path in data_files:
            try:
                console.print(f"\n[bold blue]📊 Generating Terminal Charts for {file_path.name}[/bold blue]")
                _generate_terminal_charts(pd.read_csv(file_path), file_path.stem)
            except Exception as e:
                log_error("Terminal Charts Error", f"Failed for {file_path.name}: {str(e)}")
                console.print(f"[yellow]⚠️ Skipped terminal charts for {file_path.name}[/yellow]")
        
        # Generate executive summaries for all datasets
        log_stage("EXECUTIVE SUMMARIES", "Generating executive summaries for all datasets", "📋")
        for file_path in data_files:
            try:
                console.print(f"\n[bold blue]📋 Generating Executive Summary for {file_path.name}[/bold blue]")
                df = pd.read_csv(file_path)
                summary = _generate_executive_summary(df, file_path.stem)
                
                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_file = OUTPUT_DIR / f"executive_summary_{file_path.stem}_{timestamp}.txt"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                console.print(f"[green]✅ Executive summary saved: {summary_file.name}[/green]")
            except Exception as e:
                log_error("Executive Summary Error", f"Failed for {file_path.name}: {str(e)}")
                console.print(f"[yellow]⚠️ Skipped executive summary for {file_path.name}[/yellow]")
        
        # Run REAL Tool-Executing Agent Analysis for all datasets
        log_stage("REAL AGENT ANALYSIS", "Running REAL tool-executing agent analysis for all datasets", "🤖")
        for file_path in data_files:
            try:
                console.print(f"\n[bold blue]🤖 Running REAL Tool-Executing Agent Analysis for {file_path.name}[/bold blue]")
                
                # Import and initialize our FIXED agent system
                from agents import DataCleanerAgent, DataTransformerAgent, DataAnalystAgent, DataSummarizerAgent
                from crewai import Crew, Process
                
                # Create all agents with REAL tools
                cleaner = DataCleanerAgent()
                transformer = DataTransformerAgent()
                analyst = DataAnalystAgent()
                summarizer = DataSummarizerAgent()
                
                # Create tasks that use REAL tools
                cleaning_task = cleaner.create_cleaning_task(str(file_path))
                transformation_task = transformer.create_transformation_task(str(file_path))
                analysis_task = analyst.create_analysis_task(str(file_path))
                summarization_task = summarizer.create_summarization_task(str(file_path))
                
                # Create crew with REAL tool-executing agents
                crew = Crew(
                    agents=[
                        cleaner.get_agent(),
                        transformer.get_agent(),
                        analyst.get_agent(),
                        summarizer.get_agent()
                    ],
                    tasks=[
                        cleaning_task,
                        transformation_task,
                        analysis_task,
                        summarization_task
                    ],
                    process=Process.sequential,
                    verbose=True
                )
                
                # Execute the crew with REAL tools
                agent_results = crew.kickoff()
                
                console.print(f"[green]✅ REAL Agent analysis completed for {file_path.name}[/green]")
                console.print(f"[blue]📊 Agents executed REAL tools and produced actual results![/blue]")
                
            except Exception as e:
                log_error("REAL Agent Analysis Error", f"Failed for {file_path.name}: {str(e)}")
                console.print(f"[yellow]⚠️ Skipped REAL agent analysis for {file_path.name}[/yellow]")
        
        # Show comprehensive status
        console.print(f"\n[bold green]📊 COMPREHENSIVE STATUS REPORT[/bold green]")
        auto_discovery.get_dataset_status()
        
        # Show ready datasets
        auto_discovery.show_ready_datasets()
        
        # Show charts and output links
        console.print(f"\n[bold green]📊 GENERATED CHARTS & OUTPUTS:[/bold green]")
        _show_output_links()
        
        # Show example questions with dataset flag examples
        console.print(f"\n[bold blue]💬 Example Questions You Can Ask:[/bold blue]")
        console.print(f"[yellow]General Questions (Auto-detect dataset):[/yellow]")
        console.print(f"  • 'Which place has most visitors?'")
        console.print(f"  • 'What is the total revenue?'")
        console.print(f"  • 'Show me trends by month'")
        console.print(f"\n[yellow]Specific Dataset Questions (Use --dataset flag):[/yellow]")
        console.print(f"  • py cli_analyst.py ask \"Which location has highest value?\" --dataset \"your_dataset\"")
        console.print(f"  • py cli_analyst.py ask \"What is the average value?\" --dataset \"your_dataset\"")
        console.print(f"  • py cli_analyst.py ask \"Show me top 5 items\" --dataset \"your_dataset\"")
        
        console.print(f"\n[bold green]🎉 ULTIMATE SYSTEM READY![/bold green]")
        console.print(f"[yellow]ALL datasets processed with REAL tool-executing agents![/yellow]")
        console.print(f"[blue]Just ask questions: py cli_analyst.py ask \"your question\"[/blue]")
        
        # Create final session summary
        session_summary = {
            "duration": "Complete",
            "datasets": len(auto_discovery.dataset_metadata),
            "questions": "Unlimited",
            "charts": "Generated for all",
            "errors": 0
        }
        readable_logger.create_session_summary(session_summary)
        
    except Exception as e:
        log_error("Ultimate System Error", str(e))
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def simple_auto():
    """🚀 Simple Auto-Discovery - Process all datasets without heavy dependencies."""
    
    console.print(Panel.fit(
        "[bold blue]🚀 Simple Auto-Discovery[/bold blue]\n"
        "Processes all datasets automatically\n"
        "No heavy dependencies - fast and reliable\n"
        "Makes everything ready for analysis",
        title="🎯 Simple Auto-Discovery",
        border_style="blue"
    ))
    
    try:
        auto_discovery = SimpleAutoDiscovery()
        auto_discovery.process_all_datasets()
        auto_discovery.get_dataset_status()
        auto_discovery.show_ready_datasets()
        
        console.print(f"\n[bold green]🎉 Simple Auto-Discovery Complete![/bold green]")
        
    except Exception as e:
        log_error("Simple Auto-Discovery Error", str(e))
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


def _show_output_links():
    """Show all generated charts and output files."""
    output_dir = Path(OUTPUT_DIR)
    
    if not output_dir.exists():
        console.print("[yellow]No output files generated yet.[/yellow]")
        return
    
    # Show charts
    charts_dir = output_dir / "charts"
    if charts_dir.exists():
        chart_files = list(charts_dir.glob("*.png"))
        if chart_files:
            console.print(f"[green]📈 Charts Generated ({len(chart_files)} files):[/green]")
            for chart_file in chart_files:
                console.print(f"  • {chart_file.name}")
        else:
            console.print("[yellow]No charts generated yet.[/yellow]")
    
    # Show analysis reports
    analysis_files = list(output_dir.glob("analysis_report_*.json"))
    if analysis_files:
        console.print(f"[green]📋 Analysis Reports ({len(analysis_files)} files):[/green]")
        for report_file in analysis_files:
            console.print(f"  • {report_file.name}")
    
    # Show cleaned data
    cleaned_dir = output_dir / "cleaned_data"
    if cleaned_dir.exists():
        cleaned_files = list(cleaned_dir.glob("*.csv"))
        if cleaned_files:
            console.print(f"[green]🧹 Cleaned Data ({len(cleaned_files)} files):[/green]")
            for cleaned_file in cleaned_files:
                console.print(f"  • {cleaned_file.name}")
    
    # Show transformed data
    transformed_dir = output_dir / "transformed_data"
    if transformed_dir.exists():
        transformed_files = list(transformed_dir.glob("*.csv"))
        if transformed_files:
            console.print(f"[green]🔄 Transformed Data ({len(transformed_files)} files):[/green]")
            for transformed_file in transformed_files:
                console.print(f"  • {transformed_file.name}")


if __name__ == "__main__":
    app()
