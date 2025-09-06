#!/usr/bin/env python3
"""
Readable Logger - Human-friendly messages indicating stage and notable events
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler
from rich.traceback import install
import time

# Install rich traceback for better error display
install()

class ReadableLogger:
    """Human-friendly logger with stage indicators and notable events."""
    
    def __init__(self, log_file: str = "logs/readable.log"):
        self.console = Console()
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("TelanganaAnalyst")
        self.logger.setLevel(logging.INFO)
        
        # Rich handler for console
        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            markup=True
        )
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        self.logger.addHandler(rich_handler)
        self.logger.addHandler(file_handler)
        
        # Progress tracking
        self.current_progress = None
        self.start_time = None
    
    def log_stage(self, stage: str, description: str = "", emoji: str = "🔄"):
        """Log a major stage in the process."""
        message = f"{emoji} [bold blue]{stage}[/bold blue]"
        if description:
            message += f": {description}"
        
        self.logger.info(message)
        self._log_to_file(f"STAGE: {stage} - {description}")
    
    def log_event(self, event: str, details: str = "", emoji: str = "📊"):
        """Log a notable event."""
        message = f"{emoji} [bold green]{event}[/bold green]"
        if details:
            message += f": {details}"
        
        self.logger.info(message)
        self._log_to_file(f"EVENT: {event} - {details}")
    
    def log_warning(self, warning: str, details: str = "", emoji: str = "⚠️"):
        """Log a warning."""
        message = f"{emoji} [bold yellow]{warning}[/bold yellow]"
        if details:
            message += f": {details}"
        
        self.logger.warning(message)
        self._log_to_file(f"WARNING: {warning} - {details}")
    
    def log_error(self, error: str, details: str = "", emoji: str = "❌"):
        """Log an error."""
        message = f"{emoji} [bold red]{error}[/bold red]"
        if details:
            message += f": {details}"
        
        self.logger.error(message)
        self._log_to_file(f"ERROR: {error} - {details}")
    
    def log_success(self, success: str, details: str = "", emoji: str = "✅"):
        """Log a success."""
        message = f"{emoji} [bold green]{success}[/bold green]"
        if details:
            message += f": {details}"
        
        self.logger.info(message)
        self._log_to_file(f"SUCCESS: {success} - {details}")
    
    def log_data_processing(self, operation: str, records: int, columns: int = None):
        """Log data processing operations."""
        details = f"{records:,} records"
        if columns:
            details += f", {columns} columns"
        
        self.log_event(f"Data Processing: {operation}", details, "📊")
    
    def log_ollama_request(self, model: str, prompt_length: int, response_time: float = None):
        """Log Ollama LLM requests."""
        details = f"Model: {model}, Prompt: {prompt_length} chars"
        if response_time:
            details += f", Time: {response_time:.2f}s"
        
        self.log_event("Ollama LLM Request", details, "🤖")
    
    def log_rag_operation(self, operation: str, datasets: int, vectors: int = None):
        """Log RAG operations."""
        details = f"{datasets} datasets"
        if vectors:
            details += f", {vectors} vectors"
        
        self.log_event(f"RAG: {operation}", details, "🧠")
    
    def log_file_operation(self, operation: str, file_path: str, size: int = None):
        """Log file operations."""
        details = f"File: {Path(file_path).name}"
        if size:
            details += f", Size: {size:,} bytes"
        
        self.log_event(f"File: {operation}", details, "📁")
    
    def log_analysis_result(self, analysis_type: str, insights: int, charts: int = None):
        """Log analysis results."""
        details = f"{insights} insights"
        if charts:
            details += f", {charts} charts"
        
        self.log_success(f"Analysis Complete: {analysis_type}", details, "📈")
    
    def log_performance(self, operation: str, duration: float, records: int = None):
        """Log performance metrics."""
        details = f"Duration: {duration:.2f}s"
        if records:
            details += f", Speed: {records/duration:,.0f} records/sec"
        
        self.log_event(f"Performance: {operation}", details, "⚡")
    
    def start_progress(self, total: int, description: str = "Processing"):
        """Start a progress bar."""
        self.current_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )
        self.current_progress.start()
        self.task = self.current_progress.add_task(description, total=total)
        self.start_time = time.time()
    
    def start_loading(self, description: str = "Loading"):
        """Start a simple loading spinner."""
        self.current_progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]{description}[/bold blue]"),
            console=self.console
        )
        self.current_progress.start()
        self.task = self.current_progress.add_task(description, total=None)
        self.start_time = time.time()
    
    def update_progress(self, advance: int = 1, description: str = None):
        """Update progress bar."""
        if self.current_progress:
            if description:
                self.current_progress.update(self.task, description=description)
            self.current_progress.advance(self.task, advance)
    
    def finish_progress(self, description: str = "Complete"):
        """Finish progress bar."""
        if self.current_progress:
            duration = time.time() - self.start_time if self.start_time else 0
            self.current_progress.update(self.task, description=f"{description} ({duration:.2f}s)")
            self.current_progress.stop()
            self.current_progress = None
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log a summary of operations."""
        table = Table(title="📋 Operation Summary", show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in summary.items():
            table.add_row(key, str(value))
        
        self.console.print(table)
        self._log_to_file(f"SUMMARY: {summary}")
    
    def log_dataset_info(self, dataset_name: str, info: Dict[str, Any]):
        """Log dataset information."""
        self.log_event(f"Dataset Loaded: {dataset_name}", 
                      f"{info.get('records', 0):,} records, {info.get('columns', 0)} columns", "📊")
    
    def log_agent_activity(self, agent_name: str, task: str, status: str = "started"):
        """Log agent activity."""
        emoji = "🤖" if status == "started" else "✅" if status == "completed" else "❌"
        self.log_event(f"Agent: {agent_name}", f"{task} - {status}", emoji)
    
    def log_pipeline_stage(self, stage: str, input_records: int, output_records: int, 
                          changes: List[str] = None):
        """Log pipeline stage."""
        details = f"{input_records:,} → {output_records:,} records"
        if changes:
            details += f", Changes: {', '.join(changes)}"
        
        self.log_event(f"Pipeline: {stage}", details, "🔄")
    
    def log_chart_generation(self, chart_type: str, file_path: str):
        """Log chart generation."""
        self.log_success(f"Chart Generated: {chart_type}", f"Saved to: {Path(file_path).name}", "📊")
    
    def log_qa_interaction(self, question: str, dataset: str, response_time: float):
        """Log Q&A interactions."""
        self.log_event("Q&A Interaction", 
                      f"Question: {question[:50]}..., Dataset: {dataset}, Time: {response_time:.2f}s", "💬")
    
    def _log_to_file(self, message: str):
        """Log to file with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp} - {message}\n")
    
    def create_session_summary(self, session_data: Dict[str, Any]):
        """Create a session summary."""
        panel = Panel.fit(
            f"[bold blue]📊 Session Summary[/bold blue]\n"
            f"Duration: {session_data.get('duration', 'N/A')}\n"
            f"Datasets Processed: {session_data.get('datasets', 0)}\n"
            f"Questions Asked: {session_data.get('questions', 0)}\n"
            f"Charts Generated: {session_data.get('charts', 0)}\n"
            f"Errors: {session_data.get('errors', 0)}",
            title="🎯 Session Complete",
            border_style="green"
        )
        self.console.print(panel)
        self._log_to_file(f"SESSION_SUMMARY: {session_data}")


    def log_agent_start(self, agent_name: str, task_description: str = ""):
        """Log when an agent starts working."""
        self.log_event(f"🤖 Agent Started", f"{agent_name}: {task_description}", "🚀")
    
    def log_agent_end(self, agent_name: str, result_summary: str = ""):
        """Log when an agent finishes working."""
        self.log_success(f"Agent Complete", f"{agent_name}: {result_summary}")
    
    def log_process_start(self, process_name: str, details: str = ""):
        """Log when a process starts."""
        self.log_event(f"⚙️ Process Started", f"{process_name}: {details}", "🔄")
    
    def log_process_end(self, process_name: str, status: str = "Success", details: str = ""):
        """Log when a process ends."""
        if status.lower() == "success":
            self.log_success(f"Process Complete", f"{process_name}: {details}")
        else:
            self.log_error(f"Process Failed", f"{process_name}: {details}")
    
    def log_crew_start(self, crew_name: str, agents: list, tasks: list):
        """Log when a crew starts execution."""
        agent_list = ", ".join(agents)
        task_count = len(tasks)
        self.log_event(f"🚀 Crew Started", f"{crew_name} with {len(agents)} agents ({agent_list}) - {task_count} tasks", "🎯")
    
    def log_crew_end(self, crew_name: str, results_summary: str = ""):
        """Log when a crew finishes execution."""
        self.log_success(f"Crew Complete", f"{crew_name}: {results_summary}")
    
    def log_task_start(self, task_name: str, agent_name: str = ""):
        """Log when a task starts."""
        if agent_name:
            self.log_event(f"📋 Task Started", f"{task_name} assigned to {agent_name}", "📝")
        else:
            self.log_event(f"📋 Task Started", f"{task_name}", "📝")
    
    def log_task_end(self, task_name: str, status: str = "Success", agent_name: str = ""):
        """Log when a task ends."""
        if status.lower() == "success":
            if agent_name:
                self.log_success(f"Task Complete", f"{task_name} by {agent_name}")
            else:
                self.log_success(f"Task Complete", f"{task_name}")
        else:
            if agent_name:
                self.log_error(f"Task Failed", f"{task_name} by {agent_name}")
            else:
                self.log_error(f"Task Failed", f"{task_name}")


# Global logger instance
readable_logger = ReadableLogger()

# Convenience functions
def log_stage(stage: str, description: str = "", emoji: str = "🔄"):
    """Log a major stage."""
    readable_logger.log_stage(stage, description, emoji)

def log_event(event: str, details: str = "", emoji: str = "📊"):
    """Log a notable event."""
    readable_logger.log_event(event, details, emoji)

def log_success(success: str, details: str = "", emoji: str = "✅"):
    """Log a success."""
    readable_logger.log_success(success, details, emoji)

def log_warning(warning: str, details: str = "", emoji: str = "⚠️"):
    """Log a warning."""
    readable_logger.log_warning(warning, details, emoji)

def log_error(error: str, details: str = "", emoji: str = "❌"):
    """Log an error."""
    readable_logger.log_error(error, details, emoji)

def log_data_processing(operation: str, records: int, columns: int = None):
    """Log data processing."""
    readable_logger.log_data_processing(operation, records, columns)

def log_ollama_request(model: str, prompt_length: int, response_time: float = None):
    """Log Ollama requests."""
    readable_logger.log_ollama_request(model, prompt_length, response_time)

def log_rag_operation(operation: str, datasets: int, vectors: int = None):
    """Log RAG operations."""
    readable_logger.log_rag_operation(operation, datasets, vectors)

def log_performance(operation: str, duration: float, records: int = None):
    """Log performance metrics."""
    readable_logger.log_performance(operation, duration, records)

def log_agent_start(agent_name: str, task_description: str = ""):
    """Log when an agent starts working."""
    readable_logger.log_agent_start(agent_name, task_description)

def log_agent_end(agent_name: str, result_summary: str = ""):
    """Log when an agent finishes working."""
    readable_logger.log_agent_end(agent_name, result_summary)

def log_process_start(process_name: str, details: str = ""):
    """Log when a process starts."""
    readable_logger.log_process_start(process_name, details)

def log_process_end(process_name: str, status: str = "Success", details: str = ""):
    """Log when a process ends."""
    readable_logger.log_process_end(process_name, status, details)

def log_crew_start(crew_name: str, agents: list, tasks: list):
    """Log when a crew starts execution."""
    readable_logger.log_crew_start(crew_name, agents, tasks)

def log_crew_end(crew_name: str, results_summary: str = ""):
    """Log when a crew finishes execution."""
    readable_logger.log_crew_end(crew_name, results_summary)

def log_task_start(task_name: str, agent_name: str = ""):
    """Log when a task starts."""
    readable_logger.log_task_start(task_name, agent_name)

def log_task_end(task_name: str, status: str = "Success", agent_name: str = ""):
    """Log when a task ends."""
    readable_logger.log_task_end(task_name, status, agent_name)
