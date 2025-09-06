#!/usr/bin/env python3
"""
Simple Auto-Discovery System - No Heavy Dependencies
Automatically detects and processes new datasets without complex imports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config import DATA_DIR, OUTPUT_DIR
from data_processor import DataProcessor
# from hybrid_analyst import HybridAnalyst  # DISABLED: Agents generate boilerplate instead of real data
from qa_bot import QABot
from data_pipeline import DataPipeline
from readable_logger import readable_logger, log_stage, log_event, log_success, log_error, log_performance

class SimpleAutoDiscovery:
    """Simple auto-discovery system without heavy dependencies."""
    
    def __init__(self):
        self.console = Console()
        self.data_dir = Path(DATA_DIR)
        self.output_dir = Path(OUTPUT_DIR)
        
        # Initialize processors (without RAG for now)
        self.data_processor = DataProcessor()
        # self.hybrid_analyst = HybridAnalyst()  # DISABLED: Agents generate boilerplate instead of real data
        self.qa_bot = QABot()
        self.data_pipeline = DataPipeline()
        
        # Track processed datasets
        self.processed_datasets = {}
        self.dataset_metadata = {}
        self.metadata_file = self.output_dir / "simple_dataset_metadata.json"
        
        # Load existing metadata
        self._load_metadata()
        
        log_success("Simple Auto-Discovery", "Initialized and ready", "🚀")
    
    def _load_metadata(self):
        """Load existing dataset metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.dataset_metadata = json.load(f)
                log_event("Metadata Loaded", f"{len(self.dataset_metadata)} datasets tracked", "📊")
            except Exception as e:
                log_error("Metadata Load Error", str(e))
                self.dataset_metadata = {}
    
    def _save_metadata(self):
        """Save dataset metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.dataset_metadata, f, indent=2)
            log_event("Metadata Saved", f"{len(self.dataset_metadata)} datasets", "💾")
        except Exception as e:
            log_error("Metadata Save Error", str(e))
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get file hash for change detection."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _is_dataset_new_or_changed(self, file_path: Path) -> bool:
        """Check if dataset is new or changed."""
        file_hash = self._get_file_hash(file_path)
        file_key = str(file_path)
        
        if file_key not in self.dataset_metadata:
            return True
        
        stored_hash = self.dataset_metadata[file_key].get('hash', '')
        return file_hash != stored_hash
    
    def process_dataset_completely(self, file_path: Path):
        """Process a dataset completely - everything in one go!"""
        try:
            if not self._is_dataset_new_or_changed(file_path):
                log_event("Dataset Unchanged", f"Skipping: {file_path.name}", "⏭️")
                return False
            
            log_stage("COMPLETE DATASET PROCESSING", f"File: {file_path.name}", "🚀")
            
            start_time = time.time()
            
            # Step 1: Complete Data Pipeline
            readable_logger.start_loading(f"🔄 Processing {file_path.name} - Data Pipeline...")
            pipeline_result = self.data_pipeline.process_dataset(str(file_path))
            readable_logger.finish_progress("✅ Data Pipeline Complete")
            
            if pipeline_result.get("status") != "success":
                log_error("Pipeline Failed", pipeline_result.get('error', 'Unknown error'))
                return False
            
            # Step 2: Hybrid Analysis
            readable_logger.start_loading(f"📊 Analyzing {file_path.name} - Generating Insights...")
            # analysis_result = self.hybrid_analyst.analyze_dataset(str(file_path))  # DISABLED: Agents generate boilerplate
            analysis_result = {"status": "skipped", "reason": "Agent analysis disabled - generates boilerplate instead of real data"}
            readable_logger.finish_progress("✅ Analysis Complete")
            
            # Step 2.5: Agent Analysis
            readable_logger.start_loading(f"🤖 Running Agent Analysis for {file_path.name}...")
            try:
                from agent_analyst import AgentAnalyst
                agent_analyst = AgentAnalyst()
                agent_result = agent_analyst.analyze_dataset(str(file_path))
                readable_logger.finish_progress("✅ Agent Analysis Complete")
            except Exception as e:
                log_error("Agent Analysis Error", str(e))
                readable_logger.finish_progress("⚠️ Agent Analysis Skipped")
            
            # Step 3: Load into Q&A Bot
            readable_logger.start_loading(f"🤖 Setting up Q&A for {file_path.name}...")
            dataset_name = file_path.stem
            df = self.qa_bot.load_dataset(str(file_path))
            if df is not None:
                self.qa_bot.datasets[dataset_name] = df
            readable_logger.finish_progress("✅ Q&A Bot Ready")
            
            # Update metadata
            file_hash = self._get_file_hash(file_path)
            self.dataset_metadata[str(file_path)] = {
                'name': file_path.name,
                'hash': file_hash,
                'processed_at': datetime.now().isoformat(),
                'records': len(df) if df is not None else 0,
                'columns': list(df.columns) if df is not None else [],
                'pipeline_files': pipeline_result.get('files', {}),
                'charts': pipeline_result.get('charts', {}),
                'analysis_insights': len(analysis_result.get('domain_insights', {}).get('key_findings', []))
            }
            
            self._save_metadata()
            
            duration = time.time() - start_time
            log_success("Dataset Fully Processed", 
                       f"{file_path.name}: {len(df):,} records, {duration:.2f}s", "✅")
            log_performance("Complete Processing", duration, len(df) if df is not None else 0)
            
            return True
            
        except Exception as e:
            readable_logger.finish_progress("❌ Processing Failed")
            log_error("Processing Error", str(e))
            return False
    
    def process_all_datasets(self):
        """Process ALL datasets in the data directory."""
        log_stage("PROCESSING ALL DATASETS", "Complete auto-discovery", "🔍")
        
        data_files = list(self.data_dir.glob("*.csv")) + \
                    list(self.data_dir.glob("*.xlsx")) + \
                    list(self.data_dir.glob("*.xls"))
        
        if not data_files:
            log_event("No Datasets Found", "Data directory is empty", "📁")
            return
        
        log_event("Datasets Found", f"{len(data_files)} files detected", "📁")
        
        processed_count = 0
        for file_path in data_files:
            if self.process_dataset_completely(file_path):
                processed_count += 1
        
        log_success("All Datasets Processed", f"{processed_count}/{len(data_files)} datasets ready", "🎉")
    
    def get_dataset_status(self):
        """Get status of all datasets."""
        status_table = Table(title="📊 Dataset Status Overview")
        status_table.add_column("Dataset", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Records", style="yellow")
        status_table.add_column("Charts", style="blue")
        status_table.add_column("Q&A Ready", style="magenta")
        status_table.add_column("Last Processed", style="blue")
        
        for file_path, metadata in self.dataset_metadata.items():
            file_name = Path(file_path).name
            status = "✅ Ready" if metadata.get('records', 0) > 0 else "❌ Failed"
            records = f"{metadata.get('records', 0):,}"
            charts = f"{len(metadata.get('charts', {}))}"
            qa_ready = "✅ Yes" if metadata.get('records', 0) > 0 else "❌ No"
            last_processed = metadata.get('processed_at', 'Never')[:19]
            
            status_table.add_row(file_name, status, records, charts, qa_ready, last_processed)
        
        self.console.print(status_table)
    
    def show_ready_datasets(self):
        """Show all ready datasets for Q&A."""
        ready_datasets = []
        for file_path, metadata in self.dataset_metadata.items():
            if metadata.get('records', 0) > 0:
                ready_datasets.append({
                    'name': Path(file_path).name,
                    'records': metadata.get('records', 0),
                    'columns': metadata.get('columns', [])
                })
        
        if ready_datasets:
            self.console.print(f"\n[bold green]🤖 Q&A Ready Datasets:[/bold green]")
            for dataset in ready_datasets:
                self.console.print(f"  • {dataset['name']}: {dataset['records']:,} records, {len(dataset['columns'])} columns")
        else:
            self.console.print(f"\n[yellow]No datasets ready for Q&A yet.[/yellow]")

def main():
    """Main function for simple auto-discovery."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]🚀 Simple Auto-Discovery System[/bold blue]\n"
        "Processes ALL datasets automatically\n"
        "Makes them ready for immediate analysis\n"
        "No heavy dependencies - fast and reliable!",
        title="🎯 Simple Auto-Discovery",
        border_style="blue"
    ))
    
    auto_discovery = SimpleAutoDiscovery()
    
    try:
        # Process all datasets
        auto_discovery.process_all_datasets()
        
        # Show status
        auto_discovery.get_dataset_status()
        
        # Show ready datasets
        auto_discovery.show_ready_datasets()
        
        console.print(f"\n[bold green]🎉 All Datasets Ready![/bold green]")
        console.print(f"[yellow]You can now ask questions about any dataset![/yellow]")
        console.print(f"[blue]Use: py cli_analyst.py ask \"your question\"[/blue]")
        
    except Exception as e:
        log_error("Auto-Discovery Error", str(e))
        console.print(f"[red]Error: {str(e)}[/red]")

if __name__ == "__main__":
    main()
