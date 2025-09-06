#!/usr/bin/env python3
"""
Agent-Based Analyst using CrewAI agents for dataset analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from agents import (
    CoordinatorAgent,
    DataCleanerAgent,
    DataTransformerAgent,
    DataAnalystAgent,
    DataSummarizerAgent
)
from crewai import Crew, Process
from readable_logger import readable_logger, log_stage, log_event, log_success, log_error, log_agent_start, log_agent_end, log_crew_start, log_crew_end, log_task_start, log_task_end, log_process_start, log_process_end


class AgentAnalyst:
    """Agent-based analyst using CrewAI agents."""
    
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.console = Console()
        
        # Initialize all 5 agents
        self.coordinator = CoordinatorAgent()
        self.data_cleaner = DataCleanerAgent()
        self.data_transformer = DataTransformerAgent()
        self.data_analyst = DataAnalystAgent()
        self.data_summarizer = DataSummarizerAgent()
    
    def analyze_dataset(self, file_path: str, dataset_name: str = None) -> Dict[str, Any]:
        """Analyze dataset using CrewAI agents."""
        
        log_stage("AGENT ANALYSIS", f"Starting agent-based analysis for {file_path}", "🤖")
        
        try:
            # Read dataset
            df = pd.read_csv(file_path)
            dataset_name = dataset_name or Path(file_path).stem
            
            log_event("Dataset Loaded", f"{len(df)} records, {len(df.columns)} columns", "📊")
            
            # Create agent crew with all 5 agents
            crew = Crew(
                agents=[
                    self.coordinator.get_agent(),
                    self.data_cleaner.get_agent(),
                    self.data_transformer.get_agent(), 
                    self.data_analyst.get_agent(),
                    self.data_summarizer.get_agent()
                ],
                tasks=[
                    self._create_coordination_task(file_path),
                    self._create_cleaning_task(file_path),
                    self._create_transformation_task(file_path),
                    self._create_analysis_task(file_path),
                    self._create_summarization_task(file_path)
                ],
                process=Process.sequential,
                verbose=True
            )
            
            log_event("Agent Crew Created", "5 agents ready for analysis", "🚀")
            
            # Log crew start
            agent_names = ["Coordinator", "DataCleaner", "DataTransformer", "DataAnalyst", "DataSummarizer"]
            task_names = ["Coordination", "Data Cleaning", "Data Transformation", "Data Analysis", "Data Summarization"]
            log_crew_start("RTGS Analysis Crew", agent_names, task_names)
            
            # Execute agent analysis
            readable_logger.start_loading("🤖 Running Agent Analysis...")
            log_process_start("Agent Analysis", f"Processing {dataset_name} with 5 agents")
            
            result = crew.kickoff()
            
            log_process_end("Agent Analysis", "Success", f"Completed analysis for {dataset_name}")
            log_crew_end("RTGS Analysis Crew", f"Analysis complete for {dataset_name}")
            readable_logger.finish_progress("✅ Agent Analysis Complete")
            
            # Process results
            analysis_result = self._process_agent_results(result, df, dataset_name)
            
            # Save analysis report
            self._save_analysis_report(analysis_result, dataset_name)
            
            log_success("Agent Analysis", f"Analysis complete for {dataset_name}")
            
            return analysis_result
            
        except Exception as e:
            log_error("Agent Analysis Error", str(e))
            raise e
    
    def _create_coordination_task(self, file_path: str):
        """Create coordination task for agent."""
        log_task_start("Coordination Task", "Coordinator")
        return self.coordinator.create_coordination_task(file_path)
    
    def _create_cleaning_task(self, file_path: str):
        """Create data cleaning task for agent."""
        log_task_start("Data Cleaning Task", "DataCleaner")
        return self.data_cleaner.create_cleaning_task(file_path)
    
    def _create_transformation_task(self, file_path: str):
        """Create data transformation task for agent."""
        log_task_start("Data Transformation Task", "DataTransformer")
        return self.data_transformer.create_transformation_task(file_path)
    
    def _create_analysis_task(self, file_path: str):
        """Create data analysis task for agent."""
        log_task_start("Data Analysis Task", "DataAnalyst")
        return self.data_analyst.create_analysis_task(file_path)
    
    def _create_summarization_task(self, file_path: str):
        """Create data summarization task for agent."""
        log_task_start("Data Summarization Task", "DataSummarizer")
        return self.data_summarizer.create_summarization_task(file_path)
    
    def _process_agent_results(self, result, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Process agent results into structured format."""
        
        analysis_result = {
            "dataset_info": {
                "file_name": dataset_name,
                "records": len(df),
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict()
            },
            "agent_analysis": {
                "timestamp": datetime.now().isoformat(),
                "agents_used": ["Coordinator", "DataCleaner", "DataTransformer", "DataAnalyst", "DataSummarizer"],
                "analysis_result": str(result),
                "status": "completed"
            },
            "insights": {
                "data_quality": self._analyze_data_quality(df),
                "statistical_summary": self._get_statistical_summary(df),
                "recommendations": self._generate_recommendations(df)
            }
        }
        
        return analysis_result
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality."""
        return {
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
    
    def _get_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical summary."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        summary = {}
        
        for col in numeric_cols:
            summary[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "count": int(df[col].count())
            }
        
        return summary
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on data analysis."""
        recommendations = []
        
        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 20]
        
        if len(high_missing) > 0:
            recommendations.append(f"High missing values detected in: {', '.join(high_missing.index)}")
        
        # Check for duplicates
        if df.duplicated().sum() > 0:
            recommendations.append(f"Found {df.duplicated().sum()} duplicate rows - consider removing")
        
        # Check data types
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            recommendations.append(f"Consider normalization for numeric columns: {', '.join(numeric_cols)}")
        
        return recommendations
    
    def _save_analysis_report(self, analysis_result: Dict[str, Any], dataset_name: str):
        """Save analysis report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"agent_analysis_report_{dataset_name}_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(analysis_result, f, indent=2, default=str)
        
        log_event("Report Saved", f"Agent analysis report saved: {report_file.name}", "📄")


def main():
    """Test the agent analyst."""
    analyst = AgentAnalyst()
    
    # Test with tourism data
    result = analyst.analyze_dataset("data/Tourism Foreign Visitors Data 2024.csv")
    print("Agent Analysis Complete!")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()