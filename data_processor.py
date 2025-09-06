"""Data processing pipeline for Telangana Open Data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
import json
from datetime import datetime

from config import DATA_DIR, OUTPUT_DIR
from agents import (
    DataCleanerAgent, 
    DataTransformerAgent, 
    DataAnalystAgent, 
    DataSummarizerAgent
)
from crewai import Crew, Process
# from dataset_info_injector import get_dataset_info  # Removed unused file


class DataProcessor:
    """Main data processing pipeline for Telangana Open Data."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.cleaner = DataCleanerAgent()
        self.transformer = DataTransformerAgent()
        self.analyst = DataAnalystAgent()
        self.summarizer = DataSummarizerAgent()
        
        # Setup logging
        logger.add(
            OUTPUT_DIR / "processing.log",
            rotation="10 MB",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
    
    def process_dataset(self, dataset_path: str) -> Dict:
        """
        Process a dataset through the complete pipeline.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Dictionary containing processing results and file paths
        """
        logger.info(f"Starting processing of dataset: {dataset_path}")
        
        try:
            # Validate dataset
            if not self._validate_dataset(dataset_path):
                raise ValueError(f"Invalid dataset: {dataset_path}")
            
            # Create crew for processing
            crew = Crew(
                agents=[
                    self.cleaner.get_agent(),
                    self.transformer.get_agent(),
                    self.analyst.get_agent(),
                    self.summarizer.get_agent()
                ],
                tasks=[],  # Will be populated dynamically
                process=Process.sequential,
                verbose=True
            )
            
            # Step 1: Data Cleaning
            logger.info("Step 1: Data Cleaning")
            cleaning_task = self._create_cleaning_task_with_data_info(dataset_path)
            crew.tasks = [cleaning_task]
            cleaning_result = crew.kickoff()
            
            # Extract cleaned dataset path from result
            cleaned_path = self._extract_file_path(str(cleaning_result), "cleaned")
            
            # Step 2: Data Transformation
            logger.info("Step 2: Data Transformation")
            transformation_task = self.transformer.create_transformation_task(cleaned_path)
            crew.tasks = [transformation_task]
            transformation_result = crew.kickoff()
            
            # Extract transformed dataset path from result
            transformed_path = self._extract_file_path(str(transformation_result), "transformed")
            
            # Step 3: Data Analysis
            logger.info("Step 3: Data Analysis")
            analysis_task = self.analyst.create_analysis_task(transformed_path)
            crew.tasks = [analysis_task]
            analysis_result = crew.kickoff()
            
            # Step 4: Summarization
            logger.info("Step 4: Summarization")
            summarization_task = self.summarizer.create_summarization_task(str(analysis_result))
            crew.tasks = [summarization_task]
            summary_result = crew.kickoff()
            
            # Generate processing report
            processing_report = self._generate_processing_report(
                dataset_path, cleaned_path, transformed_path, 
                str(cleaning_result), str(transformation_result), 
                str(analysis_result), str(summary_result)
            )
            
            logger.info("Dataset processing completed successfully")
            
            return {
                "status": "success",
                "original_dataset": dataset_path,
                "cleaned_dataset": cleaned_path,
                "transformed_dataset": transformed_path,
                "analysis_report": str(analysis_result),
                "executive_summary": str(summary_result),
                "processing_report": processing_report,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_path}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_dataset(self, dataset_path: str) -> bool:
        """Validate the dataset file."""
        path = Path(dataset_path)
        
        if not path.exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            return False
        
        if path.suffix.lower() not in ['.csv', '.xlsx', '.xls', '.json']:
            logger.error(f"Unsupported file format: {path.suffix}")
            return False
        
        try:
            # Try to read the dataset
            if path.suffix.lower() == '.csv':
                pd.read_csv(dataset_path, nrows=5)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                pd.read_excel(dataset_path, nrows=5)
            elif path.suffix.lower() == '.json':
                pd.read_json(dataset_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error reading dataset: {str(e)}")
            return False
    
    def _extract_file_path(self, result: str, prefix: str) -> str:
        """Extract file path from agent result."""
        # This is a simplified extraction - in practice, you'd parse the result more carefully
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(OUTPUT_DIR / f"{prefix}_dataset_{timestamp}.csv")
    
    def _generate_processing_report(self, original_path: str, cleaned_path: str, 
                                  transformed_path: str, cleaning_result: str,
                                  transformation_result: str, analysis_result: str,
                                  summary_result: str) -> Dict:
        """Generate a comprehensive processing report."""
        return {
            "processing_timestamp": datetime.now().isoformat(),
            "original_dataset": original_path,
            "cleaned_dataset": cleaned_path,
            "transformed_dataset": transformed_path,
            "processing_steps": [
                {
                    "step": "data_cleaning",
                    "description": "Clean and standardize data",
                    "result_summary": cleaning_result[:200] + "..." if len(cleaning_result) > 200 else cleaning_result
                },
                {
                    "step": "data_transformation", 
                    "description": "Transform and enrich data",
                    "result_summary": transformation_result[:200] + "..." if len(transformation_result) > 200 else transformation_result
                },
                {
                    "step": "data_analysis",
                    "description": "Analyze data and generate insights",
                    "result_summary": analysis_result[:200] + "..." if len(analysis_result) > 200 else analysis_result
                },
                {
                    "step": "summarization",
                    "description": "Create executive summary",
                    "result_summary": summary_result[:200] + "..." if len(summary_result) > 200 else summary_result
                }
            ],
            "output_files": {
                "cleaned_dataset": cleaned_path,
                "transformed_dataset": transformed_path,
                "analysis_report": "analysis_report.txt",
                "executive_summary": "executive_summary.txt"
            }
        }
    
    def get_dataset_info(self, dataset_path: str) -> Dict:
        """Get basic information about a dataset."""
        try:
            df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_excel(dataset_path)
            
            return {
                "file_name": Path(dataset_path).name,
                "file_size": Path(dataset_path).stat().st_size,
                "shape": df.shape,
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "sample_data": df.head().to_dict()
            }
        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            return {"error": str(e)}
    
    def _create_cleaning_task_with_data_info(self, dataset_path: str):
        """Create cleaning task with actual dataset information injected."""
        from crewai import Task
        
        # Get actual dataset information
        dataset_info = self.get_dataset_info(dataset_path)
        
        # Create enhanced task description
        enhanced_description = f"""
        CRITICAL: You are analyzing the following ACTUAL dataset:
        
        {dataset_info}
        
        Based on this ACTUAL dataset content, clean and standardize it:
        
        1. Data Quality Assessment (based on actual dataset):
           - Identify missing values and their patterns in the actual data
           - Check for inconsistent formats in the actual columns (dates, numbers, text)
           - Find duplicate records in this specific dataset
           - Identify outliers and anomalies in the actual data
           - Check column naming inconsistencies in this dataset
        
        2. Apply cleaning operations (tailored to this dataset):
           - Handle missing values appropriately for the specific variables
           - Standardize date formats if date columns exist in this dataset
           - Convert numeric columns to proper types based on actual data
           - Clean and standardize text data for the actual text columns
           - Remove or flag duplicates found in this dataset
           - Standardize column names (snake_case) for the actual columns
        
        3. Validate data (based on actual content):
           - If geographic data exists, validate district/mandal names
           - If administrative codes exist, validate them
           - If coordinates exist, validate they are within appropriate bounds
           - Validate any other domain-specific constraints for this dataset
        
        4. Generate a cleaning report with:
           - Specific issues found in this dataset and actions taken
           - Data quality metrics before/after for the actual data
           - Recommendations specific to this dataset's data collection
        
        Return the cleaned dataset file path and detailed cleaning report based on actual dataset analysis.
        """
        
        return Task(
            description=enhanced_description,
            agent=self.cleaner.get_agent(),
            expected_output="Cleaned dataset file path and comprehensive cleaning report based on actual dataset analysis"
        )
