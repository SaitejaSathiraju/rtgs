"""Data cleaning agent for dynamic data analysis."""

from crewai import Task
from .base_agent import BaseAgent


class DataCleanerAgent(BaseAgent):
    """Agent responsible for cleaning and standardizing data."""
    
    def __init__(self):
        super().__init__(
            name="DataCleaner",
            role="Data Quality Specialist",
            goal="Clean and standardize any dataset to ensure high quality and consistency",
            backstory="""You are an expert data quality specialist with deep knowledge of 
            data quality standards, cleaning techniques, and best practices. 
            You excel at identifying and fixing data quality issues, handling missing values, 
            standardizing formats, and ensuring data consistency across different sources."""
        )
    
    def create_cleaning_task(self, dataset_path: str) -> Task:
        """Create a data cleaning task."""
        return Task(
            description=f"""
            You are a Data Quality Specialist. Your task is to clean and standardize the dataset at {dataset_path}.
            
            IMPORTANT: You must use the available tools to perform REAL data cleaning operations, not just generate text.
            
            Follow these steps using the tools:
            
            1. FIRST, use the read_dataset tool to understand the dataset:
               read_dataset(file_path="{dataset_path}")
            
            2. THEN, use the analyze_data_quality tool to identify quality issues:
               analyze_data_quality_tool(file_path="{dataset_path}")
            
            3. NEXT, use the clean_dataset tool to actually clean the data:
               clean_dataset_tool(file_path="{dataset_path}")
            
            4. FINALLY, verify the cleaning by analyzing the cleaned dataset again.
            
            Your output should include:
            - Summary of the original dataset
            - Data quality issues identified
            - Cleaning operations performed
            - Results of the cleaning process
            - Path to the cleaned dataset file
            
            Make sure to use the tools to perform actual data operations, not just describe what you would do.
            """,
            agent=self.agent,
            expected_output="Comprehensive cleaning report with actual cleaned dataset file path and detailed cleaning operations performed"
        )
