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
            
            IMPORTANT: You have access to ONLY these 3 tools:
            1. read_dataset - to read and understand the dataset
            2. analyze_data_quality - to assess data quality issues  
            3. clean_dataset - to clean the dataset
            
            Follow these steps using ONLY the available tools:
            
            1. FIRST, use read_dataset to understand the dataset:
               Action: read_dataset
               Action Input: {{"file_path": "{dataset_path}"}}
            
            2. THEN, use analyze_data_quality to identify quality issues:
               Action: analyze_data_quality
               Action Input: {{"file_path": "{dataset_path}"}}
            
            3. NEXT, use clean_dataset to clean the data (use null for output_path to auto-generate filename):
               Action: clean_dataset
               Action Input: {{"file_path": "{dataset_path}", "output_path": null}}
            
            4. FINALLY, provide a comprehensive final answer summarizing:
               - Summary of the original dataset
               - Data quality issues identified
               - Cleaning operations performed
               - Results of the cleaning process
               - Path to the cleaned dataset file
            
            CRITICAL: After completing the analysis, you MUST provide a Final Answer in this format:
            ```
            Thought: I now know the final answer
            Final Answer: [Your comprehensive cleaning report here]
            ```
            
            Do not keep repeating the same tool calls. Complete the analysis and provide the final answer.
            """,
            agent=self.agent,
            expected_output="Comprehensive cleaning report with actual cleaned dataset file path and detailed cleaning operations performed"
        )
