"""Data cleaning agent for Telangana Open Data."""

from crewai import Task
from .base_agent import BaseAgent


class DataCleanerAgent(BaseAgent):
    """Agent responsible for cleaning and standardizing data."""
    
    def __init__(self):
        super().__init__(
            name="DataCleaner",
            role="Data Quality Specialist",
            goal="Clean and standardize Telangana Open Data to ensure high quality and consistency",
            backstory="""You are an expert data quality specialist with deep knowledge of 
            Telangana's administrative structure, demographics, and data standards. 
            You excel at identifying and fixing data quality issues, handling missing values, 
            standardizing formats, and ensuring data consistency across different sources."""
        )
    
    def create_cleaning_task(self, dataset_path: str) -> Task:
        """Create a data cleaning task."""
        return Task(
            description=f"""
            CRITICAL: You must FIRST use the analyze_dataset tool to read and analyze the actual dataset at {dataset_path} to understand its specific structure, columns, data types, and content.
            
            Use this command: analyze_dataset(file_path="{dataset_path}")
            
            After reading the dataset, clean and standardize it based on the ACTUAL data structure (completely dynamic):
            
            1. Data Quality Assessment (100% data-driven):
               - Read the dataset and identify the specific columns: {dataset_path}
               - Identify missing values and their patterns in the ACTUAL data
               - Check for inconsistent formats in the ACTUAL columns (dates, numbers, text)
               - Find duplicate records in this specific dataset
               - Identify outliers and anomalies in the ACTUAL data
               - Check column naming inconsistencies in this dataset
            
            2. Apply cleaning operations (completely adaptive):
               - Handle missing values appropriately for the specific variables found
               - Standardize date formats if date columns exist in this dataset
               - Convert numeric columns to proper types based on actual data
               - Clean and standardize text data for the actual text columns
               - Remove or flag duplicates found in this dataset
               - Standardize column names (snake_case) for the actual columns
            
            3. Validate data (based on actual content):
               - If geographic data exists, validate location names
               - If administrative codes exist, validate them
               - If coordinates exist, validate they are within appropriate bounds
               - Validate any other domain-specific constraints for this dataset
               - Adapt validation to whatever data types and structures are found
            
            4. Generate a cleaning report with REAL DATA:
               - Specific issues found in this dataset and actions taken
               - Data quality metrics before/after for the actual data
               - Recommendations specific to this dataset's data collection
               - Use actual numbers and percentages from the data
            
            CRITICAL REQUIREMENTS:
            - Use ONLY the actual data insights from the analyze_dataset tool
            - Do NOT use placeholder text or predefined templates
            - Do NOT make assumptions about the data structure
            - Adapt completely to whatever dataset structure is found
            - Provide specific cleaning recommendations with real numbers
            
            Return the cleaned dataset file path and detailed cleaning report based on actual dataset analysis.
            """,
            agent=self.agent,
            expected_output="Cleaned dataset file path and comprehensive cleaning report based on actual dataset analysis"
        )
