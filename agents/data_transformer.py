"""Data transformation agent for Telangana Open Data."""

from crewai import Task
from .base_agent import BaseAgent
import pandas as pd
from typing import Dict, Any, List


class DataTransformerAgent(BaseAgent):
    """Agent responsible for transforming and restructuring data."""
    
    def __init__(self):
        super().__init__(
            name="DataTransformer",
            role="Data Transformation Specialist",
            goal="Transform and restructure Telangana Open Data into optimal formats for analysis and decision-making",
            backstory="""You are an expert data transformation specialist with deep knowledge of 
            data engineering, feature engineering, and data restructuring techniques. You excel at 
            identifying transformation opportunities, creating derived features, normalizing data, 
            and restructuring datasets for optimal analysis. You understand Telangana's data patterns 
            and can suggest the best transformations for government datasets."""
        )
    
    def create_transformation_task(self, dataset_path: str) -> Task:
        """Create a data transformation task."""
        return Task(
            description=f"""
            CRITICAL: You must FIRST use the analyze_dataset tool to read and analyze the actual dataset at {dataset_path} to understand its specific structure, columns, data types, and content.
            
            Use this command: analyze_dataset(file_path="{dataset_path}")
            
            After reading the dataset, transform and restructure it based on the ACTUAL data (completely dynamic):
            
            1. Data Structure Analysis (100% data-driven):
               - Read the dataset and identify the specific columns: {dataset_path}
               - Analyze data types and identify transformation opportunities
               - Identify categorical variables that need encoding
               - Find numeric variables that need normalization/scaling
               - Identify date/time columns that need parsing
               - Check for hierarchical or nested data structures
            
            2. Feature Engineering (completely adaptive):
               - Create derived features based on the actual data patterns
               - Generate aggregations relevant to the specific domain found
               - Create categorical encodings for the actual categorical variables
               - Generate time-based features if date columns exist
               - Create geographic features if location data exists
               - Generate statistical features for numeric columns
            
            3. Data Restructuring (based on actual content):
               - Reshape data if needed (wide to long, long to wide)
               - Create pivot tables relevant to the domain
               - Generate summary tables for the actual data
               - Create hierarchical groupings if applicable
               - Restructure for optimal analysis based on the data
            
            4. Data Normalization (tailored to the dataset):
               - Apply appropriate scaling to numeric columns
               - Normalize categorical variables
               - Handle outliers in the actual data
               - Apply domain-specific transformations
            
            5. Generate transformation report with REAL DATA:
               - Specific transformations applied to this dataset
               - New features created and their purposes
               - Data structure changes made
               - Recommendations for further transformations
               - Use actual numbers and percentages from the data
            
            CRITICAL REQUIREMENTS:
            - Use ONLY the actual data insights from the analyze_dataset tool
            - Do NOT use placeholder text or predefined templates
            - Do NOT make assumptions about the data structure
            - Adapt completely to whatever dataset structure is found
            - Provide specific transformation recommendations with real numbers
            
            Return the transformed dataset file path and comprehensive transformation report based on actual dataset analysis.
            """,
            agent=self.agent,
            expected_output="Transformed dataset file path and comprehensive transformation report based on actual dataset analysis"
        )
    
    def think_and_act(self, task_description: str, context: Dict[str, Any] = None) -> str:
        """Think and act autonomously for data transformation."""
        
        thinking = f"""
        🔄 THINKING PROCESS for DataTransformer:
        
        Task: {task_description}
        Context: {context or 'No additional context provided'}
        
        As a Data Transformation Specialist, I need to:
        1. Analyze the dataset structure and identify transformation opportunities
        2. Consider what transformations would be most valuable for this specific data
        3. Apply feature engineering techniques appropriate to the domain
        4. Restructure data for optimal analysis
        5. Create derived features that add analytical value
        
        My expertise includes:
        - Feature engineering and creation
        - Data normalization and scaling
        - Categorical encoding
        - Time series transformations
        - Geographic data processing
        - Statistical feature generation
        
        I will autonomously decide on the best transformation approach based on the actual data characteristics.
        """
        
        return thinking
