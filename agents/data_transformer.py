"""Data transformation agent for dynamic data analysis."""

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
            goal="Transform and restructure any dataset into optimal formats for analysis and decision-making",
            backstory="""You are an expert data transformation specialist with deep knowledge of 
            data engineering, feature engineering, and data restructuring techniques. You excel at 
            identifying transformation opportunities, creating derived features, normalizing data, 
            and restructuring datasets for optimal analysis. You understand data transformation patterns 
            and can suggest the best transformations for government datasets."""
        )
    
    def create_transformation_task(self, dataset_path: str) -> Task:
        """Create a data transformation task."""
        return Task(
            description=f"""
            You are a Data Transformation Specialist. Your task is to transform and restructure the dataset at {dataset_path}.
            
            IMPORTANT: You have access to ONLY these 3 tools:
            1. read_dataset - to read and understand the dataset
            2. analyze_data_quality - to assess data quality issues  
            3. clean_dataset - to clean the dataset if needed
            
            Follow these steps using ONLY the available tools:
            
            1. FIRST, use read_dataset to understand the dataset structure:
               Action: read_dataset
               Action Input: {{"file_path": "{dataset_path}"}}
            
            2. THEN, use analyze_data_quality to understand the data patterns:
               Action: analyze_data_quality
               Action Input: {{"file_path": "{dataset_path}"}}
            
            3. FINALLY, provide a comprehensive final answer summarizing:
               - Summary of the original dataset structure
               - Data quality assessment
               - Transformation recommendations
               - New features that could be created
               - Optimal data structure for analysis
            
            CRITICAL: After completing the analysis, you MUST provide a Final Answer in this format:
            ```
            Thought: I now know the final answer
            Final Answer: [Your comprehensive transformation report here]
            ```
            
            Do not keep repeating the same tool calls. Complete the analysis and provide the final answer.
            """,
            agent=self.agent,
            expected_output="Comprehensive transformation report with actual transformed dataset file path and detailed transformation operations performed"
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
