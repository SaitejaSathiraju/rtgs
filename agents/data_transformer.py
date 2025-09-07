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
            
            IMPORTANT: You must use the available tools to perform REAL data transformation operations, not just generate text.
            
            Follow these steps using the tools:
            
            1. FIRST, use the read_dataset tool to understand the dataset structure:
               read_dataset(file_path="{dataset_path}")
            
            2. THEN, use the analyze_statistics tool to understand the data patterns:
               analyze_statistics_tool(file_path="{dataset_path}")
            
            3. NEXT, use the transform_dataset tool to actually transform the data:
               transform_dataset_tool(file_path="{dataset_path}")
            
            4. OPTIONALLY, create visualizations to show the transformation effects:
               create_visualization_tool(file_path="{dataset_path}", chart_type="distribution")
            
            Your output should include:
            - Summary of the original dataset structure
            - Statistical analysis of the data
            - Transformation operations performed
            - New features created
            - Path to the transformed dataset file
            - Visualization files created (if any)
            
            Make sure to use the tools to perform actual data operations, not just describe what you would do.
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
