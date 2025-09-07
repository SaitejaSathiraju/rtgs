"""Data analysis agent for dynamic data analysis."""

from crewai import Task
from .base_agent import BaseAgent
import pandas as pd
import numpy as np
from typing import Dict, Any, List


class DataAnalystAgent(BaseAgent):
    """Agent responsible for performing deep data analysis."""
    
    def __init__(self):
        super().__init__(
            name="DataAnalyst",
            role="Senior Data Analyst",
            goal="Perform comprehensive statistical and analytical analysis of any dataset to extract actionable insights",
            backstory="""You are a senior data analyst with extensive experience in statistical analysis, 
            data mining, and business intelligence. You have deep knowledge of statistical analysis, 
            demographics, and policy areas. You excel at identifying patterns, trends, correlations, 
            and anomalies in government datasets. You can translate complex analytical findings into 
            clear, actionable insights for policy makers and government officials."""
        )
    
    def create_analysis_task(self, dataset_path: str) -> Task:
        """Create a data analysis task."""
        return Task(
            description=f"""
            You are a Senior Data Analyst. Your task is to perform comprehensive statistical and analytical analysis of the dataset at {dataset_path}.
            
            IMPORTANT: You have access to ONLY these 3 tools:
            1. read_dataset - to read and understand the dataset
            2. analyze_data_quality - to assess data quality issues  
            3. clean_dataset - to clean the dataset if needed
            
            Follow these steps using ONLY the available tools:
            
            1. FIRST, use read_dataset to understand the dataset:
               Action: read_dataset
               Action Input: {{"file_path": "{dataset_path}"}}
            
            2. THEN, use analyze_data_quality to assess data quality:
               Action: analyze_data_quality
               Action Input: {{"file_path": "{dataset_path}"}}
            
            3. FINALLY, provide a comprehensive final answer summarizing:
               - Dataset overview and structure
               - Data quality assessment
               - Statistical findings and patterns
               - Actionable insights and recommendations
               - Policy implications
            
            CRITICAL: After completing the analysis, you MUST provide a Final Answer in this format:
            ```
            Thought: I now know the final answer
            Final Answer: [Your comprehensive analysis report here]
            ```
            
            Do not keep repeating the same tool calls. Complete the analysis and provide the final answer.
            """,
            agent=self.agent,
            expected_output="Comprehensive analysis report with statistical findings, patterns, insights, and recommendations based on actual dataset analysis"
        )
    
    def think_and_act(self, task_description: str, context: Dict[str, Any] = None) -> str:
        """Think and act autonomously for data analysis."""
        
        thinking = f"""
        📊 THINKING PROCESS for DataAnalyst:
        
        Task: {task_description}
        Context: {context or 'No additional context provided'}
        
        As a Senior Data Analyst, I need to:
        1. Perform comprehensive statistical analysis of the dataset
        2. Identify meaningful patterns, trends, and correlations
        3. Detect anomalies and outliers that require attention
        4. Provide actionable insights based on the data
        5. Translate findings into policy recommendations
        
        My analytical capabilities include:
        - Statistical analysis and hypothesis testing
        - Pattern recognition and trend analysis
        - Correlation and regression analysis
        - Time series analysis
        - Geographic and demographic analysis
        - Performance benchmarking
        - Anomaly detection
        
        I will autonomously decide on the most appropriate analytical approaches based on the data characteristics and domain context.
        """
        
        return thinking
