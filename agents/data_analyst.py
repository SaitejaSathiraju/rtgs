"""Data analysis agent for Telangana Open Data."""

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
            goal="Perform comprehensive statistical and analytical analysis of Telangana Open Data to extract actionable insights",
            backstory="""You are a senior data analyst with extensive experience in statistical analysis, 
            data mining, and business intelligence. You have deep knowledge of Telangana's governance, 
            demographics, and policy areas. You excel at identifying patterns, trends, correlations, 
            and anomalies in government datasets. You can translate complex analytical findings into 
            clear, actionable insights for policy makers and government officials."""
        )
    
    def create_analysis_task(self, dataset_path: str) -> Task:
        """Create a data analysis task."""
        return Task(
            description=f"""
            CRITICAL: You must FIRST use the analyze_dataset tool to read and analyze the actual dataset at {dataset_path} to understand its specific structure, columns, data types, and content.
            
            Use this command: analyze_dataset(file_path="{dataset_path}")
            
            After analyzing the ACTUAL dataset, perform comprehensive analysis that is COMPLETELY DYNAMIC and adaptive:
            
            1. Statistical Analysis (100% data-driven):
               - Use the analyze_dataset tool to get detailed statistics: {dataset_path}
               - Calculate measures of central tendency for ALL numeric columns found
               - Analyze distributions and identify patterns in the ACTUAL data
               - Perform correlation analysis between ALL numeric variables
               - Identify outliers and anomalies in the ACTUAL data
               - Calculate confidence intervals and statistical significance
            
            2. Pattern Recognition (completely adaptive):
               - Identify trends in the actual data over time (if temporal data exists)
               - Find seasonal patterns or cyclical behavior in the ACTUAL data
               - Detect clusters or groupings in the ACTUAL data
               - Identify relationships between different variables in the ACTUAL data
               - Find unusual patterns or anomalies in the ACTUAL data
            
            3. Dynamic Analysis (based on actual content):
               - Analyze geographic patterns if location data exists
               - Perform demographic analysis if population data exists
               - Analyze economic indicators if financial data exists
               - Perform performance analysis if metrics data exists
               - Analyze policy impact if policy-related data exists
               - Adapt analysis to whatever domain the data represents
            
            4. Comparative Analysis (tailored to the dataset):
               - Compare different categories or groups in the ACTUAL data
               - Analyze performance across different regions/time periods
               - Identify best and worst performers in the ACTUAL data
               - Calculate growth rates and changes over time
               - Perform benchmarking analysis
            
            5. Predictive Insights (based on actual patterns):
               - Identify factors that influence key outcomes in the ACTUAL data
               - Suggest predictive models based on the ACTUAL data
               - Identify leading indicators in the ACTUAL dataset
               - Recommend forecasting approaches
            
            6. Generate comprehensive analysis report with REAL DATA ONLY:
               - Key statistical findings from the actual data
               - Important patterns and trends identified in the ACTUAL data
               - Significant correlations and relationships found
               - Anomalies and outliers found in the ACTUAL data
               - Actionable insights and recommendations based on ACTUAL findings
               - Policy implications based on the ACTUAL analysis
            
            CRITICAL REQUIREMENTS:
            - Use ONLY the actual data insights from the analyze_dataset tool
            - Do NOT use placeholder text or predefined templates
            - Do NOT make assumptions about the data structure
            - Adapt completely to whatever dataset structure is found
            - Provide specific insights with real numbers from the data
            
            Return comprehensive analysis report with statistical findings, patterns, insights, and recommendations based on actual dataset analysis.
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
