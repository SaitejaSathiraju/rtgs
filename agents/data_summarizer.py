"""Data summarization agent for dynamic data analysis."""

from crewai import Task
from .base_agent import BaseAgent


class DataSummarizerAgent(BaseAgent):
    """Agent responsible for summarizing analysis results."""
    
    def __init__(self):
        super().__init__(
            name="DataSummarizer",
            role="Executive Summary Specialist",
            goal="Create clear, actionable summaries of any dataset analysis for decision makers",
            backstory="""You are an expert communicator who specializes in translating 
            complex data analysis into clear, actionable insights for government officials 
            and policy makers. You have deep understanding of data analysis and reporting 
            structure and can effectively communicate technical findings to non-technical 
            audiences while maintaining accuracy and highlighting critical information."""
        )
    
    def create_summarization_task(self, dataset_path: str) -> Task:
        """Create a data summarization task."""
        return Task(
            description=f"""
            CRITICAL: You must FIRST use the analyze_dataset tool to read and analyze the actual dataset at {dataset_path} to understand its specific structure, columns, data types, and content.
            
            Use this command: analyze_dataset(file_path="{dataset_path}")
            
            After analyzing the ACTUAL dataset, create a comprehensive executive summary that is COMPLETELY DYNAMIC and based on the real data:
            
            1. EXECUTIVE SUMMARY (100% data-driven):
               - Extract ONLY the key findings from the actual dataset analysis
               - Identify the most critical insights based on the ACTUAL data patterns
               - Determine urgent actions based on the ACTUAL findings from the data
               - Highlight positive developments found in the ACTUAL data
            
            2. POLICY RECOMMENDATIONS (completely adaptive):
               - Create specific recommendations based ONLY on the ACTUAL analysis findings
               - Estimate impact based on the ACTUAL data insights and numbers
               - Suggest implementation timeline based on the ACTUAL findings
               - Define success metrics relevant to the analyzed dataset
            
            3. DATA QUALITY ASSESSMENT (based on actual analysis):
               - Extract the ACTUAL data quality score from the analysis
               - Identify the specific data quality issues found in the analysis
               - Recommend improvements based on the ACTUAL data quality findings
               - Assess confidence level based on the actual analysis
            
            4. IMMEDIATE ACTIONS (based on actual findings):
               - Suggest immediate actions based on the ACTUAL findings
               - Recommend follow-up analysis based on the ACTUAL data gaps
               - Suggest additional data sources relevant to this dataset's domain
               - Identify stakeholders relevant to this dataset's domain
            
            5. Create presentation-ready summary with REAL DATA ONLY:
               - Use clear, non-technical language
               - Include specific numbers and percentages from the ACTUAL analysis
               - Highlight context specific to the analyzed dataset's domain
               - Focus on actionable insights based on ACTUAL findings
               - Include visual recommendations relevant to the actual data
               - Provide concrete policy recommendations with specific numbers
            
            CRITICAL REQUIREMENTS:
            - Do NOT use placeholder text like "data would go here" or "general analysis"
            - Do NOT make assumptions about the data structure
            - Do NOT use predefined templates
            - Use ONLY the ACTUAL data insights from the analyze_dataset tool
            - Adapt completely to whatever dataset structure is found
            - Provide specific, actionable recommendations with real numbers
            
            Return a comprehensive executive summary based on the ACTUAL dataset analysis findings.
            """,
            agent=self.agent,
            expected_output="Executive summary with REAL key findings, specific recommendations, and actionable insights based on actual dataset analysis"
        )
