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
            You are an Executive Summary Specialist. Your task is to create a clear, actionable summary of the analysis for the dataset at {dataset_path}.
            
            IMPORTANT: You have access to ONLY these 3 tools:
            1. read_dataset - to read and understand the dataset
            2. analyze_data_quality - to assess data quality issues  
            3. clean_dataset - to clean the dataset if needed
            
            Follow these steps using ONLY the available tools:
            
            1. FIRST, use read_dataset to understand the dataset:
               Action: read_dataset
               Action Input: {{"file_path": "{dataset_path}"}}
            
            2. THEN, use analyze_data_quality to get key findings:
               Action: analyze_data_quality
               Action Input: {{"file_path": "{dataset_path}"}}
            
            3. FINALLY, provide a comprehensive final answer summarizing:
               - Executive overview of key findings
               - Critical insights and patterns
               - Policy recommendations
               - Implementation roadmap
               - Risk assessment
               - Success metrics
            
            CRITICAL: After completing the analysis, you MUST provide a Final Answer in this format:
            ```
            Thought: I now know the final answer
            Final Answer: [Your comprehensive executive summary here]
            ```
            
            Do not keep repeating the same tool calls. Complete the analysis and provide the final answer.
            """,
            agent=self.agent,
            expected_output="Executive summary with key findings, specific recommendations, and actionable insights based on actual dataset analysis"
        )
