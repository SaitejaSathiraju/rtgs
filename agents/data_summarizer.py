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
            
            IMPORTANT: You must use the available tools to perform REAL analysis operations, not just generate text.
            
            Follow these steps using the tools:
            
            1. FIRST, use the read_dataset tool to understand the dataset:
               read_dataset(file_path="{dataset_path}")
            
            2. THEN, use the analyze_statistics tool to get key findings:
               analyze_statistics_tool(file_path="{dataset_path}")
            
            3. NEXT, use the generate_insights tool to extract actionable insights:
               generate_insights_tool(file_path="{dataset_path}")
            
            4. OPTIONALLY, create visualizations to support your summary:
               create_visualization_tool(file_path="{dataset_path}", chart_type="auto")
            
            5. FINALLY, save your executive summary:
               save_analysis_report_tool(analysis_content="[your executive summary]", dataset_name="[dataset_name]")
            
            Your executive summary should include:
            - Executive overview of key findings
            - Critical insights and patterns
            - Policy recommendations
            - Implementation roadmap
            - Risk assessment
            - Success metrics
            - Path to saved summary report
            
            Make sure to use the tools to perform actual data operations, not just describe what you would do.
            """,
            agent=self.agent,
            expected_output="Executive summary with key findings, specific recommendations, and actionable insights based on actual dataset analysis"
        )
