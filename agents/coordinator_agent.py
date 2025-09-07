"""Coordinator agent for orchestrating multi-agent analysis."""

from crewai import Task
from .base_agent import BaseAgent
from typing import Dict, Any, List
import pandas as pd
from pathlib import Path


class CoordinatorAgent(BaseAgent):
    """Main coordinator and task delegator agent."""
    
    def __init__(self):
        super().__init__(
            name="Coordinator",
            role="Analysis Coordinator and Task Delegator",
            goal="Coordinate multi-agent analysis workflows and delegate tasks to specialized agents for comprehensive data analysis",
            backstory="""You are an expert project coordinator with deep understanding of data analysis 
            workflows and multi-agent systems. You excel at breaking down complex analysis tasks, 
            delegating work to specialized agents, coordinating their efforts, and synthesizing 
            results into comprehensive reports. You have extensive experience with data 
            ecosystems and can ensure all analysis components work together seamlessly."""
        )
    
    def create_coordination_task(self, dataset_path: str, analysis_type: str = "comprehensive") -> Task:
        """Create a coordination task for multi-agent analysis."""
        return Task(
            description=f"""
            You are an Analysis Coordinator and Task Delegator. Your task is to coordinate comprehensive multi-agent analysis of the dataset at {dataset_path}.
            
            IMPORTANT: You must use the available tools to perform REAL analysis operations, not just generate text.
            
            Follow these steps using the tools:
            
            1. FIRST, use the read_dataset tool to understand the dataset:
               read_dataset(file_path="{dataset_path}")
            
            2. THEN, use the analyze_data_quality tool to assess the dataset:
               analyze_data_quality_tool(file_path="{dataset_path}")
            
            3. NEXT, use the analyze_statistics tool to understand data patterns:
               analyze_statistics_tool(file_path="{dataset_path}")
            
            4. THEN, use the generate_insights tool to extract insights:
               generate_insights_tool(file_path="{dataset_path}")
            
            5. OPTIONALLY, create visualizations to support coordination:
               create_visualization_tool(file_path="{dataset_path}", chart_type="auto")
            
            6. FINALLY, save your coordination report:
               save_analysis_report_tool(analysis_content="[your coordination report]", dataset_name="[dataset_name]")
            
            Your coordination report should include:
            - Dataset assessment and analysis plan
            - Task delegation strategy for specialized agents
            - Quality assurance measures
            - Integration approach and synthesis methodology
            - Comprehensive analysis results
            - Recommendations for future analysis workflows
            - Path to saved coordination report
            
            Make sure to use the tools to perform actual data operations, not just describe what you would do.
            """,
            agent=self.agent,
            expected_output="Comprehensive coordination report with analysis plan, execution details, and integrated results from all agents"
        )
    
    def think_and_act(self, task_description: str, context: Dict[str, Any] = None) -> str:
        """Think and act autonomously for coordination."""
        
        thinking = f"""
        🎯 THINKING PROCESS for Coordinator:
        
        Task: {task_description}
        Context: {context or 'No additional context provided'}
        
        As an Analysis Coordinator, I need to:
        1. Assess the dataset and determine the best analysis approach
        2. Break down the analysis into specialized tasks for different agents
        3. Coordinate the workflow and ensure proper sequencing
        4. Monitor progress and quality across all agents
        5. Synthesize results into a comprehensive final report
        
        My coordination capabilities include:
        - Multi-agent workflow design and management
        - Task decomposition and delegation
        - Quality assurance and validation
        - Result integration and synthesis
        - Project management and progress tracking
        - Stakeholder communication and reporting
        
        I will autonomously decide on the best coordination strategy based on the dataset characteristics and analysis requirements.
        """
        
        return thinking
    
    def delegate_to_agents(self, dataset_path: str, agents: List[BaseAgent]) -> Dict[str, Any]:
        """Delegate tasks to specialized agents."""
        delegation_plan = {
            "dataset_path": dataset_path,
            "agents_assigned": [],
            "task_sequence": [],
            "expected_outputs": {}
        }
        
        for agent in agents:
            agent_info = {
                "name": agent.name,
                "role": agent.role,
                "task_type": self._determine_task_type(agent.name),
                "priority": self._determine_priority(agent.name)
            }
            delegation_plan["agents_assigned"].append(agent_info)
            delegation_plan["task_sequence"].append(agent.name)
            delegation_plan["expected_outputs"][agent.name] = f"{agent.name} analysis report"
        
        return delegation_plan
    
    def _determine_task_type(self, agent_name: str) -> str:
        """Determine task type based on agent name."""
        task_types = {
            "DataCleaner": "data_preprocessing",
            "DataTransformer": "data_transformation", 
            "DataAnalyst": "statistical_analysis",
            "DataSummarizer": "report_generation",
            "Coordinator": "workflow_coordination"
        }
        return task_types.get(agent_name, "general_analysis")
    
    def _determine_priority(self, agent_name: str) -> int:
        """Determine task priority based on agent name."""
        priorities = {
            "DataCleaner": 1,
            "DataTransformer": 2,
            "DataAnalyst": 3,
            "DataSummarizer": 4,
            "Coordinator": 0
        }
        return priorities.get(agent_name, 5)
