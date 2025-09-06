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
            goal="Coordinate multi-agent analysis workflows and delegate tasks to specialized agents for comprehensive Telangana Open Data analysis",
            backstory="""You are an expert project coordinator with deep understanding of data analysis 
            workflows and multi-agent systems. You excel at breaking down complex analysis tasks, 
            delegating work to specialized agents, coordinating their efforts, and synthesizing 
            results into comprehensive reports. You have extensive experience with Telangana's 
            data ecosystem and can ensure all analysis components work together seamlessly."""
        )
    
    def create_coordination_task(self, dataset_path: str, analysis_type: str = "comprehensive") -> Task:
        """Create a coordination task for multi-agent analysis."""
        return Task(
            description=f"""
            CRITICAL: You must FIRST use the analyze_dataset tool to read and analyze the actual dataset at {dataset_path} to understand its specific structure, columns, data types, and content.
            
            Use this command: analyze_dataset(file_path="{dataset_path}")
            
            After reading the dataset, coordinate a comprehensive multi-agent analysis:
            
            1. Dataset Assessment and Planning (based on actual dataset):
               - Read the dataset and understand its structure: {dataset_path}
               - Assess the dataset complexity and analysis requirements
               - Determine which agents are needed for this specific dataset
               - Create a detailed analysis plan based on the actual data
               - Identify potential challenges and mitigation strategies
            
            2. Task Delegation Strategy (tailored to this dataset):
               - Delegate data cleaning to DataCleaner based on actual data quality issues
               - Assign transformation tasks to DataTransformer based on actual data structure
               - Coordinate analysis tasks with DataAnalyst based on actual data patterns
               - Plan summarization with DataSummarizer based on expected insights
               - Ensure proper sequencing and dependencies between tasks
            
            3. Quality Assurance Coordination (based on actual content):
               - Establish quality checkpoints for each agent's work
               - Define validation criteria based on the dataset domain
               - Plan cross-validation between agent outputs
               - Ensure consistency across all agent analyses
               - Monitor progress and adjust plans as needed
            
            4. Result Synthesis and Integration (tailored to the analysis):
               - Collect and integrate outputs from all agents
               - Resolve any conflicts or inconsistencies between agent findings
               - Create a unified analysis narrative
               - Ensure all insights are properly contextualized
               - Validate the completeness of the analysis
            
            5. Generate coordination report with:
               - Analysis plan and execution strategy
               - Task delegation details and rationale
               - Quality assurance measures implemented
               - Integration approach and synthesis methodology
               - Final comprehensive analysis results
               - Recommendations for future analysis workflows
            
            Return comprehensive coordination report with analysis plan, execution details, and integrated results from all agents.
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
