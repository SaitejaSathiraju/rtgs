"""Base agent class for Telangana Open Data analysis agents."""

from crewai import Agent
from langchain_ollama import OllamaLLM
from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path


class BaseAgent:
    """Base class for all Telangana Open Data analysis agents."""
    
    def __init__(self, name: str, role: str, goal: str, backstory: str):
        """Initialize the base agent."""
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        
        # Initialize Ollama LLM with proper format for CrewAI
        self.llm = OllamaLLM(
            model=f"ollama/{OLLAMA_MODEL}",
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            timeout=120
        )
        
        # Create CrewAI agent
        self.agent = Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=self._get_tools()
        )
    
    def _get_tools(self) -> List:
        """Get tools available to this agent. Override in subclasses."""
        # For now, return empty list - tools will be added later
        return []
    
    def get_agent(self) -> Agent:
        """Get the CrewAI agent instance."""
        return self.agent
    
    def think_and_act(self, task_description: str, context: Dict[str, Any] = None) -> str:
        """Main method for agents to think and take autonomous action."""
        # This is where the agent's autonomous thinking happens
        # The agent will analyze the task, think about the best approach,
        # and take appropriate actions
        
        thinking_process = f"""
        🤔 THINKING PROCESS for {self.name}:
        
        Task: {task_description}
        Context: {context or 'No additional context provided'}
        
        As {self.role}, I need to:
        1. Analyze the task requirements
        2. Consider the best approach based on my expertise
        3. Take autonomous actions to complete the task
        4. Provide comprehensive results
        
        My expertise: {self.backstory}
        My goal: {self.goal}
        """
        
        return thinking_process
    
    def analyze_dataset(self, file_path: str) -> Dict[str, Any]:
        """Analyze a dataset file."""
        try:
            df = pd.read_csv(file_path)
            return {
                "success": True,
                "records": len(df),
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "sample_data": df.head(3).to_dict('records')
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_output(self, content: str, filename: str, output_dir: str = "output") -> str:
        """Save agent output to file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        file_path = output_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(file_path)
    
    def log_action(self, action: str, details: str = ""):
        """Log agent actions."""
        print(f"🤖 {self.name}: {action}")
        if details:
            print(f"   Details: {details}")
