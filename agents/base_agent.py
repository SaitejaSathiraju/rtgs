"""Base agent class for Real-Time Government System analysis agents."""

from crewai import Agent
from langchain_ollama import OllamaLLM
from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path


class BaseAgent:
    """Base class for all Real-Time Government System analysis agents."""
    
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
        try:
            from crewai_tools import FileReadTool
            from crewai import tool
            
            # Tool 1: Read CSV files
            csv_tool = FileReadTool()
            
            # Tool 2: Analyze dataset - REAL DATA ANALYSIS
            @tool("analyze_dataset")
            def analyze_dataset_tool(file_path: str) -> str:
                """Analyze a CSV file and return comprehensive real data insights."""
                try:
                    import pandas as pd
                    import numpy as np
                    
                    df = pd.read_csv(file_path)
                    
                    analysis = f"""
REAL DATA ANALYSIS REPORT:
==========================

📊 DATASET OVERVIEW:
- File: {file_path}
- Records: {len(df):,}
- Columns: {len(df.columns)}
- Column names: {', '.join(df.columns)}
- Data types: {dict(df.dtypes.value_counts())}
- Missing values: {df.isnull().sum().sum()}

📈 NUMERIC ANALYSIS:
"""
                    
                    # Add detailed statistics for numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        for col in numeric_cols:
                            stats = df[col].describe()
                            analysis += f"- {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}, median={stats['50%']:.2f}\n"
                    
                    # Add categorical analysis
                    analysis += "\n📋 CATEGORICAL ANALYSIS:\n"
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                            value_counts = df[col].value_counts().head(10)
                            analysis += f"- {col} (top 10): {dict(value_counts)}\n"
                    
                    # Add correlation analysis
                    if len(numeric_cols) > 1:
                        analysis += "\n🔗 CORRELATION ANALYSIS:\n"
                        corr_matrix = df[numeric_cols].corr()
                        strong_corr = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_val = corr_matrix.iloc[i, j]
                                if abs(corr_val) > 0.5:
                                    strong_corr.append(f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
                        if strong_corr:
                            analysis += "\n".join(strong_corr[:5]) + "\n"
                    
                    # Add data quality insights
                    analysis += "\n🔍 DATA QUALITY INSIGHTS:\n"
                    analysis += f"- Completeness: {((len(df) * len(df.columns) - df.isnull().sum().sum()) / (len(df) * len(df.columns)) * 100):.1f}%\n"
                    
                    # Check for outliers
                    outlier_cols = []
                    for col in numeric_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                        if len(outliers) > 0:
                            outlier_cols.append(f"{col}: {len(outliers)} outliers")
                    
                    if outlier_cols:
                        analysis += f"- Outliers detected: {', '.join(outlier_cols[:3])}\n"
                    
                    return analysis
                    
                except Exception as e:
                    return f"Error analyzing data: {str(e)}"
            
            return [csv_tool, analyze_dataset_tool]
            
        except Exception as e:
            # Fallback: return empty list if tools can't be created
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
