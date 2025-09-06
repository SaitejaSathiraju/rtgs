#!/usr/bin/env python3
"""
Real-Time Q&A Bot for Dataset Analysis
Answers questions about analyzed datasets using actual data insights.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import re
from langchain_ollama import OllamaLLM
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

class QABot:
    """Real-time Q&A bot for dataset analysis."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.console = Console()
        self.datasets = {}
        self.analysis_cache = {}
        
        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )
        
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset from file."""
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            self.console.print(f"[red]Error loading dataset: {str(e)}[/red]")
            return None
    
    def analyze_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Analyze dataset and cache results."""
        if dataset_name in self.analysis_cache:
            return self.analysis_cache[dataset_name]
        
        analysis = {
            "name": dataset_name,
            "records": len(df),
            "columns": list(df.columns),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "stats": {},
            "insights": {}
        }
        
        # Statistical analysis
        for col in analysis["numeric_columns"]:
            analysis["stats"][col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "sum": float(df[col].sum()),
                "zero_count": int((df[col] == 0).sum()),
                "missing_count": int(df[col].isnull().sum())
            }
        
        # Categorical analysis
        for col in analysis["categorical_columns"]:
            value_counts = df[col].value_counts().head(10)
            analysis["insights"][col] = {
                "top_values": value_counts.to_dict(),
                "unique_count": df[col].nunique(),
                "most_common": value_counts.index[0] if len(value_counts) > 0 else None
            }
        
        # Domain-specific insights
        analysis["domain"] = self._detect_domain(df)
        analysis["insights"]["domain_insights"] = self._generate_domain_insights(df, analysis["domain"])
        
        self.analysis_cache[dataset_name] = analysis
        return analysis
    
    def _detect_domain(self, df: pd.DataFrame) -> str:
        """Detect the domain of the dataset."""
        columns_str = ' '.join(df.columns).lower()
        
        if any(word in columns_str for word in ['rain', 'rainfall', 'humidity', 'weather']):
            return "rainfall_weather"
        elif any(word in columns_str for word in ['consumption', 'units', 'load', 'power', 'electricity']):
            return "utility_consumption"
        elif any(word in columns_str for word in ['health', 'hospital', 'medical']):
            return "healthcare"
        elif any(word in columns_str for word in ['education', 'school', 'student']):
            return "education"
        elif any(word in columns_str for word in ['revenue', 'income', 'gdp', 'economic']):
            return "economic"
        else:
            return "general"
    
    def _generate_domain_insights(self, df: pd.DataFrame, domain: str) -> Dict[str, Any]:
        """Generate domain-specific insights."""
        insights = {}
        
        if domain == "rainfall_weather":
            rain_col = None
            for col in df.columns:
                if 'rain' in col.lower():
                    rain_col = col
                    break
            
            if rain_col:
                rain_data = df[rain_col]
                insights = {
                    "total_rainfall": float(rain_data.sum()),
                    "average_rainfall": float(rain_data.mean()),
                    "zero_rainfall_days": int((rain_data == 0).sum()),
                    "peak_rainfall": float(rain_data.max()),
                    "drought_percentage": float((rain_data == 0).sum() / len(df) * 100),
                    "extreme_rainfall_days": int((rain_data > 50).sum())
                }
            else:
                insights = {}
        
        elif domain == "utility_consumption":
            units_col = None
            for col in df.columns:
                if 'units' in col.lower():
                    units_col = col
                    break
            
            if units_col:
                units_data = df[units_col]
                insights = {
                    "total_consumption": float(units_data.sum()),
                    "average_consumption": float(units_data.mean()),
                    "zero_consumption_locations": int((units_data == 0).sum()),
                    "peak_consumption": float(units_data.max()),
                    "revenue_loss_locations": int((units_data == 0).sum()),
                    "high_consumption_locations": int((units_data > units_data.mean() * 2).sum())
                }
        
        return insights
    
    def answer_question(self, question: str, dataset_name: str = None) -> str:
        """Answer a question about the dataset."""
        if not self.datasets:
            return "❌ No datasets loaded. Please load a dataset first."
        
        if dataset_name and dataset_name not in self.datasets:
            return f"❌ Dataset '{dataset_name}' not found. Available datasets: {list(self.datasets.keys())}"
        
        # If no specific dataset mentioned, find the best match based on question
        if not dataset_name:
            dataset_name = self._find_best_dataset(question)
        
        df = self.datasets[dataset_name]
        analysis = self.analyze_dataset(df, dataset_name)
        
        # Parse question and generate answer
        answer = self._generate_answer(question, analysis, df)
        return answer
    
    def _find_best_dataset(self, question: str) -> str:
        """Find the best dataset based on the question."""
        question_lower = question.lower()
        
        # Check for district-related questions - prioritize rainfall data
        if any(word in question_lower for word in ['district', 'districts', 'mandal', 'mandals']):
            for name, df in self.datasets.items():
                if 'rain' in name.lower() or 'District' in df.columns:
                    return name
        
        # Check for rainfall-related keywords
        elif any(word in question_lower for word in ['rain', 'rainfall', 'drought', 'humidity', 'weather']):
            for name, df in self.datasets.items():
                if 'rain' in name.lower() or 'rain' in ' '.join(df.columns).lower():
                    return name
        
        # Check for consumption-related keywords
        elif any(word in question_lower for word in ['consumption', 'units', 'power', 'electricity', 'outage']):
            for name, df in self.datasets.items():
                if 'consumption' in name.lower() or 'units' in ' '.join(df.columns).lower():
                    return name
        
        # Default to first dataset
        return list(self.datasets.keys())[0]
    
    def _generate_answer(self, question: str, analysis: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate answer based on question and analysis - 100% Ollama powered."""
        # ALWAYS use Ollama - no predefined responses
        return self._answer_with_ollama(question, analysis, df)
    
    
    def _answer_with_ollama(self, question: str, analysis: Dict[str, Any], df: pd.DataFrame) -> str:
        """Answer complex questions using Ollama LLM - 100% data-driven."""
        try:
            # Prepare comprehensive context for the LLM
            context = f"""
DATASET ANALYSIS CONTEXT:
========================

Dataset Name: {analysis['name']}
Total Records: {analysis['records']:,}
Columns: {', '.join(df.columns)}
Domain: {analysis['domain']}

COLUMN DETAILS:
- Numeric Columns: {', '.join(analysis['numeric_columns'])}
- Categorical Columns: {', '.join(analysis['categorical_columns'])}

STATISTICAL SUMMARY:
"""
            
            # Add detailed statistics for each numeric column
            for col in analysis['numeric_columns']:
                stats = analysis['stats'][col]
                context += f"- {col}: Mean={stats['mean']:.2f}, Max={stats['max']:.2f}, Min={stats['min']:.2f}, Sum={stats['sum']:,.2f}, Zeros={stats['zero_count']:,}\n"
            
            # Add categorical insights
            context += "\nCATEGORICAL INSIGHTS:\n"
            for col in analysis['categorical_columns']:
                if col in analysis['insights']:
                    insights = analysis['insights'][col]
                    context += f"- {col}: {insights['unique_count']} unique values, Most common: {insights['most_common']}\n"
            
            # Add sample data
            context += f"\nSAMPLE DATA (first 5 rows):\n{df.head(5).to_string()}\n"
            
            # Add comprehensive data analysis
            context += f"\nCOMPREHENSIVE DATA ANALYSIS:\n"
            
            # Add unique values for categorical columns
            for col in analysis['categorical_columns']:
                unique_vals = df[col].unique()
                if len(unique_vals) <= 50:  # Only show if not too many
                    context += f"- {col} unique values: {', '.join(map(str, unique_vals))}\n"
                else:
                    context += f"- {col}: {len(unique_vals)} unique values (too many to list)\n"
            
            # Add groupby analysis for key columns
            if 'District' in df.columns:
                context += f"\nDISTRICT ANALYSIS:\n"
                district_counts = df['District'].value_counts()
                context += f"- Total districts: {len(district_counts)}\n"
                context += f"- District list: {', '.join(district_counts.index.tolist())}\n"
                
                if 'Rain (mm)' in df.columns:
                    district_rainfall = df.groupby('District')['Rain (mm)'].sum().sort_values(ascending=False)
                    context += f"- Top 5 districts by total rainfall:\n"
                    for i, (district, rainfall) in enumerate(district_rainfall.head(5).items(), 1):
                        context += f"  {i}. {district}: {rainfall:.1f} mm\n"
            
            # Add domain-specific insights if available
            if 'domain_insights' in analysis['insights']:
                context += f"\nDOMAIN-SPECIFIC INSIGHTS:\n"
                for key, value in analysis['insights']['domain_insights'].items():
                    context += f"- {key}: {value}\n"
            
            context += f"\nQUESTION: {question}\n\n"
            context += """
INSTRUCTIONS:
- Analyze the actual data provided above
- Give specific, data-driven answers
- Use exact numbers from the statistics
- Provide insights based on the actual dataset
- Be precise and factual
- If asking about specific values, calculate them from the data
- No generic responses - everything must be based on the actual data
"""
            
            # Use Ollama to generate answer
            response = self.llm.invoke(context)
            
            # Add chart links to the response
            chart_links = self._get_chart_links(analysis['name'])
            if chart_links:
                response += f"\n\n📊 **Available Charts**:\n{chart_links}"
            
            return f"🤖 **RTGS AI Analysis**: {response}"
            
        except Exception as e:
             # Fallback to actual data analysis if Ollama fails
            print(f"Ollama error: {e}")  # Debug print
            
            # Provide actual data analysis as fallback
            if 'visitor' in question.lower() or 'place' in question.lower():
                if 'Place' in df.columns and 'Visitors' in df.columns:
                    top_places = df.groupby('Place')['Visitors'].sum().sort_values(ascending=False).head(5)
                    answer = "🏆 **Top 5 Places by Total Visitors**:\n"
                    for i, (place, visitors) in enumerate(top_places.items(), 1):
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
                        answer += f"{medal} {place}: {visitors:,} visitors\n"
                    return f"🤖 **RTGS Analysis**: {answer}"
            
            # Generic fallback with actual data info
            return f"📊 **RTGS Dataset Analysis**: {analysis['records']:,} records with {len(df.columns)} columns. Columns: {', '.join(df.columns)}. Ask me specific questions about the data!"
    
    def _get_chart_links(self, dataset_name: str) -> str:
        """Get available chart links for the dataset."""
        charts_dir = self.output_dir / "charts"
        if not charts_dir.exists():
            return ""
        
        chart_files = list(charts_dir.glob(f"{dataset_name}*.png"))
        if not chart_files:
            return ""
        
        links = []
        for chart_file in chart_files:
            chart_type = chart_file.stem.replace(f"{dataset_name}_", "").split("_")[0]
            links.append(f"• {chart_type.title()} Chart: {chart_file.name}")
        
        return "\n".join(links)
    
    def _get_domain_description(self, domain: str) -> str:
        """Get description for domain."""
        descriptions = {
            "rainfall_weather": "Weather and rainfall data analysis",
            "utility_consumption": "Power and utility consumption data",
            "healthcare": "Healthcare and medical data",
            "education": "Education and academic data",
            "economic": "Economic and financial data",
            "general": "General data analysis"
        }
        return descriptions.get(domain, "Unknown domain")
    
    def interactive_mode(self):
        """Start interactive Q&A mode."""
        self.console.print(Panel.fit(
            "[bold blue]🤖 Real-Time Q&A Bot[/bold blue]\n"
            "Ask questions about your datasets!\n"
            "Type 'quit' to exit, 'help' for examples",
            title="💬 Interactive Mode",
            border_style="blue"
        ))
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    self.console.print("[yellow]👋 Goodbye![/yellow]")
                    break
                
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                if not question:
                    continue
                
                answer = self.answer_question(question)
                self.console.print(f"\n🤖 **Answer**: {answer}")
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]❌ Error: {str(e)}[/red]")
    
    def _show_help(self):
        """Show help examples."""
        examples = [
            "📊 **Basic Questions**:",
            "  • How many records are there?",
            "  • What columns are in the dataset?",
            "  • What is the average rainfall?",
            "",
            "🌧️ **Rainfall Questions**:",
            "  • How many drought days are there?",
            "  • What is the total rainfall?",
            "  • Which district has the highest rainfall?",
            "",
            "⚡ **Consumption Questions**:",
            "  • How many locations have zero consumption?",
            "  • What is the total consumption?",
            "  • Which circle has the most connections?",
            "",
            "🔗 **Analysis Questions**:",
            "  • What is the correlation between rainfall and humidity?",
            "  • Show me the top 5 districts",
            "  • What are the statistics for units?"
        ]
        
        self.console.print("\n".join(examples))


def main():
    """Main function to run the Q&A bot."""
    bot = QABot()
    
    # Load available datasets
    data_files = list(bot.data_dir.glob("*.csv")) + list(bot.data_dir.glob("*.xlsx"))
    
    if not data_files:
        bot.console.print("[red]❌ No datasets found in data directory[/red]")
        return
    
    bot.console.print(f"[green]📁 Found {len(data_files)} datasets:[/green]")
    for i, file_path in enumerate(data_files, 1):
        bot.console.print(f"  {i}. {file_path.name}")
    
    # Load first dataset
    first_dataset = data_files[0]
    bot.console.print(f"\n[blue]📊 Loading dataset: {first_dataset.name}[/blue]")
    
    df = bot.load_dataset(str(first_dataset))
    if df is not None:
        bot.datasets[first_dataset.stem] = df
        bot.console.print(f"[green]✅ Dataset loaded: {len(df):,} records, {len(df.columns)} columns[/green]")
        
        # Start interactive mode
        bot.interactive_mode()
    else:
        bot.console.print("[red]❌ Failed to load dataset[/red]")


if __name__ == "__main__":
    main()
