#!/usr/bin/env python3
"""
100% Ollama-Powered RAG System for Government Data
Zero predefined responses - Everything handled by Ollama with dataset as RAG knowledge base
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
import chromadb
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class OllamaRAGSystem:
    """100% Ollama-powered RAG system with zero predefined responses."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.console = Console()
        
        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )
        
        # Initialize RAG components
        self.vector_db = None
        self.embedding_model = None
        self.datasets_info = {}
        self.knowledge_base = {}
        
        # Initialize RAG system
        self._initialize_rag_system()
    
    def _initialize_rag_system(self):
        """Initialize the RAG system with vector database and embeddings."""
        try:
            # Initialize ChromaDB for vector storage
            self.vector_db = chromadb.Client()
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.console.print("[green]✅ RAG System initialized with ChromaDB and embeddings[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ RAG initialization failed: {e}[/red]")
            self.console.print("[yellow]⚠️ Falling back to simple RAG mode[/yellow]")
    
    def load_and_index_datasets(self):
        """Load all datasets and create RAG knowledge base."""
        self.console.print("[blue]📚 Building RAG Knowledge Base from Datasets...[/blue]")
        
        # Find all dataset files
        dataset_files = list(self.data_dir.glob("*.csv")) + list(self.data_dir.glob("*.xlsx"))
        
        if not dataset_files:
            self.console.print("[red]❌ No datasets found in data directory[/red]")
            return
        
        for file_path in dataset_files:
            self.console.print(f"[blue]📊 Processing: {file_path.name}[/blue]")
            
            try:
                # Load dataset
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                # Create comprehensive dataset knowledge
                dataset_knowledge = self._create_dataset_knowledge(df, file_path)
                
                # Store in knowledge base
                dataset_name = file_path.stem
                self.knowledge_base[dataset_name] = dataset_knowledge
                
                # Index in vector database if available
                if self.vector_db:
                    self._index_dataset_in_vector_db(dataset_name, dataset_knowledge)
                
                self.console.print(f"[green]✅ Indexed: {dataset_name} ({len(df):,} records)[/green]")
                
            except Exception as e:
                self.console.print(f"[red]❌ Error processing {file_path.name}: {e}[/red]")
        
        self.console.print(f"[green]🎉 Knowledge base ready with {len(self.knowledge_base)} datasets![/green]")
    
    def _create_dataset_knowledge(self, df: pd.DataFrame, file_path: Path) -> Dict[str, Any]:
        """Create comprehensive knowledge about a dataset."""
        
        # Basic dataset info
        knowledge = {
            "file_name": file_path.name,
            "total_records": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "column_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(10).to_dict('records'),
            "statistical_summary": df.describe().to_dict(),
            "unique_values": {col: df[col].nunique() for col in df.columns},
            "domain_analysis": self._analyze_domain_with_ollama(df),
            "insights": self._generate_insights_with_ollama(df),
            "patterns": self._find_patterns_with_ollama(df),
            "anomalies": self._detect_anomalies_with_ollama(df)
        }
        
        return knowledge
    
    def _analyze_domain_with_ollama(self, df: pd.DataFrame) -> str:
        """Use Ollama to analyze the domain of the dataset."""
        try:
            context = f"""
Analyze the domain/type of this dataset based on its columns and sample data:

Columns: {', '.join(df.columns)}
Sample data (first 5 rows):
{df.head(5).to_string()}

What type of data is this? What domain does it belong to? (e.g., rainfall/weather, consumption/utility, healthcare, education, economic, etc.)
Provide a brief domain analysis.
"""
            
            response = self.llm.invoke(context)
            return response.strip()
            
        except Exception as e:
            return f"Domain analysis failed: {e}"
    
    def _generate_insights_with_ollama(self, df: pd.DataFrame) -> str:
        """Use Ollama to generate insights from the dataset."""
        try:
            context = f"""
Generate key insights from this dataset:

Dataset Info:
- Records: {len(df):,}
- Columns: {', '.join(df.columns)}
- Data types: {df.dtypes.to_dict()}

Statistical Summary:
{df.describe().to_string()}

Sample Data:
{df.head(10).to_string()}

What are the key insights, trends, and important findings in this data?
"""
            
            response = self.llm.invoke(context)
            return response.strip()
            
        except Exception as e:
            return f"Insights generation failed: {e}"
    
    def _find_patterns_with_ollama(self, df: pd.DataFrame) -> str:
        """Use Ollama to find patterns in the dataset."""
        try:
            context = f"""
Find patterns and trends in this dataset:

Dataset: {len(df):,} records, {len(df.columns)} columns
Columns: {', '.join(df.columns)}

Statistical Summary:
{df.describe().to_string()}

Sample Data:
{df.head(10).to_string()}

What patterns, trends, correlations, or relationships do you see in this data?
"""
            
            response = self.llm.invoke(context)
            return response.strip()
            
        except Exception as e:
            return f"Pattern analysis failed: {e}"
    
    def _detect_anomalies_with_ollama(self, df: pd.DataFrame) -> str:
        """Use Ollama to detect anomalies in the dataset."""
        try:
            context = f"""
Detect anomalies and unusual patterns in this dataset:

Dataset: {len(df):,} records, {len(df.columns)} columns
Columns: {', '.join(df.columns)}

Statistical Summary:
{df.describe().to_string()}

Missing Values:
{df.isnull().sum().to_string()}

Sample Data:
{df.head(10).to_string()}

What anomalies, outliers, or unusual patterns do you detect in this data?
"""
            
            response = self.llm.invoke(context)
            return response.strip()
            
        except Exception as e:
            return f"Anomaly detection failed: {e}"
    
    def _index_dataset_in_vector_db(self, dataset_name: str, knowledge: Dict[str, Any]):
        """Index dataset knowledge in vector database."""
        try:
            # Create collection for this dataset
            collection = self.vector_db.create_collection(name=dataset_name)
            
            # Prepare documents for indexing
            documents = []
            metadatas = []
            ids = []
            
            # Index different aspects of the knowledge
            aspects = [
                ("domain_analysis", knowledge["domain_analysis"]),
                ("insights", knowledge["insights"]),
                ("patterns", knowledge["patterns"]),
                ("anomalies", knowledge["anomalies"])
            ]
            
            for i, (aspect, content) in enumerate(aspects):
                documents.append(content)
                metadatas.append({
                    "dataset": dataset_name,
                    "aspect": aspect,
                    "file_name": knowledge["file_name"],
                    "records": knowledge["total_records"]
                })
                ids.append(f"{dataset_name}_{aspect}_{i}")
            
            # Add to collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
        except Exception as e:
            self.console.print(f"[yellow]⚠️ Vector indexing failed for {dataset_name}: {e}[/yellow]")
    
    def ask_question(self, question: str, dataset_name: str = None) -> str:
        """Ask any question - 100% handled by Ollama with RAG."""
        
        self.console.print(Panel.fit(
            f"[bold blue]🤖 Ollama RAG System[/bold blue]\n"
            f"Question: [green]{question}[/green]\n"
            f"Dataset: [yellow]{dataset_name or 'All Datasets'}[/yellow]",
            title="💬 100% Ollama-Powered Q&A",
            border_style="blue"
        ))
        
        try:
            # Retrieve relevant context using RAG
            relevant_context = self._retrieve_relevant_context(question, dataset_name)
            
            # Generate answer using Ollama with RAG context
            answer = self._generate_ollama_answer(question, relevant_context)
            
            return answer
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _retrieve_relevant_context(self, question: str, dataset_name: str = None) -> str:
        """Retrieve relevant context using RAG."""
        
        context_parts = []
        
        if dataset_name and dataset_name in self.knowledge_base:
            # Specific dataset requested
            knowledge = self.knowledge_base[dataset_name]
            context_parts.append(f"=== DATASET: {dataset_name} ===")
            context_parts.append(f"File: {knowledge['file_name']}")
            context_parts.append(f"Records: {knowledge['total_records']:,}")
            context_parts.append(f"Columns: {', '.join(knowledge['columns'])}")
            context_parts.append(f"Domain Analysis: {knowledge['domain_analysis']}")
            context_parts.append(f"Key Insights: {knowledge['insights']}")
            context_parts.append(f"Patterns: {knowledge['patterns']}")
            context_parts.append(f"Anomalies: {knowledge['anomalies']}")
            context_parts.append(f"Sample Data: {json.dumps(knowledge['sample_data'][:5], indent=2)}")
            
        else:
            # Search across all datasets
            for name, knowledge in self.knowledge_base.items():
                context_parts.append(f"=== DATASET: {name} ===")
                context_parts.append(f"File: {knowledge['file_name']}")
                context_parts.append(f"Records: {knowledge['total_records']:,}")
                context_parts.append(f"Columns: {', '.join(knowledge['columns'])}")
                context_parts.append(f"Domain: {knowledge['domain_analysis']}")
                context_parts.append(f"Insights: {knowledge['insights']}")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _generate_ollama_answer(self, question: str, context: str) -> str:
        """Generate answer using Ollama with RAG context."""
        
        prompt = f"""
You are an expert data analyst for government data. You have access to comprehensive dataset information through RAG (Retrieval-Augmented Generation).

CONTEXT FROM DATASETS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided dataset context
2. Be specific and use actual data insights
3. Provide detailed analysis with numbers and facts
4. If the question is about implications, provide policy recommendations
5. If the question is about patterns, explain what you found
6. If the question is about anomalies, describe what's unusual
7. Always base your answer on the actual data provided
8. Be comprehensive and insightful

ANSWER:
"""
        
        try:
            response = self.llm.invoke(prompt)
            return f"🤖 **Ollama Analysis**: {response}"
            
        except Exception as e:
            return f"❌ Ollama error: {str(e)}"
    
    def interactive_mode(self):
        """Start interactive Q&A mode."""
        self.console.print(Panel.fit(
            "[bold blue]🤖 Interactive Ollama RAG Mode[/bold blue]\n"
            "Ask any question about your datasets!\n"
            "Type 'quit' to exit, 'help' for examples",
            title="💬 100% Ollama-Powered Chat",
            border_style="blue"
        ))
        
        while True:
            try:
                question = input("\n🤖 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    self.console.print("[yellow]👋 Goodbye![/yellow]")
                    break
                
                if question.lower() in ['help', 'h']:
                    self.console.print("""
[bold blue]Example Questions:[/bold blue]
• "What insights can you provide about the rainfall data?"
• "What are the implications for agriculture?"
• "What patterns do you see in the consumption data?"
• "What anomalies are present in the datasets?"
• "How can this data be used for policy making?"
• "What are the key findings across all datasets?"
• "Which dataset is most important for development?"
""")
                    continue
                
                if not question:
                    continue
                
                # Ask question
                answer = self.ask_question(question)
                self.console.print(f"\n{answer}\n")
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")


def main():
    """Main function to run the Ollama RAG system."""
    import sys
    
    rag_system = OllamaRAGSystem()
    
    # Load and index all datasets
    rag_system.load_and_index_datasets()
    
    if len(sys.argv) > 1:
        # Command line question
        question = " ".join(sys.argv[1:])
        answer = rag_system.ask_question(question)
        print(answer)
    else:
        # Interactive mode
        rag_system.interactive_mode()


if __name__ == "__main__":
    main()



