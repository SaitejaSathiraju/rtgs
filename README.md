# 🚀 RTGS - Real-Time Government System AI Analyst

A comprehensive CLI-first AI analyst for Telangana Open Data, powered by Ollama and CrewAI agents.

## 🎯 Overview

RTGS is an intelligent data analysis system that processes government datasets, generates insights, and provides interactive Q&A capabilities. It uses local LLMs via Ollama and multi-agent coordination through CrewAI for comprehensive data analysis.

## 🤖 Agent Architecture

The system is built around **5 core agents**, each with a specific role:

1. **Coordinator Agent** – main coordinator and task delegator
2. **Data Cleaner Agent** – preprocesses and cleans raw data/text
3. **Data Transformer Agent** – converts or restructures text/data into other formats
4. **Data Analyst Agent** – performs deeper analysis on given input
5. **Data Summarizer Agent** – generates concise summaries from content

Each agent operates autonomously, thinking through problems and taking appropriate actions based on their specialized expertise.

## 🛠️ Features

- **🤖 Multi-Agent Analysis**: 5 specialized CrewAI agents for comprehensive data analysis
- **🧠 100% Ollama-Powered**: Local LLM processing with no external dependencies
- **📊 Interactive Q&A**: Ask questions about your data in natural language
- **🔄 Complete Data Pipeline**: Raw → Cleaned → Standardized → Transformed
- **📈 Rich Visualizations**: Terminal charts and generated plot files
- **🎯 Auto-Discovery**: Automatically processes new datasets in `/data` folder
- **📝 Comprehensive Logging**: Detailed agent and process tracking

## 🚀 Quick Start

### 1. Process Everything (Recommended)
```bash
py cli_analyst.py ultimate
```

### 2. Ask Questions
```bash
py cli_analyst.py ask "Which district has highest rainfall?"
py cli_analyst.py ask "What is the total revenue?"
```

### 3. Agent Analysis
```bash
py cli_analyst.py agent-analyze "data/your_dataset.csv"
```

## 📋 Commands Reference

### 🎯 Core Analysis Commands

#### **Ultimate Command (Process Everything)**
```bash
py cli_analyst.py ultimate
```
- **Purpose**: Processes ALL datasets in `/data` folder automatically
- **Features**: Auto-discovery, processing, charts, analysis, Q&A bot ready
- **Best for**: Getting everything ready at once

#### **Agent-Based Analysis**
```bash
py cli_analyst.py agent-analyze "data/your_dataset.csv"
```
- **Purpose**: Uses CrewAI agents for comprehensive analysis
- **Agents**: Coordinator, DataCleaner, DataTransformer, DataAnalyst, DataSummarizer
- **Features**: Complete agent logging, sequential processing, detailed reports
- **Best for**: Deep analysis with agent collaboration

#### **Hybrid Analysis**
```bash
py cli_analyst.py hybrid-analyze "data/your_dataset.csv" [OPTIONS]
```
- **Purpose**: Fast programmatic analysis
- **Options**: 
  - `--use-agents` - Use agents instead of hybrid
  - `--output-format json/text/both`
- **Best for**: Quick insights and analysis

### 💬 Q&A Commands

#### **Ask Questions (Auto-detect dataset)**
```bash
py cli_analyst.py ask "Which district has highest rainfall?"
py cli_analyst.py ask "What is the total revenue?"
py cli_analyst.py ask "Show me trends by month"
```

#### **Ask Questions (Specific dataset)**
```bash
py cli_analyst.py ask "Which district has highest rainfall?" --dataset "Tourism Foreign Visitors Data 2024"
py cli_analyst.py ask "What is the average visitors?" --dataset "Tourism Foreign Visitors Data 2024"
py cli_analyst.py ask "Show me top 5 places" --dataset "Tourism Foreign Visitors Data 2024"
```

### 🔍 RAG System Commands

#### **RAG Analysis**
```bash
py cli_analyst.py rag "What are the key insights from all datasets?"
py cli_analyst.py rag "Compare rainfall patterns across districts"
```

#### **RAG with Specific Dataset**
```bash
py cli_analyst.py rag "Analyze tourism trends" --dataset "Tourism Foreign Visitors Data 2024"
```

### 📈 Data Pipeline Commands

#### **Complete Data Pipeline**
```bash
py cli_analyst.py pipeline "data/your_dataset.csv"
```
- **Purpose**: Raw → Cleaned → Standardized → Transformed
- **Features**: Before/after visibility, progress tracking, file outputs

#### **Data Processing**
```bash
py cli_analyst.py process "data/your_dataset.csv"
```
- **Purpose**: Clean and standardize data
- **Features**: Data quality analysis, cleaning recommendations

### 🎨 Visualization Commands

#### **Generate Charts**
```bash
py cli_analyst.py charts "data/your_dataset.csv"
```
- **Purpose**: Creates multiple chart types
- **Features**: Data quality, correlation, distribution charts

#### **Terminal Charts**
```bash
py cli_analyst.py terminal-charts "data/your_dataset.csv"
```
- **Purpose**: ASCII charts in terminal
- **Features**: Bar charts, histograms, correlation matrices

### 🔧 Utility Commands

#### **List Datasets**
```bash
py cli_analyst.py list-datasets
```

#### **Dataset Status**
```bash
py cli_analyst.py status
```

#### **Clean Outputs**
```bash
py cli_analyst.py clean-outputs
```

#### **Show Output Links**
```bash
py cli_analyst.py show-outputs
```

## 📁 Project Structure

```
rtgs-v3/
├── data/                           # Raw datasets
│   └── Tourism Foreign Visitors Data 2024.csv
├── output/                         # Generated outputs
│   ├── charts/                     # Generated charts
│   ├── analysis/                   # Analysis reports
│   ├── cleaned_data/              # Cleaned datasets
│   └── transformed_data/          # Transformed datasets
├── logs/                          # System logs
│   └── readable.log              # Human-readable logs
├── agents/                        # CrewAI agents
│   ├── base_agent.py
│   ├── coordinator_agent.py
│   ├── data_cleaner.py
│   ├── data_transformer.py
│   ├── data_analyst.py
│   └── data_summarizer.py
├── cli_analyst.py                 # Main CLI interface
├── agent_analyst.py              # Agent orchestration
├── hybrid_analyst.py             # Hybrid analysis
├── qa_bot.py                     # Q&A bot
├── data_pipeline.py              # Data processing pipeline
├── ollama_rag_system.py          # RAG system
├── readable_logger.py            # Enhanced logging
└── config.py                     # Configuration
```

## 🎯 Usage Examples

### **Complete Workflow**
```bash
# 1. Process all datasets
py cli_analyst.py ultimate

# 2. Ask questions
py cli_analyst.py ask "Which district has highest rainfall?"
py cli_analyst.py ask "What is the total revenue?"

# 3. Deep analysis with agents
py cli_analyst.py agent-analyze "data/Tourism Foreign Visitors Data 2024.csv"

# 4. RAG analysis
py cli_analyst.py rag "What are the key insights from all datasets?"
```

### **Dataset-Specific Analysis**
```bash
# Analyze specific dataset
py cli_analyst.py hybrid-analyze "data/Tourism Foreign Visitors Data 2024.csv"

# Ask questions about specific dataset
py cli_analyst.py ask "Which place has most visitors?" --dataset "Tourism Foreign Visitors Data 2024"

# RAG analysis on specific dataset
py cli_analyst.py rag "Analyze tourism trends" --dataset "Tourism Foreign Visitors Data 2024"
```

### **Data Pipeline**
```bash
# Complete data pipeline
py cli_analyst.py pipeline "data/Tourism Foreign Visitors Data 2024.csv"

# Generate charts
py cli_analyst.py charts "data/Tourism Foreign Visitors Data 2024.csv"
```

## 🔧 Configuration

### **Ollama Model**
The system uses `llama3.2:latest` by default. To change the model:

1. Edit `config.py`:
```python
OLLAMA_MODEL = "your-preferred-model"
```

2. Available models:
- `llama3.2:latest`
- `llama3.1:latest`
- `mistral:latest`
- `gpt-oss:120b`
- `deepseek-r1:14b-qwen-distill-q8_0`

### **Application Name**
The application name is set to "RTGS" (Real-Time Government System) in `config.py`.

## 📊 Output Files

### **Charts**
- Data quality charts
- Correlation matrices
- Distribution plots
- Location: `output/charts/`

### **Analysis Reports**
- Agent analysis reports
- Hybrid analysis reports
- RAG analysis reports
- Location: `output/analysis/`

### **Processed Data**
- Cleaned datasets
- Standardized datasets
- Transformed datasets
- Location: `output/cleaned_data/`, `output/transformed_data/`

### **Logs**
- Human-readable logs with timestamps
- Agent and process tracking
- Location: `logs/readable.log`

## 🚀 Pro Tips

1. **Start with Ultimate**: Always run `py cli_analyst.py ultimate` first to process all datasets
2. **Use Dataset Flag**: For specific questions, use `--dataset` flag to target specific datasets
3. **Agent Analysis**: Use `agent-analyze` for comprehensive analysis with detailed logging
4. **RAG Insights**: Use `rag` command for context-aware analysis across all datasets
5. **Check Outputs**: Use `show-outputs` to see all generated files and charts
6. **Auto-Discovery**: Just add new datasets to `/data` folder and run `ultimate` to process them

## 🎯 Available Datasets

- **Tourism Foreign Visitors Data 2024.csv** (348 records, 4 columns)
- Add more datasets to `/data` folder for automatic processing

## 📝 Logging Features

The system provides comprehensive logging with:
- **🤖 Agent Logging**: Start/end tracking for all agents
- **📋 Task Logging**: Individual task assignment and completion
- **🚀 Crew Logging**: Multi-agent crew execution monitoring
- **⚙️ Process Logging**: Process start/end with detailed status
- **📊 Rich Formatting**: Emoji-based, color-coded logging
- **⏱️ Timestamps**: Precise timing for all operations

## 🎉 Getting Started

1. **Install Dependencies**: Ensure all requirements are installed
2. **Run Ultimate**: `py cli_analyst.py ultimate`
3. **Ask Questions**: `py cli_analyst.py ask "your question"`
4. **Explore Outputs**: Check `output/` folder for generated files

**Your RTGS system is ready for intelligent data analysis!** 🎯