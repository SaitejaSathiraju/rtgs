"""
Real executable tools for CrewAI agents to perform actual data analysis.
These tools actually execute pandas code and return real results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class ReadDatasetInput(BaseModel):
    file_path: str = Field(description="Path to the dataset file")


class ReadDatasetTool(BaseTool):
    name: str = "read_dataset"
    description: str = "Read a dataset file and return basic information about it"
    args_schema: Type[BaseModel] = ReadDatasetInput

    def _run(self, file_path: str) -> str:
        try:
            # Read the dataset
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return f"Error: Unsupported file format for {file_path}"
            
            # Generate comprehensive dataset info
            info = f"""
DATASET INFORMATION:
==================
File: {file_path}
Records: {len(df):,}
Columns: {len(df.columns)}
Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

COLUMN DETAILS:
===============
"""
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100
                unique_count = df[col].nunique()
                
                info += f"• {col}: {dtype}, {null_count} nulls ({null_pct:.1f}%), {unique_count} unique values\n"
            
            # Add sample data
            info += f"\nSAMPLE DATA (first 3 rows):\n"
            info += f"{df.head(3).to_string()}\n"
            
            return info
            
        except Exception as e:
            return f"Error reading dataset {file_path}: {str(e)}"


class AnalyzeDataQualityInput(BaseModel):
    file_path: str = Field(description="Path to the dataset file")


class AnalyzeDataQualityTool(BaseTool):
    name: str = "analyze_data_quality"
    description: str = "Perform comprehensive data quality analysis on a dataset"
    args_schema: Type[BaseModel] = AnalyzeDataQualityInput

    def _run(self, file_path: str) -> str:
        try:
            # Read dataset
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return f"Error: Unsupported file format for {file_path}"
            
            analysis = f"""
DATA QUALITY ANALYSIS:
=====================
Dataset: {file_path}
Total Records: {len(df):,}
Total Columns: {len(df.columns)}

MISSING VALUES ANALYSIS:
=======================
"""
            
            missing_data = df.isnull().sum()
            total_missing = missing_data.sum()
            
            if total_missing > 0:
                analysis += f"Total Missing Values: {total_missing:,} ({(total_missing / (len(df) * len(df.columns)) * 100):.2f}% of all data)\n\n"
                
                for col, missing_count in missing_data[missing_data > 0].items():
                    percentage = (missing_count / len(df)) * 100
                    analysis += f"• {col}: {missing_count:,} missing ({percentage:.1f}%)\n"
            else:
                analysis += "✅ No missing values found!\n"
            
            analysis += f"\nDUPLICATE ANALYSIS:\n"
            analysis += f"==================\n"
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                analysis += f"Duplicate Rows: {duplicate_count:,} ({(duplicate_count / len(df) * 100):.1f}%)\n"
            else:
                analysis += "✅ No duplicate rows found!\n"
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing data quality for {file_path}: {str(e)}"


class CleanDatasetInput(BaseModel):
    file_path: str = Field(description="Path to the input dataset")
    output_path: Optional[str] = Field(default=None, description="Path to save cleaned dataset")


class CleanDatasetTool(BaseTool):
    name: str = "clean_dataset"
    description: str = "Clean a dataset by handling missing values, duplicates, and data type issues"
    args_schema: Type[BaseModel] = CleanDatasetInput

    def _run(self, file_path: str, output_path: str = None) -> str:
        try:
            # Read dataset
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return f"Error: Unsupported file format for {file_path}"
            
            original_df = df.copy()
            cleaning_report = f"""
DATA CLEANING REPORT:
====================
Original Records: {len(df):,}
Original Columns: {len(df.columns)}

CLEANING OPERATIONS PERFORMED:
=============================
"""
            
            # 1. Handle missing values
            missing_before = df.isnull().sum().sum()
            
            # For numeric columns: fill with median
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    cleaning_report += f"• Filled {col} missing values with median: {median_val:.2f}\n"
            
            # For categorical columns: fill with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                    cleaning_report += f"• Filled {col} missing values with mode: '{mode_val}'\n"
            
            missing_after = df.isnull().sum().sum()
            cleaning_report += f"• Missing values: {missing_before:,} → {missing_after:,}\n"
            
            # 2. Remove duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                df = df.drop_duplicates()
                cleaning_report += f"• Removed {duplicate_count:,} duplicate rows\n"
            
            # Save cleaned dataset
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"cleaned_dataset_{timestamp}.csv"
            
            df.to_csv(output_path, index=False)
            
            cleaning_report += f"""
FINAL RESULTS:
=============
Cleaned Records: {len(df):,}
Cleaned Columns: {len(df.columns)}
Cleaned Dataset Saved: {output_path}

QUALITY IMPROVEMENT:
===================
Missing Values Reduced: {missing_before - missing_after:,}
Duplicates Removed: {duplicate_count:,}
"""
            
            return cleaning_report
            
        except Exception as e:
            return f"Error cleaning dataset {file_path}: {str(e)}"


# Export all tools for use in agents
ALL_TOOLS = [
    ReadDatasetTool(),
    AnalyzeDataQualityTool(),
    CleanDatasetTool()
]
