"""File reading tools for CrewAI agents."""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List


def read_csv_file(file_path: str) -> Dict[str, Any]:
    """
    Read a CSV file and return its structure and content summary.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary containing file information
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Get basic information
        info = {
            "file_path": file_path,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(5).to_dict(),
            "basic_stats": {}
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            info["basic_stats"] = df[numeric_cols].describe().to_dict()
        
        # Add unique values for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        info["categorical_info"] = {}
        for col in categorical_cols:
            info["categorical_info"][col] = {
                "unique_count": df[col].nunique(),
                "unique_values": df[col].unique().tolist()[:10]  # First 10 unique values
            }
        
        return info
        
    except Exception as e:
        return {
            "error": f"Failed to read CSV file: {str(e)}",
            "file_path": file_path
        }


def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read a JSON file and return its content.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing file content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        return {
            "file_path": file_path,
            "content": content,
            "type": "json"
        }
        
    except Exception as e:
        return {
            "error": f"Failed to read JSON file: {str(e)}",
            "file_path": file_path
        }


def analyze_dataset_structure(file_path: str) -> Dict[str, Any]:
    """
    Analyze the structure of a dataset file.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Dictionary containing dataset analysis
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.csv':
        return read_csv_file(file_path)
    elif file_ext == '.json':
        return read_json_file(file_path)
    else:
        return {
            "error": f"Unsupported file type: {file_ext}",
            "file_path": file_path
        }


def get_dataset_summary(file_path: str) -> str:
    """
    Get a text summary of the dataset for agent consumption.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        String summary of the dataset
    """
    analysis = analyze_dataset_structure(file_path)
    
    if "error" in analysis:
        return f"Error analyzing dataset: {analysis['error']}"
    
    summary = f"""
Dataset Analysis Summary:
- File: {analysis['file_path']}
- Total Rows: {analysis['total_rows']:,}
- Total Columns: {analysis['total_columns']}
- Columns: {', '.join(analysis['columns'])}

Data Types:
"""
    
    for col, dtype in analysis['data_types'].items():
        summary += f"- {col}: {dtype}\n"
    
    summary += "\nMissing Values:\n"
    for col, missing in analysis['missing_values'].items():
        if missing > 0:
            summary += f"- {col}: {missing} missing values\n"
    
    if analysis['basic_stats']:
        summary += "\nNumeric Column Statistics:\n"
        for col, stats in analysis['basic_stats'].items():
            summary += f"- {col}: mean={stats.get('mean', 'N/A'):.2f}, std={stats.get('std', 'N/A'):.2f}\n"
    
    if analysis.get('categorical_info'):
        summary += "\nCategorical Column Info:\n"
        for col, info in analysis['categorical_info'].items():
            summary += f"- {col}: {info['unique_count']} unique values\n"
    
    summary += "\nSample Data (first 3 rows):\n"
    for i, row in enumerate(list(analysis['sample_data'].values())[:3]):
        summary += f"Row {i+1}: {row}\n"
    
    return summary

