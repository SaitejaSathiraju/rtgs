"""Data utility functions for Telangana Open Data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class DataUtils:
    """Utility class for data processing operations."""
    
    # Telangana districts (as of 2023)
    TELANGANA_DISTRICTS = [
        "Adilabad", "Bhadradri Kothagudem", "Hyderabad", "Jagtial", 
        "Jangaon", "Jayashankar Bhupalpally", "Jogulamba Gadwal", 
        "Kamareddy", "Karimnagar", "Khammam", "Komaram Bheem Asifabad",
        "Mahabubabad", "Mahabubnagar", "Mancherial", "Medak", 
        "Medchal-Malkajgiri", "Mulugu", "Nagarkurnool", "Nalgonda",
        "Narayanpet", "Nirmal", "Nizamabad", "Peddapalli", 
        "Rajanna Sircilla", "Rangareddy", "Sangareddy", "Siddipet",
        "Suryapet", "Vikarabad", "Wanaparthy", "Warangal Urban",
        "Warangal Rural", "Yadadri Bhuvanagiri"
    ]
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to snake_case."""
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        return df
    
    @staticmethod
    def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """Detect and suggest data types for columns."""
        type_suggestions = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it's a date
                if DataUtils._is_date_column(df[col]):
                    type_suggestions[col] = 'datetime'
                # Check if it's numeric
                elif DataUtils._is_numeric_column(df[col]):
                    type_suggestions[col] = 'numeric'
                else:
                    type_suggestions[col] = 'categorical'
            else:
                type_suggestions[col] = str(df[col].dtype)
        
        return type_suggestions
    
    @staticmethod
    def _is_date_column(series: pd.Series) -> bool:
        """Check if a column contains date-like data."""
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        
        try:
            pd.to_datetime(sample)
            return True
        except:
            return False
    
    @staticmethod
    def _is_numeric_column(series: pd.Series) -> bool:
        """Check if a column contains numeric data."""
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        
        try:
            pd.to_numeric(sample)
            return True
        except:
            return False
    
    @staticmethod
    def calculate_data_quality_score(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        completeness = 1 - (missing_cells / total_cells)
        
        # Check for duplicates
        duplicate_rows = df.duplicated().sum()
        uniqueness = 1 - (duplicate_rows / df.shape[0])
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_score = 1.0
        if len(numeric_cols) > 0:
            outlier_counts = 0
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_counts += outliers
            
            outlier_score = 1 - (outlier_counts / (len(numeric_cols) * df.shape[0]))
        
        return {
            "completeness": completeness,
            "uniqueness": uniqueness,
            "outlier_score": outlier_score,
            "overall_score": (completeness + uniqueness + outlier_score) / 3
        }
    
    @staticmethod
    def generate_data_profile(df: pd.DataFrame) -> Dict:
        """Generate a comprehensive data profile."""
        profile = {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_values": df.nunique().to_dict(),
            "data_quality": DataUtils.calculate_data_quality_score(df)
        }
        
        # Add statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            profile["numeric_statistics"] = df[numeric_cols].describe().to_dict()
        
        # Add value counts for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            profile["categorical_summary"] = {}
            for col in categorical_cols:
                profile["categorical_summary"][col] = df[col].value_counts().head(10).to_dict()
        
        return profile
    
    @staticmethod
    def validate_telangana_data(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate data specific to Telangana."""
        issues = {
            "district_issues": [],
            "administrative_issues": [],
            "geographic_issues": []
        }
        
        # Check for district names
        if 'district' in df.columns.str.lower():
            district_col = [col for col in df.columns if 'district' in col.lower()][0]
            invalid_districts = df[district_col].dropna().unique()
            invalid_districts = [d for d in invalid_districts 
                               if d not in DataUtils.TELANGANA_DISTRICTS]
            
            if invalid_districts:
                issues["district_issues"] = invalid_districts
        
        return issues
    
    @staticmethod
    def create_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
        """Create common derived fields for Telangana data."""
        df_derived = df.copy()
        
        # Create year field if date column exists
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            df_derived['year'] = pd.to_datetime(df_derived[date_col], errors='coerce').dt.year
        
        # Create per capita fields if population and value columns exist
        if 'population' in df.columns.str.lower() and 'value' in df.columns.str.lower():
            pop_col = [col for col in df.columns if 'population' in col.lower()][0]
            value_col = [col for col in df.columns if 'value' in col.lower()][0]
            df_derived['per_capita'] = df_derived[value_col] / df_derived[pop_col]
        
        return df_derived

