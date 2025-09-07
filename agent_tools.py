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


@tool("read_dataset")
def read_dataset_tool(file_path: str) -> str:
    """
    Read a dataset file and return basic information about it.
    
    Args:
        file_path: Path to the dataset file (CSV, Excel, etc.)
    
    Returns:
        String containing dataset information
    """
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


@tool("analyze_data_quality")
def analyze_data_quality_tool(file_path: str) -> str:
    """
    Perform comprehensive data quality analysis on a dataset.
    
    Args:
        file_path: Path to the dataset file
    
    Returns:
        String containing detailed data quality analysis
    """
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
        
        analysis += f"\nDATA TYPE ANALYSIS:\n"
        analysis += f"===================\n"
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            analysis += f"• {dtype}: {count} columns\n"
        
        analysis += f"\nOUTLIER ANALYSIS:\n"
        analysis += f"=================\n"
        numeric_cols = df.select_dtypes(include=['number']).columns
        outlier_count = 0
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                outlier_count += len(outliers)
                analysis += f"• {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)\n"
        
        if outlier_count == 0:
            analysis += "✅ No outliers detected!\n"
        
        return analysis
        
    except Exception as e:
        return f"Error analyzing data quality for {file_path}: {str(e)}"


@tool("clean_dataset")
def clean_dataset_tool(file_path: str, output_path: str = None) -> str:
    """
    Clean a dataset by handling missing values, duplicates, and data type issues.
    
    Args:
        file_path: Path to the input dataset
        output_path: Path to save cleaned dataset (optional)
    
    Returns:
        String containing cleaning report and file path
    """
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
        
        # 3. Standardize text data
        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip().str.title()
        
        cleaning_report += f"• Standardized text formatting in {len(categorical_cols)} columns\n"
        
        # 4. Convert date columns
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                cleaning_report += f"• Converted {col} to datetime format\n"
            except:
                pass
        
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
Data Types Standardized: {len(date_cols)} date columns converted
"""
        
        return cleaning_report
        
    except Exception as e:
        return f"Error cleaning dataset {file_path}: {str(e)}"


@tool("transform_dataset")
def transform_dataset_tool(file_path: str, output_path: str = None) -> str:
    """
    Transform a dataset by creating derived features, encoding categorical variables, and normalizing data.
    
    Args:
        file_path: Path to the input dataset
        output_path: Path to save transformed dataset (optional)
    
    Returns:
        String containing transformation report and file path
    """
    try:
        # Read dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return f"Error: Unsupported file format for {file_path}"
        
        original_df = df.copy()
        transformation_report = f"""
DATA TRANSFORMATION REPORT:
==========================
Original Records: {len(df):,}
Original Columns: {len(df.columns)}

TRANSFORMATION OPERATIONS PERFORMED:
===================================
"""
        
        # 1. Create derived features for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        features_created = 0
        
        for col in numeric_cols:
            if col in df.columns:
                try:
                    # Create log transformation (handle zeros)
                    df[f'{col}_log'] = np.log1p(df[col])
                    
                    # Create square root transformation
                    df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
                    
                    # Create normalized version
                    if df[col].std() > 0:
                        df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
                    
                    # Create categorical bins
                    df[f'{col}_category'] = pd.cut(df[col], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                    
                    features_created += 4
                    transformation_report += f"• Created 4 derived features for {col}\n"
                    
                except Exception as e:
                    transformation_report += f"• Could not create features for {col}: {str(e)}\n"
        
        # 2. Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        encodings_created = 0
        
        for col in categorical_cols:
            if col in df.columns and df[col].nunique() < 50:  # Only encode if reasonable number of categories
                try:
                    # Label encoding
                    unique_values = df[col].unique()
                    label_map = {val: idx for idx, val in enumerate(unique_values)}
                    df[f'{col}_encoded'] = df[col].map(label_map)
                    
                    # One-hot encoding for binary variables
                    if df[col].nunique() == 2:
                        df[f'{col}_binary'] = (df[col] == df[col].mode()[0]).astype(int)
                        encodings_created += 2
                    else:
                        encodings_created += 1
                    
                    transformation_report += f"• Encoded categorical variable {col}\n"
                    
                except Exception as e:
                    transformation_report += f"• Could not encode {col}: {str(e)}\n"
        
        # 3. Create date features
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        date_features_created = 0
        
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                    df[f'{col}_quarter'] = df[col].dt.quarter
                    
                    date_features_created += 5
                    transformation_report += f"• Created 5 date features for {col}\n"
                    
                except Exception as e:
                    transformation_report += f"• Could not create date features for {col}: {str(e)}\n"
        
        # 4. Create aggregation features
        geo_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['district', 'city', 'village', 'location', 'place'])]
        aggregation_features = 0
        
        for geo_col in geo_cols[:2]:  # Limit to first 2 geographic columns
            for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                if geo_col != num_col:
                    try:
                        # Group statistics
                        group_stats = df.groupby(geo_col)[num_col].agg(['mean', 'sum', 'count', 'std']).reset_index()
                        group_stats.columns = [geo_col, f'{num_col}_group_mean', f'{num_col}_group_sum', f'{num_col}_group_count', f'{num_col}_group_std']
                        
                        # Merge back
                        df = df.merge(group_stats, on=geo_col, how='left')
                        aggregation_features += 4
                        
                    except Exception as e:
                        transformation_report += f"• Could not create aggregation for {geo_col} x {num_col}: {str(e)}\n"
        
        # Save transformed dataset
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"transformed_dataset_{timestamp}.csv"
        
        df.to_csv(output_path, index=False)
        
        transformation_report += f"""
FINAL RESULTS:
=============
Transformed Records: {len(df):,}
Transformed Columns: {len(df.columns)}
Transformed Dataset Saved: {output_path}

FEATURES CREATED:
================
Derived Features: {features_created}
Categorical Encodings: {encodings_created}
Date Features: {date_features_created}
Aggregation Features: {aggregation_features}
Total New Features: {len(df.columns) - len(original_df.columns)}
"""
        
        return transformation_report
        
    except Exception as e:
        return f"Error transforming dataset {file_path}: {str(e)}"


@tool("analyze_statistics")
def analyze_statistics_tool(file_path: str) -> str:
    """
    Perform comprehensive statistical analysis on a dataset.
    
    Args:
        file_path: Path to the dataset file
    
    Returns:
        String containing detailed statistical analysis
    """
    try:
        # Read dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return f"Error: Unsupported file format for {file_path}"
        
        analysis = f"""
STATISTICAL ANALYSIS REPORT:
===========================
Dataset: {file_path}
Records: {len(df):,}
Columns: {len(df.columns)}

DESCRIPTIVE STATISTICS:
=====================
"""
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            analysis += f"\nNUMERIC VARIABLES ({len(numeric_cols)} columns):\n"
            analysis += f"{'='*50}\n"
            
            for col in numeric_cols:
                stats = df[col].describe()
                analysis += f"\n{col}:\n"
                analysis += f"  Mean: {stats['mean']:.2f}\n"
                analysis += f"  Median: {stats['50%']:.2f}\n"
                analysis += f"  Std Dev: {stats['std']:.2f}\n"
                analysis += f"  Min: {stats['min']:.2f}\n"
                analysis += f"  Max: {stats['max']:.2f}\n"
                analysis += f"  Range: {stats['max'] - stats['min']:.2f}\n"
                analysis += f"  Skewness: {df[col].skew():.3f}\n"
                analysis += f"  Kurtosis: {df[col].kurtosis():.3f}\n"
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            analysis += f"\nCATEGORICAL VARIABLES ({len(categorical_cols)} columns):\n"
            analysis += f"{'='*50}\n"
            
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                analysis += f"\n{col}:\n"
                analysis += f"  Unique Values: {df[col].nunique()}\n"
                analysis += f"  Most Common: '{value_counts.index[0]}' ({value_counts.iloc[0]} occurrences)\n"
                analysis += f"  Least Common: '{value_counts.index[-1]}' ({value_counts.iloc[-1]} occurrences)\n"
                analysis += f"  Top 3 Values:\n"
                for i, (value, count) in enumerate(value_counts.head(3).items()):
                    percentage = (count / len(df)) * 100
                    analysis += f"    {i+1}. '{value}': {count} ({percentage:.1f}%)\n"
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            analysis += f"\nCORRELATION ANALYSIS:\n"
            analysis += f"{'='*30}\n"
            
            corr_matrix = df[numeric_cols].corr()
            strong_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        strong_correlations.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if strong_correlations:
                analysis += f"Strong Correlations (|r| > 0.5):\n"
                for col1, col2, corr in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
                    strength = "Strong" if abs(corr) > 0.7 else "Moderate"
                    direction = "Positive" if corr > 0 else "Negative"
                    analysis += f"  • {col1} ↔ {col2}: {corr:.3f} ({strength} {direction})\n"
            else:
                analysis += f"No strong correlations found between numeric variables.\n"
        
        # Geographic analysis
        geo_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['district', 'city', 'village', 'location', 'place'])]
        if geo_cols:
            analysis += f"\nGEOGRAPHIC DISTRIBUTION:\n"
            analysis += f"{'='*30}\n"
            
            for col in geo_cols:
                geo_counts = df[col].value_counts()
                analysis += f"\n{col}:\n"
                analysis += f"  Total Locations: {len(geo_counts)}\n"
                analysis += f"  Top Location: '{geo_counts.index[0]}' ({geo_counts.iloc[0]} records)\n"
                analysis += f"  Geographic Concentration: {(geo_counts.iloc[0] / len(df) * 100):.1f}% in top location\n"
        
        # Temporal analysis
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            analysis += f"\nTEMPORAL ANALYSIS:\n"
            analysis += f"{'='*20}\n"
            
            for col in date_cols:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    valid_dates = df[col].dropna()
                    if len(valid_dates) > 0:
                        analysis += f"\n{col}:\n"
                        analysis += f"  Valid Dates: {len(valid_dates)}\n"
                        analysis += f"  Date Range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}\n"
                        analysis += f"  Span: {(valid_dates.max() - valid_dates.min()).days} days\n"
                        
                        # Monthly distribution
                        monthly_counts = valid_dates.dt.month.value_counts()
                        if len(monthly_counts) > 0:
                            peak_month = monthly_counts.idxmax()
                            analysis += f"  Peak Month: {peak_month} ({monthly_counts.max()} records)\n"
                except:
                    pass
        
        return analysis
        
    except Exception as e:
        return f"Error performing statistical analysis on {file_path}: {str(e)}"


@tool("generate_insights")
def generate_insights_tool(file_path: str) -> str:
    """
    Generate actionable insights and recommendations from dataset analysis.
    
    Args:
        file_path: Path to the dataset file
    
    Returns:
        String containing insights and recommendations
    """
    try:
        # Read dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return f"Error: Unsupported file format for {file_path}"
        
        insights = f"""
ACTIONABLE INSIGHTS & RECOMMENDATIONS:
=====================================
Dataset: {file_path}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY FINDINGS:
=============
"""
        
        # Data quality insights
        missing_data = df.isnull().sum()
        total_missing = missing_data.sum()
        if total_missing > 0:
            missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100
            insights += f"• Data Quality Issue: {total_missing:,} missing values ({missing_percentage:.1f}% of all data)\n"
            insights += f"  Recommendation: Implement data validation and collection improvements\n"
        else:
            insights += f"• Data Quality: Excellent - no missing values detected\n"
        
        # Geographic insights
        geo_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['district', 'city', 'village', 'location', 'place'])]
        for col in geo_cols:
            geo_counts = df[col].value_counts()
            if len(geo_counts) > 0:
                top_location = geo_counts.iloc[0]
                percentage = (top_location / len(df)) * 100
                if percentage > 50:
                    insights += f"• Geographic Concentration: {percentage:.1f}% of records in '{geo_counts.index[0]}'\n"
                    insights += f"  Recommendation: Consider expanding operations to underserved areas\n"
        
        # Sector/Industry insights
        sector_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sector', 'industry', 'type', 'category'])]
        for col in sector_cols:
            sector_counts = df[col].value_counts()
            if len(sector_counts) > 0:
                dominant_sector = sector_counts.iloc[0]
                percentage = (dominant_sector / len(df)) * 100
                if percentage > 70:
                    insights += f"• Sector Concentration: {percentage:.1f}% in '{sector_counts.index[0]}' sector\n"
                    insights += f"  Recommendation: Diversify sector focus to reduce risk\n"
        
        # Investment/Financial insights
        investment_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['investment', 'revenue', 'cost', 'amount', 'value'])]
        for col in investment_cols:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                total_investment = df[col].sum()
                avg_investment = df[col].mean()
                max_investment = df[col].max()
                min_investment = df[col].min()
                
                insights += f"• Investment Analysis ({col}):\n"
                insights += f"  Total: {total_investment:,.2f}\n"
                insights += f"  Average: {avg_investment:,.2f}\n"
                insights += f"  Range: {min_investment:,.2f} to {max_investment:,.2f}\n"
                
                # Investment distribution insights
                high_investment = df[df[col] > df[col].quantile(0.8)]
                if len(high_investment) > 0:
                    insights += f"  Top 20% investments: {len(high_investment)} records\n"
        
        # Employment insights
        employee_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['employee', 'worker', 'staff', 'personnel'])]
        for col in employee_cols:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                total_employees = df[col].sum()
                avg_employees = df[col].mean()
                
                insights += f"• Employment Analysis ({col}):\n"
                insights += f"  Total Employees: {total_employees:,.0f}\n"
                insights += f"  Average per Unit: {avg_employees:.1f}\n"
                
                # Employment distribution
                large_employers = df[df[col] > df[col].quantile(0.9)]
                if len(large_employers) > 0:
                    insights += f"  Large Employers (>90th percentile): {len(large_employers)} units\n"
        
        # Temporal insights
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                valid_dates = df[col].dropna()
                if len(valid_dates) > 0:
                    recent_records = valid_dates[valid_dates >= valid_dates.max() - pd.Timedelta(days=30)]
                    insights += f"• Recent Activity ({col}): {len(recent_records)} records in last 30 days\n"
            except:
                pass
        
        # Policy recommendations
        insights += f"\nPOLICY RECOMMENDATIONS:\n"
        insights += f"=====================\n"
        
        # Data quality recommendations
        if total_missing > 0:
            insights += f"1. Data Quality Improvement:\n"
            insights += f"   • Implement mandatory field validation\n"
            insights += f"   • Establish data collection standards\n"
            insights += f"   • Create data quality monitoring dashboard\n\n"
        
        # Geographic equity recommendations
        if geo_cols:
            insights += f"2. Geographic Equity:\n"
            insights += f"   • Identify underserved regions for expansion\n"
            insights += f"   • Implement regional development programs\n"
            insights += f"   • Monitor geographic distribution metrics\n\n"
        
        # Sector diversification recommendations
        if sector_cols:
            insights += f"3. Sector Diversification:\n"
            insights += f"   • Encourage growth in underrepresented sectors\n"
            insights += f"   • Provide sector-specific incentives\n"
            insights += f"   • Monitor sector concentration ratios\n\n"
        
        # Process efficiency recommendations
        insights += f"4. Process Efficiency:\n"
        insights += f"   • Implement automated data processing\n"
        insights += f"   • Establish standard operating procedures\n"
        insights += f"   • Create performance monitoring systems\n\n"
        
        # Success metrics
        insights += f"SUCCESS METRICS TO TRACK:\n"
        insights += f"=======================\n"
        insights += f"• Data Completeness: Target 95%+ field completion\n"
        insights += f"• Geographic Distribution: Reduce concentration in top location to <40%\n"
        insights += f"• Sector Diversification: No single sector >60% of total\n"
        insights += f"• Processing Time: Reduce average processing time by 25%\n"
        insights += f"• Data Quality Score: Maintain >90% quality rating\n"
        
        return insights
        
    except Exception as e:
        return f"Error generating insights for {file_path}: {str(e)}"


@tool("create_visualization")
def create_visualization_tool(file_path: str, chart_type: str = "auto", output_path: str = None) -> str:
    """
    Create data visualizations and save them as image files.
    
    Args:
        file_path: Path to the dataset file
        chart_type: Type of chart to create (auto, distribution, correlation, geographic, temporal)
        output_path: Path to save the chart image (optional)
    
    Returns:
        String containing visualization report and file path
    """
    try:
        # Read dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return f"Error: Unsupported file format for {file_path}"
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"visualization_{timestamp}.png"
        
        # Set style
        plt.style.use('default')
        
        report = f"""
VISUALIZATION REPORT:
====================
Dataset: {file_path}
Chart Type: {chart_type}
Output File: {output_path}

VISUALIZATION CREATED:
=====================
"""
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if chart_type == "auto" or chart_type == "distribution":
            # Create distribution plots
            if len(numeric_cols) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'Data Distribution Analysis - {Path(file_path).stem}', fontsize=16, fontweight='bold')
                
                # Plot distributions for first 4 numeric columns
                for i, col in enumerate(numeric_cols[:4]):
                    row = i // 2
                    col_idx = i % 2
                    
                    if row < 2 and col_idx < 2:
                        axes[row, col_idx].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                        axes[row, col_idx].set_title(f'Distribution of {col}')
                        axes[row, col_idx].set_xlabel(col)
                        axes[row, col_idx].set_ylabel('Frequency')
                
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                report += f"• Created distribution plots for {min(4, len(numeric_cols))} numeric variables\n"
        
        if chart_type == "auto" or chart_type == "correlation":
            # Create correlation matrix
            if len(numeric_cols) > 1:
                plt.figure(figsize=(12, 10))
                corr_matrix = df[numeric_cols].corr()
                
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
                plt.title(f'Correlation Matrix - {Path(file_path).stem}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                corr_output = output_path.replace('.png', '_correlation.png')
                plt.savefig(corr_output, dpi=300, bbox_inches='tight')
                plt.close()
                
                report += f"• Created correlation matrix heatmap\n"
        
        if chart_type == "auto" or chart_type == "geographic":
            # Create geographic distribution plots
            geo_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['district', 'city', 'village', 'location', 'place'])]
            
            if geo_cols and len(categorical_cols) > 0:
                geo_col = geo_cols[0]
                geo_counts = df[geo_col].value_counts().head(10)
                
                plt.figure(figsize=(12, 8))
                geo_counts.plot(kind='bar')
                plt.title(f'Geographic Distribution - {geo_col}', fontsize=14, fontweight='bold')
                plt.xlabel(geo_col)
                plt.ylabel('Number of Records')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                geo_output = output_path.replace('.png', '_geographic.png')
                plt.savefig(geo_output, dpi=300, bbox_inches='tight')
                plt.close()
                
                report += f"• Created geographic distribution chart for {geo_col}\n"
        
        if chart_type == "auto" or chart_type == "temporal":
            # Create temporal plots
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            
            if date_cols:
                date_col = date_cols[0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    valid_dates = df[date_col].dropna()
                    
                    if len(valid_dates) > 0:
                        monthly_counts = valid_dates.dt.to_period('M').value_counts().sort_index()
                        
                        plt.figure(figsize=(12, 6))
                        monthly_counts.plot(kind='line', marker='o')
                        plt.title(f'Temporal Trends - {date_col}', fontsize=14, fontweight='bold')
                        plt.xlabel('Month')
                        plt.ylabel('Number of Records')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        temporal_output = output_path.replace('.png', '_temporal.png')
                        plt.savefig(temporal_output, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        report += f"• Created temporal trend chart for {date_col}\n"
                except:
                    pass
        
        report += f"""
VISUALIZATION SUMMARY:
=====================
Chart saved to: {output_path}
Chart type: {chart_type}
Dataset records: {len(df):,}
Dataset columns: {len(df.columns)}
"""
        
        return report
        
    except Exception as e:
        return f"Error creating visualization for {file_path}: {str(e)}"


@tool("save_analysis_report")
def save_analysis_report_tool(analysis_content: str, dataset_name: str, output_dir: str = "output") -> str:
    """
    Save analysis results to a file.
    
    Args:
        analysis_content: The analysis content to save
        dataset_name: Name of the dataset
        output_dir: Output directory (optional)
    
    Returns:
        String containing save confirmation and file path
    """
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_report_{dataset_name}_{timestamp}.txt"
        file_path = output_path / filename
        
        # Save content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(analysis_content)
        
        return f"Analysis report saved successfully!\nFile: {file_path}\nSize: {file_path.stat().st_size} bytes\nTimestamp: {timestamp}"
        
    except Exception as e:
        return f"Error saving analysis report: {str(e)}"


# Export all tools for use in agents
ALL_TOOLS = [
    read_dataset_tool,
    analyze_data_quality_tool,
    clean_dataset_tool,
    transform_dataset_tool,
    analyze_statistics_tool,
    generate_insights_tool,
    create_visualization_tool,
    save_analysis_report_tool
]
