#!/usr/bin/env python3
"""
Complete Data Pipeline with Before/After Visibility
Generates: Raw → Cleaned → Standardized → Transformed data files + Plot Charts
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import warnings
warnings.filterwarnings('ignore')

class DataPipeline:
    """Complete data pipeline with before/after visibility."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.raw_dir = self.output_dir / "raw_data"
        self.cleaned_dir = self.output_dir / "cleaned_data"
        self.standardized_dir = self.output_dir / "standardized_data"
        self.transformed_dir = self.output_dir / "transformed_data"
        self.charts_dir = self.output_dir / "charts"
        
        for dir_path in [self.raw_dir, self.cleaned_dir, self.standardized_dir, self.transformed_dir, self.charts_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.console = Console()
        
    def process_dataset(self, file_path: str) -> Dict[str, Any]:
        """Process dataset through complete pipeline."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = Path(file_path).stem
        
        self.console.print(Panel.fit(
            f"[bold blue]🔄 Data Pipeline Processing[/bold blue]\n"
            f"Dataset: [green]{dataset_name}[/green]\n"
            f"Timestamp: [yellow]{timestamp}[/yellow]",
            title="📊 Pipeline Status",
            border_style="blue"
        ))
        
        # Step 1: Load Raw Data
        self.console.print("\n[bold green]📁 Step 1: Loading Raw Data[/bold green]")
        raw_df = self._load_raw_data(file_path)
        if raw_df is None:
            return {"error": "Failed to load raw data"}
        
        # Save raw data
        raw_file = self.raw_dir / f"{dataset_name}_raw_{timestamp}.csv"
        raw_df.to_csv(raw_file, index=False)
        self.console.print(f"✅ Raw data saved: {raw_file}")
        
        # Step 2: Clean Data
        self.console.print("\n[bold green]🧹 Step 2: Cleaning Data[/bold green]")
        cleaned_df = self._clean_data(raw_df)
        cleaned_file = self.cleaned_dir / f"{dataset_name}_cleaned_{timestamp}.csv"
        cleaned_df.to_csv(cleaned_file, index=False)
        self.console.print(f"✅ Cleaned data saved: {cleaned_file}")
        
        # Step 3: Standardize Data
        self.console.print("\n[bold green]📏 Step 3: Standardizing Data[/bold green]")
        standardized_df = self._standardize_data(cleaned_df)
        standardized_file = self.standardized_dir / f"{dataset_name}_standardized_{timestamp}.csv"
        standardized_df.to_csv(standardized_file, index=False)
        self.console.print(f"✅ Standardized data saved: {standardized_file}")
        
        # Step 4: Transform Data
        self.console.print("\n[bold green]🔄 Step 4: Transforming Data[/bold green]")
        transformed_df = self._transform_data(standardized_df)
        transformed_file = self.transformed_dir / f"{dataset_name}_transformed_{timestamp}.csv"
        transformed_df.to_csv(transformed_file, index=False)
        self.console.print(f"✅ Transformed data saved: {transformed_file}")
        
        # Step 5: Generate Charts
        self.console.print("\n[bold green]📊 Step 5: Generating Charts[/bold green]")
        charts_info = self._generate_charts(raw_df, cleaned_df, standardized_df, transformed_df, dataset_name, timestamp)
        
        # Generate Summary Report
        summary = self._generate_summary_report(raw_df, cleaned_df, standardized_df, transformed_df, dataset_name, timestamp)
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "files": {
                "raw": str(raw_file),
                "cleaned": str(cleaned_file),
                "standardized": str(standardized_file),
                "transformed": str(transformed_file)
            },
            "charts": charts_info,
            "summary": summary
        }
    
    def _load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Load raw data from file."""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self.console.print(f"📊 Raw Data: {len(df):,} records × {len(df.columns)} columns")
            self.console.print(f"📋 Columns: {', '.join(df.columns)}")
            
            # Show data quality issues
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                self.console.print(f"⚠️ Missing Data: {missing_data.sum():,} missing values")
                for col, count in missing_data[missing_data > 0].items():
                    self.console.print(f"   • {col}: {count:,} missing")
            
            return df
            
        except Exception as e:
            self.console.print(f"[red]❌ Error loading raw data: {str(e)}[/red]")
            return None
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the data."""
        cleaned_df = df.copy()
        
        # 1. Handle missing values
        self.console.print("🧹 Handling missing values...")
        missing_before = cleaned_df.isnull().sum().sum()
        
        # For numeric columns: fill with median
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().sum() > 0:
                median_val = cleaned_df[col].median()
                cleaned_df[col].fillna(median_val, inplace=True)
        
        # For categorical columns: fill with mode
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if cleaned_df[col].isnull().sum() > 0:
                mode_val = cleaned_df[col].mode()[0] if len(cleaned_df[col].mode()) > 0 else 'Unknown'
                cleaned_df[col].fillna(mode_val, inplace=True)
        
        missing_after = cleaned_df.isnull().sum().sum()
        self.console.print(f"✅ Missing values: {missing_before:,} → {missing_after:,}")
        
        # 2. Handle outliers
        self.console.print("🧹 Handling outliers...")
        outliers_removed = 0
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
            if outliers > 0:
                # Cap outliers instead of removing
                cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
                outliers_removed += outliers
        
        self.console.print(f"✅ Outliers handled: {outliers_removed:,} values capped")
        
        # 3. Standardize text data
        self.console.print("🧹 Standardizing text data...")
        for col in categorical_cols:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.upper()
        
        self.console.print(f"✅ Text standardized: {len(categorical_cols)} columns")
        
        return cleaned_df
    
    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the data."""
        standardized_df = df.copy()
        
        # 1. Standardize numeric columns (Z-score normalization)
        self.console.print("📏 Standardizing numeric columns...")
        numeric_cols = standardized_df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            mean_val = standardized_df[col].mean()
            std_val = standardized_df[col].std()
            if std_val > 0:  # Avoid division by zero
                standardized_df[f"{col}_standardized"] = (standardized_df[col] - mean_val) / std_val
        
        self.console.print(f"✅ Numeric columns standardized: {len(numeric_cols)} columns")
        
        # 2. Create categorical encodings
        self.console.print("📏 Creating categorical encodings...")
        categorical_cols = standardized_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Label encoding
            unique_values = standardized_df[col].unique()
            label_map = {val: idx for idx, val in enumerate(unique_values)}
            standardized_df[f"{col}_encoded"] = standardized_df[col].map(label_map)
        
        self.console.print(f"✅ Categorical columns encoded: {len(categorical_cols)} columns")
        
        # 3. Create date features if date column exists
        date_cols = [col for col in standardized_df.columns if 'date' in col.lower()]
        if date_cols:
            self.console.print("📏 Creating date features...")
            for col in date_cols:
                try:
                    standardized_df[col] = pd.to_datetime(standardized_df[col], errors='coerce')
                    standardized_df[f"{col}_year"] = standardized_df[col].dt.year
                    standardized_df[f"{col}_month"] = standardized_df[col].dt.month
                    standardized_df[f"{col}_day"] = standardized_df[col].dt.day
                    standardized_df[f"{col}_dayofweek"] = standardized_df[col].dt.dayofweek
                except:
                    pass
            self.console.print(f"✅ Date features created: {len(date_cols)} columns")
        
        return standardized_df
    
    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        transformed_df = df.copy()
        
        # 1. Create derived features
        self.console.print("🔄 Creating derived features...")
        
        # For rainfall data
        if 'rain' in ' '.join(transformed_df.columns).lower():
            rain_col = None
            for col in transformed_df.columns:
                if 'rain' in col.lower() and '_' not in col:
                    rain_col = col
                    break
            
            if rain_col:
                # Rainfall categories
                bins = [-np.inf, 0, 5, 20, 50, np.inf]
                labels = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain', 'Extreme Rain']
                
                # Ensure labels match number of bins (labels = bins - 1)
                if len(labels) != len(bins) - 1:
                    labels = labels[:len(bins)-1]
                
                # Handle duplicates by removing them from bins and labels
                unique_bins = []
                unique_labels = []
                for i, bin_val in enumerate(bins[:-1]):
                    if bin_val not in unique_bins:
                        unique_bins.append(bin_val)
                        if i < len(labels):
                            unique_labels.append(labels[i])
                unique_bins.append(bins[-1])  # Add the last bin edge
                
                transformed_df['rainfall_category'] = pd.cut(
                    transformed_df[rain_col],
                    bins=unique_bins,
                    labels=unique_labels
                )
                
                # Drought indicator
                transformed_df['drought_indicator'] = (transformed_df[rain_col] == 0).astype(int)
                
                # Rainfall intensity
                transformed_df['rainfall_intensity'] = transformed_df[rain_col] / transformed_df[rain_col].mean()
                
                self.console.print(f"✅ Rainfall features created: 3 new features")
        
        # For consumption data
        if 'units' in ' '.join(transformed_df.columns).lower():
            units_col = None
            for col in transformed_df.columns:
                if 'units' in col.lower() and '_' not in col:
                    units_col = col
                    break
            
            if units_col:
                # Consumption categories - handle case where all values are the same
                q25 = transformed_df[units_col].quantile(0.25)
                q75 = transformed_df[units_col].quantile(0.75)
                
                # Create bins, ensuring no duplicates and matching labels
                if q25 == q75:  # All values are the same
                    bins = [-np.inf, 0, q25, np.inf]
                    labels = ['No Consumption', 'Low Consumption', 'High Consumption']
                else:
                    bins = [-np.inf, 0, q25, q75, np.inf]
                    labels = ['No Consumption', 'Low Consumption', 'Medium Consumption', 'High Consumption']
                
                # Ensure labels match number of bins (labels = bins - 1)
                if len(labels) != len(bins) - 1:
                    labels = labels[:len(bins)-1]
                
                # Handle duplicates by removing them from bins and labels
                unique_bins = []
                unique_labels = []
                for i, bin_val in enumerate(bins[:-1]):
                    if bin_val not in unique_bins:
                        unique_bins.append(bin_val)
                        if i < len(labels):
                            unique_labels.append(labels[i])
                unique_bins.append(bins[-1])  # Add the last bin edge
                
                transformed_df['consumption_category'] = pd.cut(
                    transformed_df[units_col],
                    bins=unique_bins,
                    labels=unique_labels
                )
                
                # Outage indicator
                transformed_df['outage_indicator'] = (transformed_df[units_col] == 0).astype(int)
                
                # Consumption efficiency
                transformed_df['consumption_efficiency'] = transformed_df[units_col] / transformed_df[units_col].mean()
                
                self.console.print(f"✅ Consumption features created: 3 new features")
        
        # 2. Create aggregation features
        self.console.print("🔄 Creating aggregation features...")
        
        # Group by categorical columns and create aggregations
        categorical_cols = transformed_df.select_dtypes(include=['object']).columns
        numeric_cols = transformed_df.select_dtypes(include=['number']).columns
        
        for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if num_col != cat_col:
                    # Group statistics
                    group_stats = transformed_df.groupby(cat_col)[num_col].agg(['mean', 'sum', 'count']).reset_index()
                    group_stats.columns = [cat_col, f'{num_col}_group_mean', f'{num_col}_group_sum', f'{num_col}_group_count']
                    
                    # Merge back
                    transformed_df = transformed_df.merge(group_stats, on=cat_col, how='left')
        
        self.console.print(f"✅ Aggregation features created: {len(categorical_cols) * len(numeric_cols)} features")
        
        return transformed_df
    
    def _generate_charts(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                        standardized_df: pd.DataFrame, transformed_df: pd.DataFrame,
                        dataset_name: str, timestamp: str) -> Dict[str, str]:
        """Generate before/after comparison charts."""
        charts_info = {}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Data Quality Chart
        self.console.print("📊 Generating data quality chart...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Data Quality Analysis - {dataset_name}', fontsize=16, fontweight='bold')
        
        # Missing data comparison
        missing_raw = raw_df.isnull().sum()
        missing_cleaned = cleaned_df.isnull().sum()
        
        axes[0, 0].bar(range(len(missing_raw)), missing_raw.values, alpha=0.7, label='Raw Data', color='red')
        axes[0, 0].bar(range(len(missing_cleaned)), missing_cleaned.values, alpha=0.7, label='Cleaned Data', color='green')
        axes[0, 0].set_title('Missing Data Comparison')
        axes[0, 0].set_xlabel('Columns')
        axes[0, 0].set_ylabel('Missing Values')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Data distribution comparison
        numeric_cols = raw_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            axes[0, 1].hist(raw_df[col].dropna(), alpha=0.7, label='Raw Data', bins=30, color='red')
            axes[0, 1].hist(cleaned_df[col].dropna(), alpha=0.7, label='Cleaned Data', bins=30, color='green')
            axes[0, 1].set_title(f'Distribution Comparison - {col}')
            axes[0, 1].set_xlabel(col)
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
        
        # Standardization effect
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            standardized_col = f"{col}_standardized"
            if standardized_col in standardized_df.columns:
                axes[1, 0].hist(standardized_df[col].dropna(), alpha=0.7, label='Original', bins=30, color='blue')
                axes[1, 0].hist(standardized_df[standardized_col].dropna(), alpha=0.7, label='Standardized', bins=30, color='orange')
                axes[1, 0].set_title(f'Standardization Effect - {col}')
                axes[1, 0].set_xlabel('Value')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].legend()
        
        # Transformation effect
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            axes[1, 1].hist(cleaned_df[col].dropna(), alpha=0.7, label='Cleaned', bins=30, color='green')
            axes[1, 1].hist(transformed_df[col].dropna(), alpha=0.7, label='Transformed', bins=30, color='purple')
            axes[1, 1].set_title(f'Transformation Effect - {col}')
            axes[1, 1].set_xlabel(col)
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        plt.tight_layout()
        quality_chart = self.charts_dir / f"{dataset_name}_data_quality_{timestamp}.png"
        plt.savefig(quality_chart, dpi=300, bbox_inches='tight')
        plt.close()
        charts_info['data_quality'] = str(quality_chart)
        
        # 2. Domain-specific charts
        if 'rain' in ' '.join(raw_df.columns).lower():
            charts_info.update(self._generate_rainfall_charts(raw_df, cleaned_df, standardized_df, transformed_df, dataset_name, timestamp))
        elif 'units' in ' '.join(raw_df.columns).lower():
            charts_info.update(self._generate_consumption_charts(raw_df, cleaned_df, standardized_df, transformed_df, dataset_name, timestamp))
        
        # 3. Correlation matrix
        self.console.print("📊 Generating correlation matrix...")
        numeric_cols = transformed_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = transformed_df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title(f'Correlation Matrix - {dataset_name} (Transformed Data)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            corr_chart = self.charts_dir / f"{dataset_name}_correlation_{timestamp}.png"
            plt.savefig(corr_chart, dpi=300, bbox_inches='tight')
            plt.close()
            charts_info['correlation'] = str(corr_chart)
        
        return charts_info
    
    def _generate_rainfall_charts(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                                 standardized_df: pd.DataFrame, transformed_df: pd.DataFrame,
                                 dataset_name: str, timestamp: str) -> Dict[str, str]:
        """Generate rainfall-specific charts."""
        charts_info = {}
        
        rain_col = None
        for col in raw_df.columns:
            if 'rain' in col.lower() and '_' not in col:
                rain_col = col
                break
        
        if not rain_col:
            return charts_info
        
        # Rainfall distribution comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Rainfall Analysis - {dataset_name}', fontsize=16, fontweight='bold')
        
        # Before/After distribution
        axes[0, 0].hist(raw_df[rain_col].dropna(), alpha=0.7, label='Raw Data', bins=30, color='red')
        axes[0, 0].hist(cleaned_df[rain_col].dropna(), alpha=0.7, label='Cleaned Data', bins=30, color='green')
        axes[0, 0].set_title('Rainfall Distribution Comparison')
        axes[0, 0].set_xlabel('Rainfall (mm)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Drought analysis
        if 'drought_indicator' in transformed_df.columns:
            drought_counts = transformed_df['drought_indicator'].value_counts()
            axes[0, 1].pie(drought_counts.values, labels=['No Drought', 'Drought'], autopct='%1.1f%%', 
                          colors=['lightgreen', 'red'])
            axes[0, 1].set_title('Drought Analysis')
        
        # Rainfall categories
        if 'rainfall_category' in transformed_df.columns:
            category_counts = transformed_df['rainfall_category'].value_counts()
            axes[1, 0].bar(range(len(category_counts)), category_counts.values, color='skyblue')
            axes[1, 0].set_title('Rainfall Categories')
            axes[1, 0].set_xlabel('Category')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_xticks(range(len(category_counts)))
            axes[1, 0].set_xticklabels(category_counts.index, rotation=45)
        
        # Time series if date column exists
        date_cols = [col for col in raw_df.columns if 'date' in col.lower()]
        if date_cols:
            try:
                date_col = date_cols[0]
                raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors='coerce')
                daily_rainfall = raw_df.groupby(raw_df[date_col].dt.date)[rain_col].mean()
                
                axes[1, 1].plot(daily_rainfall.index, daily_rainfall.values, color='blue', linewidth=2)
                axes[1, 1].set_title('Daily Rainfall Trend')
                axes[1, 1].set_xlabel('Date')
                axes[1, 1].set_ylabel('Average Rainfall (mm)')
                axes[1, 1].tick_params(axis='x', rotation=45)
            except:
                pass
        
        plt.tight_layout()
        rainfall_chart = self.charts_dir / f"{dataset_name}_rainfall_analysis_{timestamp}.png"
        plt.savefig(rainfall_chart, dpi=300, bbox_inches='tight')
        plt.close()
        charts_info['rainfall_analysis'] = str(rainfall_chart)
        
        return charts_info
    
    def _generate_consumption_charts(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                                   standardized_df: pd.DataFrame, transformed_df: pd.DataFrame,
                                   dataset_name: str, timestamp: str) -> Dict[str, str]:
        """Generate consumption-specific charts."""
        charts_info = {}
        
        units_col = None
        for col in raw_df.columns:
            if 'units' in col.lower() and '_' not in col:
                units_col = col
                break
        
        if not units_col:
            return charts_info
        
        # Consumption distribution comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Consumption Analysis - {dataset_name}', fontsize=16, fontweight='bold')
        
        # Before/After distribution
        axes[0, 0].hist(raw_df[units_col].dropna(), alpha=0.7, label='Raw Data', bins=30, color='red')
        axes[0, 0].hist(cleaned_df[units_col].dropna(), alpha=0.7, label='Cleaned Data', bins=30, color='green')
        axes[0, 0].set_title('Consumption Distribution Comparison')
        axes[0, 0].set_xlabel('Consumption (Units)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Outage analysis
        if 'outage_indicator' in transformed_df.columns:
            outage_counts = transformed_df['outage_indicator'].value_counts()
            # Create dynamic labels based on actual values
            labels = ['Active' if idx == 0 else 'Outage' for idx in outage_counts.index]
            colors = ['lightgreen' if idx == 0 else 'red' for idx in outage_counts.index]
            axes[0, 1].pie(outage_counts.values, labels=labels, autopct='%1.1f%%', colors=colors)
            axes[0, 1].set_title('Outage Analysis')
        
        # Consumption categories
        if 'consumption_category' in transformed_df.columns:
            category_counts = transformed_df['consumption_category'].value_counts()
            axes[1, 0].bar(range(len(category_counts)), category_counts.values, color='orange')
            axes[1, 0].set_title('Consumption Categories')
            axes[1, 0].set_xlabel('Category')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_xticks(range(len(category_counts)))
            axes[1, 0].set_xticklabels(category_counts.index, rotation=45)
        
        # Geographic distribution
        if 'circle' in raw_df.columns:
            circle_consumption = raw_df.groupby('circle')[units_col].sum().sort_values(ascending=False).head(10)
            axes[1, 1].barh(range(len(circle_consumption)), circle_consumption.values, color='purple')
            axes[1, 1].set_title('Top 10 Circles by Consumption')
            axes[1, 1].set_xlabel('Total Consumption (Units)')
            axes[1, 1].set_yticks(range(len(circle_consumption)))
            axes[1, 1].set_yticklabels(circle_consumption.index)
        
        plt.tight_layout()
        consumption_chart = self.charts_dir / f"{dataset_name}_consumption_analysis_{timestamp}.png"
        plt.savefig(consumption_chart, dpi=300, bbox_inches='tight')
        plt.close()
        charts_info['consumption_analysis'] = str(consumption_chart)
        
        return charts_info
    
    def _generate_summary_report(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                                standardized_df: pd.DataFrame, transformed_df: pd.DataFrame,
                                dataset_name: str, timestamp: str) -> Dict[str, Any]:
        """Generate summary report."""
        summary = {
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "pipeline_steps": {
                "raw_data": {
                    "records": len(raw_df),
                    "columns": len(raw_df.columns),
                    "missing_values": int(raw_df.isnull().sum().sum()),
                    "data_types": raw_df.dtypes.to_dict()
                },
                "cleaned_data": {
                    "records": len(cleaned_df),
                    "columns": len(cleaned_df.columns),
                    "missing_values": int(cleaned_df.isnull().sum().sum()),
                    "outliers_handled": "Yes"
                },
                "standardized_data": {
                    "records": len(standardized_df),
                    "columns": len(standardized_df.columns),
                    "numeric_standardized": len([col for col in standardized_df.columns if '_standardized' in col]),
                    "categorical_encoded": len([col for col in standardized_df.columns if '_encoded' in col])
                },
                "transformed_data": {
                    "records": len(transformed_df),
                    "columns": len(transformed_df.columns),
                    "derived_features": len(transformed_df.columns) - len(standardized_df.columns),
                    "aggregation_features": len([col for col in transformed_df.columns if '_group_' in col])
                }
            }
        }
        
        # Save summary report
        summary_file = self.output_dir / f"{dataset_name}_pipeline_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary


def main():
    """Main function to run the data pipeline."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_pipeline.py <dataset_file>")
        return
    
    file_path = sys.argv[1]
    pipeline = DataPipeline()
    
    result = pipeline.process_dataset(file_path)
    
    if result.get("status") == "success":
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Files generated:")
        for step, file_path in result["files"].items():
            print(f"   • {step}: {file_path}")
        print(f"📊 Charts generated:")
        for chart_type, chart_path in result["charts"].items():
            print(f"   • {chart_type}: {chart_path}")
    else:
        print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()


