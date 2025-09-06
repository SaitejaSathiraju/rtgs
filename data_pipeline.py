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
        
        # 2. Handle outliers (CONSERVATIVE APPROACH - Only flag, don't modify)
        self.console.print("🧹 Analyzing outliers...")
        outliers_detected = 0
        outlier_info = []
        
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Only flag extreme outliers (beyond 3*IQR) for review
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
            if outliers > 0:
                outliers_detected += outliers
                outlier_info.append(f"{col}: {outliers} extreme outliers")
        
        if outliers_detected > 0:
            self.console.print(f"⚠️ Extreme outliers detected: {outliers_detected:,} values flagged for review")
            for info in outlier_info:
                self.console.print(f"   - {info}")
            self.console.print("✅ Data integrity preserved - no values modified")
        else:
            self.console.print("✅ No extreme outliers detected")
        
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
        """Transform the data - 100% DYNAMIC, NO hardcoded column names."""
        transformed_df = df.copy()
        
        # 1. Create derived features for ALL numeric columns
        self.console.print("🔄 Creating derived features...")
        
        numeric_cols = transformed_df.select_dtypes(include=[np.number]).columns
        features_created = 0
        
        for col in numeric_cols:
            if transformed_df[col].dtype in ['int64', 'float64']:
                try:
                    # Dynamic binning based on actual data distribution
                    q1, q2, q3 = transformed_df[col].quantile([0.25, 0.5, 0.75])
                    bins = [-np.inf, q1, q2, q3, np.inf]
                    labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
                    
                    # Ensure unique bins and matching labels
                    unique_bins = []
                    unique_labels = []
                    for i, bin_val in enumerate(bins[:-1]):
                        if bin_val not in unique_bins:
                            unique_bins.append(bin_val)
                            if i < len(labels):
                                unique_labels.append(labels[i])
                    
                    unique_bins.append(np.inf)
                    
                    if len(unique_bins) > 1:
                        # Create category feature
                        transformed_df[f'{col}_category'] = pd.cut(
                            transformed_df[col], 
                            bins=unique_bins, 
                            labels=unique_labels, 
                            duplicates='drop'
                        )
                        
                        # Create intensity feature (normalized by mean)
                        transformed_df[f'{col}_intensity'] = transformed_df[col] / transformed_df[col].mean()
                        
                        # Create zero indicator
                        transformed_df[f'{col}_zero_indicator'] = (transformed_df[col] == 0).astype(int)
                        
                        features_created += 3
                        self.console.print(f"✅ Created dynamic features for {col}: {col}_category, {col}_intensity, {col}_zero_indicator")
                        
                except Exception as e:
                    self.console.print(f"⚠️ Could not create features for {col}: {str(e)}")
                    continue
        
        self.console.print(f"✅ Total derived features created: {features_created}")
        
        # 2. Create aggregation features
        self.console.print("🔄 Creating aggregation features...")
        
        categorical_cols = transformed_df.select_dtypes(include=['object']).columns
        numeric_cols = transformed_df.select_dtypes(include=['number']).columns
        
        aggregation_count = 0
        for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if num_col != cat_col:
                    try:
                        # Group statistics
                        group_stats = transformed_df.groupby(cat_col)[num_col].agg(['mean', 'sum', 'count']).reset_index()
                        group_stats.columns = [cat_col, f'{num_col}_group_mean', f'{num_col}_group_sum', f'{num_col}_group_count']
                        
                        # Merge back
                        transformed_df = transformed_df.merge(group_stats, on=cat_col, how='left')
                        aggregation_count += 3
                    except Exception as e:
                        self.console.print(f"⚠️ Could not create aggregation for {cat_col} x {num_col}: {str(e)}")
                        continue
        
        self.console.print(f"✅ Aggregation features created: {aggregation_count}")
        
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
        
        # 2. Dynamic domain-specific charts
        charts_info.update(self._generate_dynamic_charts(raw_df, cleaned_df, standardized_df, transformed_df, dataset_name, timestamp))
        
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
    
    def _generate_dynamic_charts(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                               standardized_df: pd.DataFrame, transformed_df: pd.DataFrame,
                               dataset_name: str, timestamp: str) -> Dict[str, str]:
        """Generate dynamic charts based on actual data - NO hardcoded assumptions."""
        charts_info = {}
        
        # Get numeric columns for analysis
        numeric_cols = raw_df.select_dtypes(include=['number']).columns
        categorical_cols = raw_df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) == 0:
            return charts_info
        
        self.console.print("📊 Generating dynamic analysis charts...")
        
        # 1. Top numeric columns analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Dynamic Data Analysis - {dataset_name}', fontsize=16, fontweight='bold')
        
        # Use first numeric column for distribution analysis
        main_col = numeric_cols[0]
        
        # Distribution comparison
        axes[0, 0].hist(raw_df[main_col].dropna(), alpha=0.7, label='Raw Data', bins=30, color='red')
        axes[0, 0].hist(cleaned_df[main_col].dropna(), alpha=0.7, label='Cleaned Data', bins=30, color='green')
        axes[0, 0].set_title(f'Distribution Comparison - {main_col}')
        axes[0, 0].set_xlabel(main_col)
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Box plot comparison
        data_to_plot = [raw_df[main_col].dropna(), cleaned_df[main_col].dropna()]
        axes[0, 1].boxplot(data_to_plot, labels=['Raw', 'Cleaned'])
        axes[0, 1].set_title(f'Distribution (Box Plot) - {main_col}')
        axes[0, 1].set_ylabel(main_col)
        
        # Top categories analysis (if categorical columns exist)
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
            if len(numeric_cols) > 0:
                top_categories = raw_df.groupby(cat_col)[main_col].sum().sort_values(ascending=False).head(10)
                axes[1, 0].bar(range(len(top_categories)), top_categories.values)
                axes[1, 0].set_title(f'Top 10 Categories by {main_col}')
                axes[1, 0].set_xlabel('Category Rank')
                axes[1, 0].set_ylabel(f'Total {main_col}')
                axes[1, 0].set_xticks(range(len(top_categories)))
                axes[1, 0].set_xticklabels(top_categories.index, rotation=45)
        
        # Transformation effect
        if f'{main_col}_intensity' in transformed_df.columns:
            axes[1, 1].hist(transformed_df[f'{main_col}_intensity'].dropna(), bins=30, alpha=0.7, color='purple')
            axes[1, 1].set_title(f'Intensity Distribution - {main_col}')
            axes[1, 1].set_xlabel('Intensity Ratio')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        dynamic_chart = self.charts_dir / f"{dataset_name}_dynamic_analysis_{timestamp}.png"
        plt.savefig(dynamic_chart, dpi=300, bbox_inches='tight')
        plt.close()
        charts_info['dynamic_analysis'] = str(dynamic_chart)
        
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


