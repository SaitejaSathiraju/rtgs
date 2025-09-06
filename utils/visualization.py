"""Visualization utilities for Telangana Open Data."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class VisualizationUtils:
    """Utility class for creating visualizations."""
    
    # Telangana color scheme
    TELANGANA_COLORS = {
        'primary': '#2c5aa0',
        'secondary': '#f4a261', 
        'accent': '#e76f51',
        'success': '#2a9d8f',
        'warning': '#f77f00',
        'danger': '#d62828'
    }
    
    @staticmethod
    def setup_plot_style():
        """Setup consistent plotting style."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette(list(VisualizationUtils.TELANGANA_COLORS.values()))
        
        # Set Telangana-specific styling
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
    
    @staticmethod
    def create_district_comparison_chart(df: pd.DataFrame, 
                                       value_col: str, 
                                       district_col: str = 'district',
                                       title: str = "District Comparison",
                                       save_path: Optional[str] = None) -> plt.Figure:
        """Create a district comparison chart."""
        VisualizationUtils.setup_plot_style()
        
        # Sort by value for better visualization
        df_sorted = df.sort_values(value_col, ascending=True)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create horizontal bar chart
        bars = ax.barh(df_sorted[district_col], df_sorted[value_col], 
                      color=VisualizationUtils.TELANGANA_COLORS['primary'])
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.1f}', ha='left', va='center')
        
        ax.set_xlabel(value_col.replace('_', ' ').title())
        ax.set_ylabel('District')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_trend_analysis(df: pd.DataFrame,
                             value_col: str,
                             time_col: str,
                             group_col: Optional[str] = None,
                             title: str = "Trend Analysis",
                             save_path: Optional[str] = None) -> plt.Figure:
        """Create trend analysis visualization."""
        VisualizationUtils.setup_plot_style()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if group_col:
            # Multiple lines for different groups
            for group in df[group_col].unique():
                group_data = df[df[group_col] == group]
                ax.plot(group_data[time_col], group_data[value_col], 
                       marker='o', label=group, linewidth=2)
            
            ax.legend()
        else:
            # Single line
            ax.plot(df[time_col], df[value_col], 
                   marker='o', color=VisualizationUtils.TELANGANA_COLORS['primary'],
                   linewidth=2)
        
        ax.set_xlabel(time_col.replace('_', ' ').title())
        ax.set_ylabel(value_col.replace('_', ' ').title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame,
                                  title: str = "Correlation Matrix",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Create correlation heatmap for numeric columns."""
        VisualizationUtils.setup_plot_style()
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_distribution_plot(df: pd.DataFrame,
                                column: str,
                                plot_type: str = 'histogram',
                                title: str = "Distribution Plot",
                                save_path: Optional[str] = None) -> plt.Figure:
        """Create distribution plot for a column."""
        VisualizationUtils.setup_plot_style()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == 'histogram':
            ax.hist(df[column].dropna(), bins=30, 
                   color=VisualizationUtils.TELANGANA_COLORS['primary'],
                   alpha=0.7, edgecolor='black')
        elif plot_type == 'boxplot':
            ax.boxplot(df[column].dropna())
        elif plot_type == 'violin':
            sns.violinplot(data=df, y=column, ax=ax)
        
        ax.set_xlabel(column.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_interactive_dashboard(df: pd.DataFrame,
                                   value_col: str,
                                   district_col: str = 'district',
                                   time_col: Optional[str] = None) -> go.Figure:
        """Create interactive dashboard using Plotly."""
        
        if time_col:
            # Time series dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('District Comparison', 'Trend Analysis', 
                               'Distribution', 'Correlation'),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "histogram"}, {"type": "heatmap"}]]
            )
            
            # District comparison
            district_avg = df.groupby(district_col)[value_col].mean().sort_values(ascending=True)
            fig.add_trace(
                go.Bar(x=district_avg.values, y=district_avg.index, 
                      name="District Average", orientation='h'),
                row=1, col=1
            )
            
            # Trend analysis
            if time_col in df.columns:
                time_series = df.groupby(time_col)[value_col].mean()
                fig.add_trace(
                    go.Scatter(x=time_series.index, y=time_series.values,
                              mode='lines+markers', name="Trend"),
                    row=1, col=2
                )
            
            # Distribution
            fig.add_trace(
                go.Histogram(x=df[value_col], name="Distribution"),
                row=2, col=1
            )
            
            # Correlation heatmap
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                fig.add_trace(
                    go.Heatmap(z=corr_matrix.values, 
                              x=corr_matrix.columns, 
                              y=corr_matrix.columns,
                              colorscale='RdYlBu'),
                    row=2, col=2
                )
        
        else:
            # Simple dashboard without time series
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('District Comparison', 'Distribution')
            )
            
            # District comparison
            district_avg = df.groupby(district_col)[value_col].mean().sort_values(ascending=True)
            fig.add_trace(
                go.Bar(x=district_avg.values, y=district_avg.index, 
                      name="District Average", orientation='h'),
                row=1, col=1
            )
            
            # Distribution
            fig.add_trace(
                go.Histogram(x=df[value_col], name="Distribution"),
                row=1, col=2
            )
        
        fig.update_layout(
            title_text="Telangana Data Analysis Dashboard",
            showlegend=False,
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_summary_statistics_table(df: pd.DataFrame,
                                       value_col: str,
                                       group_col: Optional[str] = None) -> pd.DataFrame:
        """Create summary statistics table."""
        
        if group_col:
            summary = df.groupby(group_col)[value_col].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
        else:
            summary = df[value_col].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).to_frame().T
        
        return summary

