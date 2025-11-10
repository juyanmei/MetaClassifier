"""
Results reporting utilities for metaClassifier.

This module contains utilities for generating comprehensive reports.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from ..utils.logger import get_logger


class ResultsReporter:
    """Reporter for metaClassifier results."""
    
    def __init__(self):
        self.logger = get_logger("ResultsReporter")
        
    def generate_report(self, 
                       results: Dict[str, Any], 
                       output_path: Union[str, Path]) -> None:
        """Generate a comprehensive HTML report."""
        self.logger.info(f"Generating report to {output_path}")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML report
        html_content = self._generate_html_report(results)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Report generated successfully: {output_path}")
        
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>metaClassifier Results Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                         background-color: #e8f4f8; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>metaClassifier Results Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                {self._generate_summary_section(results)}
            </div>
            
            <div class="section">
                <h2>Cross-Validation Results</h2>
                {self._generate_cv_section(results)}
            </div>
            
            <div class="section">
                <h2>Feature Selection</h2>
                {self._generate_feature_section(results)}
            </div>
            
            <div class="section">
                <h2>Model Performance</h2>
                {self._generate_performance_section(results)}
            </div>
            
        </body>
        </html>
        """
        
        return html
        
    def _generate_summary_section(self, results: Dict[str, Any]) -> str:
        """Generate summary section HTML."""
        summary = results.get('summary', {})
        
        html = f"""
        <div class="metric">
            <strong>Total Samples:</strong> {summary.get('total_samples', 'N/A')}
        </div>
        <div class="metric">
            <strong>CV Folds:</strong> {summary.get('n_folds', 'N/A')}
        </div>
        <div class="metric">
            <strong>Features:</strong> {summary.get('n_features', 'N/A')}
        </div>
        """
        
        return html
        
    def _generate_cv_section(self, results: Dict[str, Any]) -> str:
        """Generate cross-validation section HTML."""
        metrics = results.get('metrics', {})
        
        if not metrics:
            return "<p>No cross-validation metrics available.</p>"
        
        html = "<table><tr><th>Metric</th><th>Mean</th><th>Std</th></tr>"
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            mean_key = f"{metric_name}_mean"
            std_key = f"{metric_name}_std"
            
            if mean_key in metrics:
                mean_val = metrics[mean_key]
                std_val = metrics.get(std_key, 0)
                html += f"<tr><td>{metric_name.title()}</td><td>{mean_val:.4f}</td><td>{std_val:.4f}</td></tr>"
        
        html += "</table>"
        return html
        
    def _generate_feature_section(self, results: Dict[str, Any]) -> str:
        """Generate feature selection section HTML."""
        # This would be implemented based on the actual results structure
        return "<p>Feature selection details will be displayed here.</p>"
        
    def _generate_performance_section(self, results: Dict[str, Any]) -> str:
        """Generate model performance section HTML."""
        # This would be implemented based on the actual results structure
        return "<p>Model performance details will be displayed here.</p>"
        
    def save_results_json(self, 
                         results: Dict[str, Any], 
                         output_path: Union[str, Path]) -> None:
        """Save results as JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = self._make_json_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        self.logger.info(f"Results saved as JSON: {output_path}")
        
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
