"""
AI Disease Detection Agents Package

This package contains specialized agents for different aspects of disease detection:
- Input Collection Agent: Handles user input gathering and validation
- Preprocessing Agent: Cleans and extracts relevant information
- Prediction Agent: Makes disease predictions using ML models
- Explainability Agent: Provides explanations for predictions
- Report Generator Agent: Creates comprehensive health reports
- Feedback Agent: Handles user feedback collection
"""

from .base_agent import BaseAgent
from .input_collection_agent import InputCollectionAgent
from .preprocessing_agent import PreprocessingAgent
from .prediction_agent import PredictionAgent
from .explainability_agent import ExplainabilityAgent
from .report_generator_agent import ReportGeneratorAgent
from .feedback_agent import FeedbackAgent

__all__ = [
    'BaseAgent',
    'InputCollectionAgent',
    'PreprocessingAgent',
    'PredictionAgent',
    'ExplainabilityAgent',
    'ReportGeneratorAgent',
    'FeedbackAgent'
]
