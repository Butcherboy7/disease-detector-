"""
Report Generator Agent
Responsible for creating comprehensive, downloadable health reports
summarizing the entire analysis process and results.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List
from .base_agent import BaseAgent

class ReportGeneratorAgent(BaseAgent):
    """
    Agent responsible for generating comprehensive health assessment reports.
    Creates formatted, downloadable reports with all analysis results.
    """
    
    def __init__(self):
        super().__init__("ReportGeneratorAgent")
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive health assessment report.
        
        Args:
            data: Complete analysis data from all previous agents
            
        Returns:
            Generated report content and metadata
        """
        try:
            self.log_processing_step("Starting report generation")
            
            # Extract data from all previous agents
            report_data = self._extract_report_data(data)
            
            # Generate different report formats
            text_report = self._generate_text_report(report_data)
            summary_report = self._generate_summary_report(report_data)
            
            # Generate report metadata
            metadata = self._generate_report_metadata(report_data)
            
            self.log_processing_step("Report generation completed successfully")
            
            return self.create_success_response({
                'report_content': text_report,
                'summary_report': summary_report,
                'report_metadata': metadata,
                'generation_timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return self.handle_error(e, "report generation")
    
    def _extract_report_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and organize data from all agents for report generation."""
        self.log_processing_step("Extracting data for report")
        
        report_data = {
            'patient_info': {},
            'symptoms': {},
            'predictions': [],
            'explanations': {},
            'recommendations': [],
            'risk_assessment': {},
            'analysis_metadata': {}
        }
        
        # Extract collection data
        if 'collection_result' in data:
            collection = data['collection_result']['organized_data']
            report_data['patient_info'] = collection.get('patient_info', {})
            report_data['symptoms'] = collection.get('symptoms', {})
            report_data['analysis_metadata']['data_quality'] = data['collection_result'].get('data_quality_score', 0)
        
        # Extract preprocessing data
        if 'preprocessing_result' in data:
            preprocessing = data['preprocessing_result']['preprocessed_data']
            report_data['processed_symptoms'] = preprocessing.get('processed_symptoms', {})
            report_data['medical_features'] = preprocessing.get('medical_features', {})
        
        # Extract prediction data
        if 'prediction_result' in data:
            prediction = data['prediction_result']
            report_data['predictions'] = prediction.get('predictions', [])
            report_data['risk_assessment'] = prediction.get('overall_risk', {})
        
        # Extract explanation data
        if 'explanation_result' in data:
            explanation = data['explanation_result']
            report_data['explanations'] = {
                'main_explanation': explanation.get('explanation', ''),
                'risk_factors': explanation.get('risk_factors_explanation', ''),
                'next_steps': explanation.get('next_steps', [])
            }
            report_data['recommendations'] = explanation.get('recommendations', [])
        
        return report_data
    
    def _generate_text_report(self, report_data: Dict[str, Any]) -> str:
        """Generate a comprehensive text report."""
        self.log_processing_step("Generating text report")
        
        report = []
        
        # Header
        report.append("="*60)
        report.append("        AI-POWERED HEALTH ASSESSMENT REPORT")
        report.append("="*60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Patient Information
        report.append("PATIENT INFORMATION")
        report.append("-" * 20)
        patient_info = report_data.get('patient_info', {})
        report.append(f"Age: {patient_info.get('age', 'Not provided')}")
        report.append(f"Gender: {patient_info.get('gender', 'Not provided')}")
        
        existing_conditions = patient_info.get('existing_conditions', [])
        if existing_conditions and existing_conditions != ['None']:
            report.append(f"Existing Conditions: {', '.join(existing_conditions)}")
        else:
            report.append("Existing Conditions: None reported")
        
        if patient_info.get('medications'):
            report.append(f"Current Medications: {patient_info['medications']}")
        
        if patient_info.get('family_history'):
            report.append(f"Family History: {patient_info['family_history']}")
        
        report.append("")
        
        # Symptoms Summary
        report.append("SYMPTOMS REPORTED")
        report.append("-" * 17)
        symptoms = report_data.get('symptoms', {})
        if symptoms.get('raw_text'):
            report.append(f"Description: {symptoms['raw_text']}")
        else:
            report.append("No symptoms description provided")
        
        # Add processed symptom information if available
        processed_symptoms = report_data.get('processed_symptoms', {})
        if processed_symptoms:
            severity = processed_symptoms.get('severity_analysis', {})
            if severity:
                report.append(f"Severity Level: {severity.get('severity_score', 0)}/5")
                report.append(f"Urgency Level: {severity.get('urgency_level', 'Unknown')}")
        
        report.append("")
        
        # Analysis Results
        report.append("ANALYSIS RESULTS")
        report.append("-" * 16)
        
        # Overall Risk Assessment
        risk_assessment = report_data.get('risk_assessment', {})
        if risk_assessment:
            report.append(f"Overall Risk Level: {risk_assessment.get('level', 'Unknown')}")
            report.append(f"Risk Score: {risk_assessment.get('score', 0):.2f}")
            
            primary_concerns = risk_assessment.get('primary_concerns', [])
            if primary_concerns:
                report.append(f"Primary Concerns: {', '.join(primary_concerns)}")
            
            report.append(f"General Recommendation: {risk_assessment.get('recommendation', 'Consult healthcare provider')}")
        
        report.append("")
        
        # Detailed Predictions
        predictions = report_data.get('predictions', [])
        if predictions:
            report.append("DETAILED PREDICTIONS")
            report.append("-" * 19)
            
            for i, pred in enumerate(predictions[:5], 1):  # Top 5 predictions
                report.append(f"{i}. {pred.get('disease', 'Unknown')}")
                report.append(f"   Probability: {pred.get('probability', 0):.1%}")
                report.append(f"   Risk Level: {pred.get('risk_level', 'Unknown')}")
                
                evidence = pred.get('evidence', [])
                if evidence:
                    report.append("   Evidence:")
                    for ev in evidence[:3]:  # Top 3 evidence items
                        report.append(f"   - {ev}")
                report.append("")
        
        # AI Explanation
        explanations = report_data.get('explanations', {})
        if explanations.get('main_explanation'):
            report.append("AI EXPLANATION")
            report.append("-" * 14)
            report.append(explanations['main_explanation'])
            report.append("")
        
        # Risk Factors
        if explanations.get('risk_factors'):
            report.append("IDENTIFIED RISK FACTORS")
            report.append("-" * 23)
            report.append(explanations['risk_factors'])
            report.append("")
        
        # Recommendations
        recommendations = report_data.get('recommendations', [])
        if recommendations:
            report.append("RECOMMENDATIONS")
            report.append("-" * 15)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Next Steps
        next_steps = explanations.get('next_steps', [])
        if next_steps:
            report.append("NEXT STEPS")
            report.append("-" * 10)
            for i, step in enumerate(next_steps, 1):
                report.append(f"{i}. {step}")
            report.append("")
        
        # Important Disclaimer
        report.append("IMPORTANT DISCLAIMER")
        report.append("-" * 19)
        report.append("This AI analysis is designed to assist with health screening and should")
        report.append("not be considered a medical diagnosis. The results are based on the")
        report.append("information provided and AI algorithms, which may not capture all")
        report.append("relevant medical factors. Always consult with qualified healthcare")
        report.append("professionals for proper medical evaluation, diagnosis, and treatment.")
        report.append("")
        report.append("In case of emergency symptoms such as severe chest pain, difficulty")
        report.append("breathing, or loss of consciousness, seek immediate emergency medical care.")
        report.append("")
        
        # Technical Information
        report.append("TECHNICAL INFORMATION")
        report.append("-" * 20)
        report.append("Analysis performed by AI Disease Detection System")
        report.append("Powered by multiple specialized AI agents")
        report.append(f"Report ID: {int(time.time())}")
        
        # Data quality information
        data_quality = report_data.get('analysis_metadata', {}).get('data_quality', 0)
        report.append(f"Data Quality Score: {data_quality:.2f}")
        
        report.append("")
        report.append("="*60)
        
        return "\n".join(report)
    
    def _generate_summary_report(self, report_data: Dict[str, Any]) -> str:
        """Generate a concise summary report."""
        self.log_processing_step("Generating summary report")
        
        summary = []
        
        # Header
        summary.append("AI HEALTH ASSESSMENT SUMMARY")
        summary.append("=" * 30)
        summary.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        summary.append("")
        
        # Key Findings
        risk_assessment = report_data.get('risk_assessment', {})
        summary.append(f"Overall Risk: {risk_assessment.get('level', 'Unknown')}")
        
        predictions = report_data.get('predictions', [])
        if predictions:
            top_prediction = predictions[0]
            summary.append(f"Primary Concern: {top_prediction.get('disease', 'None')} ({top_prediction.get('probability', 0):.1%})")
        
        # Quick recommendations
        summary.append(f"Recommendation: {risk_assessment.get('recommendation', 'Consult healthcare provider')}")
        
        summary.append("")
        summary.append("This is a screening tool, not a medical diagnosis.")
        summary.append("Consult a healthcare professional for proper evaluation.")
        
        return "\n".join(summary)
    
    def _generate_report_metadata(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata about the report."""
        predictions = report_data.get('predictions', [])
        
        metadata = {
            'report_id': f"health_report_{int(time.time())}",
            'generation_timestamp': datetime.now().isoformat(),
            'patient_age': report_data.get('patient_info', {}).get('age'),
            'number_of_predictions': len(predictions),
            'highest_risk_prediction': predictions[0].get('disease') if predictions else None,
            'overall_risk_level': report_data.get('risk_assessment', {}).get('level'),
            'data_quality_score': report_data.get('analysis_metadata', {}).get('data_quality', 0),
            'report_sections': [
                'Patient Information',
                'Symptoms',
                'Analysis Results',
                'Predictions',
                'Explanations',
                'Recommendations',
                'Next Steps',
                'Disclaimer'
            ]
        }
        
        return metadata
