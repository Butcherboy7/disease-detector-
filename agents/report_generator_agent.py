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
            
            # Generate clinical summary for doctors
            clinical_summary = self._generate_clinical_summary(report_data)
            
            return self.create_success_response({
                'report_content': text_report,
                'summary_report': summary_report,
                'clinical_summary': clinical_summary,
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
    
    def _generate_clinical_summary(self, report_data: Dict[str, Any]) -> str:
        """Generate a clinical summary for healthcare providers."""
        self.log_processing_step("Generating clinical summary for doctors")
        
        clinical_summary = []
        
        # Header
        clinical_summary.append("CLINICAL SUMMARY FOR HEALTHCARE PROVIDER")
        clinical_summary.append("=" * 45)
        clinical_summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        clinical_summary.append("Source: AI-Powered Disease Detection System")
        clinical_summary.append("")
        
        # Patient Demographics
        patient_info = report_data.get('patient_info', {})
        clinical_summary.append("PATIENT DEMOGRAPHICS")
        clinical_summary.append("-" * 19)
        clinical_summary.append(f"Age: {patient_info.get('age', 'Not specified')} years")
        clinical_summary.append(f"Gender: {patient_info.get('gender', 'Not specified')}")
        
        existing_conditions = patient_info.get('existing_conditions', [])
        if existing_conditions and existing_conditions != ['None']:
            clinical_summary.append(f"Known Conditions: {', '.join(existing_conditions)}")
        else:
            clinical_summary.append("Known Conditions: None reported")
        
        if patient_info.get('medications'):
            clinical_summary.append(f"Current Medications: {patient_info['medications']}")
        else:
            clinical_summary.append("Current Medications: None reported")
        
        if patient_info.get('family_history'):
            clinical_summary.append(f"Family History: {patient_info['family_history']}")
        
        clinical_summary.append("")
        
        # Chief Complaint & Symptoms
        clinical_summary.append("CHIEF COMPLAINT & SYMPTOMS")
        clinical_summary.append("-" * 27)
        symptoms = report_data.get('symptoms', {})
        if symptoms.get('raw_text'):
            clinical_summary.append(f"Patient Description: {symptoms['raw_text']}")
        else:
            clinical_summary.append("Patient Description: No symptoms description provided")
        
        # Add symptom cluster analysis
        symptom_cluster_data = report_data.get('symptom_cluster_analysis', {})
        if symptom_cluster_data and 'cluster_analysis' in symptom_cluster_data:
            cluster_data = symptom_cluster_data['cluster_analysis']
            top_clusters = sorted(cluster_data.items(), key=lambda x: x[1], reverse=True)[:3]
            
            clinical_summary.append("")
            clinical_summary.append("Symptom Pattern Analysis:")
            for cluster_name, confidence in top_clusters:
                if confidence > 30:
                    display_name = cluster_name.replace('_', ' ').title()
                    clinical_summary.append(f"• {display_name}: {confidence:.0f}% pattern match")
        
        clinical_summary.append("")
        
        # Clinical Assessment & Risk Stratification
        clinical_summary.append("CLINICAL ASSESSMENT & RISK STRATIFICATION")
        clinical_summary.append("-" * 39)
        
        risk_assessment = report_data.get('risk_assessment', {})
        clinical_summary.append(f"Overall Risk Level: {risk_assessment.get('level', 'Unknown')}")
        clinical_summary.append(f"Risk Score: {risk_assessment.get('score', 0):.3f}")
        
        primary_concerns = risk_assessment.get('primary_concerns', [])
        if primary_concerns:
            clinical_summary.append(f"Primary Clinical Concerns: {', '.join(primary_concerns)}")
        
        clinical_summary.append("")
        
        # Top Differential Diagnoses
        predictions = report_data.get('predictions', [])
        if predictions:
            clinical_summary.append("TOP DIFFERENTIAL DIAGNOSES")
            clinical_summary.append("-" * 27)
            
            # Group by clinical relevance
            high_prob = [p for p in predictions if p.get('probability', 0) > 0.5]
            moderate_prob = [p for p in predictions if 0.2 <= p.get('probability', 0) <= 0.5]
            common_conditions = [p for p in predictions if any(keyword in p.get('disease', '').lower() 
                                                              for keyword in ['anemia', 'thyroid', 'vitamin d', 'autonomic'])]
            
            if high_prob:
                clinical_summary.append("High Probability (>50%):")
                for pred in high_prob[:3]:
                    disease = pred.get('disease', 'Unknown')
                    prob = pred.get('probability', 0)
                    severity = pred.get('severity_level', 1)
                    urgency = pred.get('urgency_level', 'low')
                    clinical_summary.append(f"• {disease}: {prob:.1%} (Severity: {severity}, Urgency: {urgency})")
                clinical_summary.append("")
            
            if moderate_prob:
                clinical_summary.append("Moderate Probability (20-50%):")
                for pred in moderate_prob[:3]:
                    disease = pred.get('disease', 'Unknown')
                    prob = pred.get('probability', 0)
                    clinical_summary.append(f"• {disease}: {prob:.1%}")
                clinical_summary.append("")
            
            if common_conditions:
                clinical_summary.append("Common Conditions to Consider:")
                for pred in common_conditions[:4]:
                    disease = pred.get('disease', 'Unknown')
                    prob = pred.get('probability', 0)
                    evidence = pred.get('evidence', [])
                    clinical_summary.append(f"• {disease}: {prob:.1%}")
                    if evidence:
                        key_evidence = [e for e in evidence[:2] if not e.startswith('Common symptom')]
                        if key_evidence:
                            clinical_summary.append(f"  Clinical indicators: {', '.join(key_evidence)}")
                clinical_summary.append("")
        
        # Lab Results (if available)
        lab_analysis = report_data.get('lab_analysis', {})
        if lab_analysis and lab_analysis.get('extracted_values'):
            clinical_summary.append("LABORATORY DATA")
            clinical_summary.append("-" * 15)
            
            extracted_values = lab_analysis['extracted_values']
            for category, values in extracted_values.items():
                if values:
                    category_display = category.replace('_', ' ').title()
                    clinical_summary.append(f"{category_display}:")
                    for test, value in values.items():
                        clinical_summary.append(f"• {test.replace('_', ' ').title()}: {value}")
            
            # Show abnormal values if available
            abnormal_analysis = lab_analysis.get('abnormal_analysis', {})
            if abnormal_analysis and abnormal_analysis.get('abnormal_values'):
                clinical_summary.append("")
                clinical_summary.append("Notable Abnormal Values:")
                for category, abnormals in abnormal_analysis['abnormal_values'].items():
                    for test, details in abnormals.items():
                        status = details['status']
                        severity = details['severity']
                        normal_range = details['normal_range']
                        clinical_summary.append(f"• {test.replace('_', ' ').title()}: {details['value']} ({status}, {severity}) [Normal: {normal_range}]")
            
            clinical_summary.append("")
        
        # Clinical Recommendations
        clinical_summary.append("CLINICAL RECOMMENDATIONS")
        clinical_summary.append("-" * 24)
        
        # Generate specific recommendations based on findings
        recommendations = self._generate_clinical_recommendations(report_data)
        for rec in recommendations:
            clinical_summary.append(f"• {rec}")
        
        clinical_summary.append("")
        
        # Follow-up and Monitoring
        clinical_summary.append("SUGGESTED FOLLOW-UP & MONITORING")
        clinical_summary.append("-" * 33)
        
        # Generate follow-up based on risk level and conditions
        followup_suggestions = self._generate_followup_suggestions(report_data)
        for suggestion in followup_suggestions:
            clinical_summary.append(f"• {suggestion}")
        
        clinical_summary.append("")
        
        # System Disclaimer
        clinical_summary.append("SYSTEM LIMITATIONS & DISCLAIMER")
        clinical_summary.append("-" * 32)
        clinical_summary.append("• This is an AI-generated screening assessment, not a diagnostic tool")
        clinical_summary.append("• Clinical correlation and professional judgment are essential")
        clinical_summary.append("• System accuracy varies by condition and symptom presentation")
        clinical_summary.append("• Designed to support, not replace, clinical decision-making")
        clinical_summary.append("• For research/educational purposes - verify all findings independently")
        
        clinical_summary.append("")
        clinical_summary.append("=" * 45)
        
        return "\n".join(clinical_summary)
    
    def _generate_clinical_recommendations(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate specific clinical recommendations based on assessment."""
        recommendations = []
        
        risk_level = report_data.get('risk_assessment', {}).get('level', 'Low')
        predictions = report_data.get('predictions', [])
        
        # Risk-based recommendations
        if risk_level == 'High':
            recommendations.append("Consider urgent evaluation within 24-48 hours")
            recommendations.append("Obtain relevant laboratory studies and imaging as indicated")
        elif risk_level == 'Medium':
            recommendations.append("Schedule follow-up within 1-2 weeks")
            recommendations.append("Consider targeted diagnostic workup based on differential")
        else:
            recommendations.append("Routine follow-up as clinically indicated")
        
        # Condition-specific recommendations
        top_predictions = [p for p in predictions if p.get('probability', 0) > 0.3]
        
        for pred in top_predictions[:3]:
            disease = pred.get('disease', '')
            
            if 'anemia' in disease.lower() or 'iron' in disease.lower():
                recommendations.append("CBC with differential, iron studies (ferritin, TIBC, transferrin saturation)")
                recommendations.append("Consider reticulocyte count and peripheral smear")
            
            elif 'thyroid' in disease.lower():
                recommendations.append("TSH, Free T4, consider TPO antibodies")
                recommendations.append("Clinical thyroid examination")
            
            elif 'vitamin d' in disease.lower():
                recommendations.append("25-hydroxyvitamin D level")
                recommendations.append("Consider PTH, calcium, phosphorus if deficient")
            
            elif 'autonomic' in disease.lower() or 'pots' in disease.lower():
                recommendations.append("Orthostatic vital signs (lying, sitting, standing)")
                recommendations.append("Consider cardiology referral for tilt table test")
                recommendations.append("ECG and echocardiogram if indicated")
            
            elif 'diabetes' in disease.lower():
                recommendations.append("Fasting glucose, HbA1c, consider glucose tolerance test")
                recommendations.append("Lipid panel, microalbumin screening")
            
            elif 'heart' in disease.lower():
                recommendations.append("ECG, chest X-ray, consider echocardiogram")
                recommendations.append("Lipid panel, cardiac enzymes if acute presentation")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_followup_suggestions(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate follow-up and monitoring suggestions."""
        suggestions = []
        
        risk_level = report_data.get('risk_assessment', {}).get('level', 'Low')
        predictions = report_data.get('predictions', [])
        
        # General monitoring
        suggestions.append("Symptom diary for 2-4 weeks to track progression")
        suggestions.append("Return precautions: worsening symptoms, new concerning features")
        
        # Risk-based monitoring
        if risk_level == 'High':
            suggestions.append("Close monitoring with serial assessments")
            suggestions.append("Patient education on warning signs")
        elif risk_level == 'Medium':
            suggestions.append("Structured follow-up in 1-2 weeks")
            suggestions.append("Telemedicine check-in if symptoms persist")
        
        # Condition-specific monitoring
        common_conditions = [p for p in predictions if any(keyword in p.get('disease', '').lower() 
                                                          for keyword in ['anemia', 'thyroid', 'vitamin d', 'autonomic'])]
        
        if common_conditions:
            suggestions.append("Consider baseline metabolic panel if multiple deficiencies suspected")
            suggestions.append("Nutritional assessment and counseling if indicated")
        
        # Lifestyle recommendations
        suggestions.append("Sleep hygiene assessment and optimization")
        suggestions.append("Stress management and coping strategies")
        suggestions.append("Activity modification based on symptom tolerance")
        
        return suggestions
