"""
Explainability Agent
Responsible for generating clear, understandable explanations of AI predictions
using OpenAI GPT models for medical insights.
"""

import json
from typing import Dict, Any, List
from .base_agent import BaseAgent
from utils.api_clients import OpenAIClient

class ExplainabilityAgent(BaseAgent):
    """
    Agent responsible for explaining AI predictions in simple, understandable terms.
    Uses OpenAI GPT models to generate medical explanations and recommendations.
    """
    
    def __init__(self):
        super().__init__("ExplainabilityAgent")
        self.openai_client = OpenAIClient()
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanations for disease predictions.
        
        Args:
            data: Data including predictions from PredictionAgent
            
        Returns:
            Detailed explanations and recommendations
        """
        try:
            self.log_processing_step("Starting explanation generation")
            
            # Extract prediction data
            if 'prediction_result' in data:
                predictions = data['prediction_result']['predictions']
                overall_risk = data['prediction_result']['overall_risk']
            else:
                predictions = data.get('predictions', [])
                overall_risk = data.get('overall_risk', {})
            
            # Get original symptoms and patient data for context
            symptoms_data = self._extract_symptoms_context(data)
            patient_context = self._extract_patient_context(data)
            
            # Generate comprehensive explanation
            main_explanation = self._generate_main_explanation(
                predictions, overall_risk, symptoms_data, patient_context
            )
            
            # Generate specific recommendations
            recommendations = self._generate_recommendations(
                predictions, overall_risk, patient_context
            )
            
            # Generate risk factor explanation
            risk_factors_explanation = self._explain_risk_factors(
                predictions, patient_context
            )
            
            # Generate next steps guidance
            next_steps = self._generate_next_steps(overall_risk, predictions)
            
            # Generate disclaimer and limitations
            disclaimer = self._generate_disclaimer()
            
            self.log_processing_step("Explanation generation completed successfully")
            
            return self.create_success_response({
                'explanation': main_explanation,
                'recommendations': recommendations,
                'risk_factors_explanation': risk_factors_explanation,
                'next_steps': next_steps,
                'disclaimer': disclaimer,
                'explanation_metadata': {
                    'predictions_analyzed': len(predictions),
                    'overall_risk_level': overall_risk.get('level', 'Unknown'),
                    'explanation_length': len(main_explanation.split()) if main_explanation else 0
                }
            })
            
        except Exception as e:
            return self.handle_error(e, "explanation generation")
    
    def _extract_symptoms_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract symptoms context from the data."""
        # Try to get from different possible locations in data structure
        if 'preprocessing_result' in data:
            return data['preprocessing_result']['preprocessed_data'].get('processed_symptoms', {})
        elif 'collection_result' in data:
            return data['collection_result']['organized_data'].get('symptoms', {})
        else:
            return data.get('symptoms', {})
    
    def _extract_patient_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patient context from the data."""
        if 'preprocessing_result' in data:
            return data['preprocessing_result']['preprocessed_data'].get('normalized_patient_data', {})
        elif 'collection_result' in data:
            return data['collection_result']['organized_data'].get('patient_info', {})
        else:
            return data.get('patient_info', {})
    
    def _generate_main_explanation(self, predictions: List[Dict[str, Any]], 
                                 overall_risk: Dict[str, Any], 
                                 symptoms_data: Dict[str, Any], 
                                 patient_context: Dict[str, Any]) -> str:
        """Generate the main explanation using OpenAI."""
        self.log_processing_step("Generating main explanation with OpenAI")
        
        try:
            # Prepare context for OpenAI
            prompt = self._create_explanation_prompt(predictions, overall_risk, symptoms_data, patient_context)
            
            # Get explanation from OpenAI
            explanation = self.openai_client.generate_medical_explanation(prompt)
            
            return explanation
            
        except Exception as e:
            self.logger.warning(f"OpenAI explanation failed, using fallback: {str(e)}")
            return self._generate_fallback_explanation(predictions, overall_risk)
    
    def _create_explanation_prompt(self, predictions: List[Dict[str, Any]], 
                                 overall_risk: Dict[str, Any], 
                                 symptoms_data: Dict[str, Any], 
                                 patient_context: Dict[str, Any]) -> str:
        """Create a detailed prompt for OpenAI explanation."""
        
        # Extract symptoms text
        symptoms_text = symptoms_data.get('cleaned_text', '') or symptoms_data.get('original_text', '')
        
        # Format predictions
        pred_text = "\n".join([
            f"- {pred['disease']}: {pred['probability']:.1%} probability ({pred['risk_level']} risk)"
            for pred in predictions[:5]  # Top 5 predictions
        ])
        
        # Format patient info
        age = patient_context.get('age', 'Unknown')
        gender = patient_context.get('gender', 'Unknown')
        conditions = ', '.join(patient_context.get('existing_conditions', []))
        
        prompt = f"""
As a medical AI assistant, please provide a clear, compassionate explanation of the following health analysis results. 
Use simple language that a patient can understand, avoid medical jargon, and be reassuring while being accurate.

PATIENT CONTEXT:
- Age: {age}
- Gender: {gender}
- Existing conditions: {conditions if conditions else 'None reported'}

SYMPTOMS DESCRIBED:
"{symptoms_text}"

AI PREDICTIONS:
{pred_text}

OVERALL RISK ASSESSMENT: {overall_risk.get('level', 'Unknown')} risk level

Please provide:
1. A clear explanation of what these results mean
2. How the symptoms relate to the potential conditions
3. The significance of the risk levels
4. Reassurance about the AI's role as a screening tool, not a diagnosis

Keep the explanation under 300 words and maintain a professional, caring tone.
"""
        
        return prompt
    
    def _generate_recommendations(self, predictions: List[Dict[str, Any]], 
                                overall_risk: Dict[str, Any], 
                                patient_context: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on predictions."""
        self.log_processing_step("Generating recommendations")
        
        recommendations = []
        
        # Based on overall risk level
        risk_level = overall_risk.get('level', 'Low')
        
        if risk_level == 'High':
            recommendations.append("Schedule an urgent appointment with your healthcare provider within 24-48 hours")
            recommendations.append("Do not delay seeking medical attention for your symptoms")
        elif risk_level == 'Medium':
            recommendations.append("Schedule an appointment with your healthcare provider within the next week")
            recommendations.append("Monitor your symptoms and seek immediate care if they worsen")
        else:
            recommendations.append("Consider scheduling a routine check-up with your healthcare provider")
            recommendations.append("Continue monitoring your symptoms")
        
        # Specific recommendations based on top predictions
        top_predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)[:3]
        
        for pred in top_predictions:
            if pred['probability'] > 0.5:
                disease = pred['disease']
                
                if 'Diabetes' in disease:
                    recommendations.extend([
                        "Consider getting blood sugar levels tested",
                        "Monitor your diet and reduce sugar intake",
                        "Increase physical activity if possible"
                    ])
                elif 'Heart' in disease:
                    recommendations.extend([
                        "Consider getting an ECG or cardiovascular screening",
                        "Monitor blood pressure regularly",
                        "Avoid strenuous physical activity until cleared by a doctor"
                    ])
                elif 'Stress' in disease or 'Anxiety' in disease:
                    recommendations.extend([
                        "Practice stress management techniques like deep breathing or meditation",
                        "Ensure adequate sleep (7-9 hours per night)",
                        "Consider speaking with a mental health professional"
                    ])
                elif 'Respiratory' in disease:
                    recommendations.extend([
                        "Avoid exposure to smoke, dust, or allergens",
                        "Consider pulmonary function tests",
                        "Use prescribed inhalers as directed if you have them"
                    ])
                elif 'Anemia' in disease or 'Iron' in disease:
                    recommendations.extend([
                        "Request complete blood count (CBC) and iron studies tests",
                        "Increase iron-rich foods in your diet (leafy greens, lean meats)",
                        "Consider vitamin C to enhance iron absorption",
                        "Avoid drinking tea or coffee with iron-rich meals"
                    ])
                elif 'Thyroid' in disease or 'Hypothyroid' in disease:
                    recommendations.extend([
                        "Request thyroid function tests (TSH, Free T4)",
                        "Monitor your weight, energy levels, and cold tolerance",
                        "Consider reducing raw cruciferous vegetables if consuming large amounts",
                        "Ensure adequate iodine and selenium in your diet"
                    ])
                elif 'Vitamin D' in disease:
                    recommendations.extend([
                        "Request 25-hydroxyvitamin D blood test",
                        "Increase safe sun exposure (10-15 minutes daily)",
                        "Consider vitamin D3 supplements after testing",
                        "Include vitamin D-rich foods (fatty fish, fortified milk)"
                    ])
                elif 'Autonomic' in disease or 'POTS' in disease:
                    recommendations.extend([
                        "Schedule consultation with cardiologist or autonomic specialist",
                        "Consider tilt table test for POTS diagnosis",
                        "Increase salt and fluid intake if medically appropriate",
                        "Wear compression stockings to improve blood flow"
                    ])
        
        # General health recommendations
        recommendations.extend([
            "Keep a symptom diary to track changes over time",
            "Maintain a healthy diet and regular exercise routine",
            "Ensure you're getting adequate sleep and managing stress"
        ])
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations[:8]  # Limit to 8 recommendations
    
    def _explain_risk_factors(self, predictions: List[Dict[str, Any]], 
                            patient_context: Dict[str, Any]) -> str:
        """Explain identified risk factors."""
        self.log_processing_step("Explaining risk factors")
        
        risk_factors = []
        
        # Age-related factors
        age = patient_context.get('age', 0)
        if age > 65:
            risk_factors.append("advanced age (over 65)")
        elif age > 45:
            risk_factors.append("middle age (over 45)")
        
        # Existing conditions
        conditions = patient_context.get('existing_conditions', [])
        for condition in conditions:
            if condition != 'None':
                risk_factors.append(f"existing {condition.lower()}")
        
        # Family history
        family_risks = patient_context.get('family_risk_factors', [])
        if family_risks:
            risk_factors.append(f"family history of {', '.join(family_risks)}")
        
        # Extract evidence from predictions
        evidence_factors = set()
        for pred in predictions:
            for evidence in pred.get('evidence', []):
                if 'Symptom:' in evidence:
                    symptom = evidence.replace('Symptom:', '').strip()
                    evidence_factors.add(f"reported {symptom}")
        
        risk_factors.extend(list(evidence_factors)[:3])  # Add top 3 symptom factors
        
        if not risk_factors:
            return "No significant risk factors were identified based on the provided information."
        
        explanation = f"The following risk factors were identified: {', '.join(risk_factors[:5])}. "
        explanation += "These factors contribute to the overall risk assessment but do not constitute a medical diagnosis. "
        explanation += "Many people with risk factors never develop the associated conditions, while others without obvious risk factors may still be affected."
        
        return explanation
    
    def _generate_next_steps(self, overall_risk: Dict[str, Any], 
                           predictions: List[Dict[str, Any]]) -> List[str]:
        """Generate specific next steps based on risk level."""
        next_steps = []
        
        risk_level = overall_risk.get('level', 'Low')
        
        if risk_level == 'High':
            next_steps = [
                "Contact your healthcare provider immediately",
                "Prepare a list of all your symptoms with dates and severity",
                "Gather any recent medical records or test results",
                "If symptoms are severe or worsening, consider emergency care"
            ]
        elif risk_level == 'Medium':
            next_steps = [
                "Schedule an appointment with your primary care physician",
                "Document your symptoms in detail before the appointment",
                "Prepare questions about the potential conditions identified",
                "Consider any relevant family medical history to discuss"
            ]
        else:
            next_steps = [
                "Continue monitoring your symptoms",
                "Schedule a routine health check-up if due",
                "Maintain healthy lifestyle habits",
                "Seek medical advice if symptoms persist or worsen"
            ]
        
        # Add general steps
        next_steps.extend([
            "Keep this analysis to show your healthcare provider",
            "Remember this is a screening tool, not a medical diagnosis"
        ])
        
        return next_steps
    
    def _generate_disclaimer(self) -> str:
        """Generate important disclaimer and limitations."""
        return """
IMPORTANT DISCLAIMER: This AI analysis is designed to assist with health screening and should not be considered a medical diagnosis. 
The results are based on the information you provided and AI algorithms, which may not capture all relevant medical factors. 
Always consult with qualified healthcare professionals for proper medical evaluation, diagnosis, and treatment. 
In case of emergency symptoms such as severe chest pain, difficulty breathing, or loss of consciousness, seek immediate emergency medical care. 
This tool is meant to supplement, not replace, professional medical judgment.
"""
    
    def _generate_fallback_explanation(self, predictions: List[Dict[str, Any]], 
                                     overall_risk: Dict[str, Any]) -> str:
        """Generate a basic explanation when OpenAI is unavailable."""
        if not predictions:
            return "Based on the analysis, no significant health concerns were identified. However, it's always good to consult with a healthcare provider for regular check-ups."
        
        top_prediction = predictions[0]
        risk_level = overall_risk.get('level', 'Low')
        
        explanation = f"Based on your symptoms and information provided, the AI analysis suggests a {risk_level.lower()} risk level. "
        explanation += f"The most likely condition identified is {top_prediction['disease']} with a {top_prediction['probability']:.1%} probability. "
        
        if risk_level == 'High':
            explanation += "This indicates you should seek medical attention promptly to discuss your symptoms with a healthcare professional."
        elif risk_level == 'Medium':
            explanation += "This suggests you should schedule an appointment with your healthcare provider to discuss your symptoms."
        else:
            explanation += "While the risk appears low, continue monitoring your symptoms and consult a healthcare provider if they persist or worsen."
        
        explanation += " Remember, this is a screening tool and not a substitute for professional medical diagnosis."
        
        return explanation
