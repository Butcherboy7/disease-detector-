"""
Prediction Agent
Responsible for making disease predictions using machine learning models
and external APIs like HuggingFace.
"""

import json
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent
from utils.disease_models import DiseaseModels
from utils.api_clients import HuggingFaceClient
from utils.symptom_clustering import SymptomClusterAnalyzer
from utils.lab_report_analyzer import LabReportAnalyzer

class PredictionAgent(BaseAgent):
    """
    Agent responsible for predicting disease likelihood using ML models.
    Integrates with HuggingFace models and local prediction algorithms.
    """
    
    def __init__(self):
        super().__init__("PredictionAgent")
        self.disease_models = DiseaseModels()
        self.hf_client = HuggingFaceClient()
        self.symptom_analyzer = SymptomClusterAnalyzer()
        self.lab_analyzer = LabReportAnalyzer()
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make disease predictions based on preprocessed data.
        
        Args:
            data: Preprocessed data from PreprocessingAgent
            
        Returns:
            Disease predictions with confidence scores
        """
        try:
            self.log_processing_step("Starting disease prediction")
            
            # Extract preprocessed data
            if 'preprocessing_result' in data:
                preprocessed_data = data['preprocessing_result']['preprocessed_data']
            else:
                preprocessed_data = data
            
            # Extract features for prediction
            feature_vector = self._extract_feature_vector(preprocessed_data)
            
            # Perform symptom clustering analysis
            symptom_cluster_analysis = self._perform_symptom_clustering(preprocessed_data)
            
            # Analyze lab reports if available
            lab_analysis = self._analyze_lab_reports(preprocessed_data)
            
            # Make predictions for different diseases
            predictions = []
            
            # Diabetes prediction
            diabetes_pred = self._predict_diabetes(feature_vector, preprocessed_data)
            predictions.append(diabetes_pred)
            
            # Heart disease prediction
            heart_pred = self._predict_heart_disease(feature_vector, preprocessed_data)
            predictions.append(heart_pred)
            
            # Stress/anxiety prediction
            stress_pred = self._predict_stress(feature_vector, preprocessed_data)
            predictions.append(stress_pred)
            
            # Respiratory condition prediction
            respiratory_pred = self._predict_respiratory_conditions(feature_vector, preprocessed_data)
            predictions.append(respiratory_pred)
            
            # Use HuggingFace models for additional predictions if available
            hf_predictions = self._get_huggingface_predictions(preprocessed_data)
            if hf_predictions:
                predictions.extend(hf_predictions)
            
            # Sort predictions by probability
            predictions.sort(key=lambda x: x['probability'], reverse=True)
            
            # Apply lab-based risk adjustments
            adjusted_predictions = self._apply_lab_adjustments(predictions, lab_analysis)
            
            # Calculate overall risk assessment
            overall_risk = self._calculate_overall_risk_enhanced(adjusted_predictions, preprocessed_data, lab_analysis or {})
            
            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(symptom_cluster_analysis, adjusted_predictions)
            
            self.log_processing_step("Disease prediction completed successfully")
            
            return self.create_success_response({
                'predictions': adjusted_predictions,
                'overall_risk': overall_risk,
                'symptom_cluster_analysis': symptom_cluster_analysis,
                'lab_analysis': lab_analysis,
                'follow_up_questions': follow_up_questions,
                'feature_vector': feature_vector.tolist() if isinstance(feature_vector, np.ndarray) else feature_vector,
                'prediction_metadata': {
                    'num_predictions': len(adjusted_predictions),
                    'highest_risk_disease': adjusted_predictions[0]['disease'] if adjusted_predictions else None,
                    'max_probability': adjusted_predictions[0]['probability'] if adjusted_predictions else 0,
                    'lab_adjusted': bool(lab_analysis.get('risk_adjustments')),
                    'clustering_confidence': symptom_cluster_analysis.get('confidence_summary', '')
                }
            })
            
        except Exception as e:
            return self.handle_error(e, "disease prediction")
    
    def _extract_feature_vector(self, preprocessed_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical feature vector for ML models."""
        self.log_processing_step("Extracting feature vector")
        
        features = []
        
        # Patient demographic features
        patient_data = preprocessed_data.get('normalized_patient_data', {})
        features.extend([
            patient_data.get('age', 0) / 100,  # Normalized age
            patient_data.get('gender_encoded', 0) / 3,  # Normalized gender
            patient_data.get('risk_score', 0),
            len(patient_data.get('existing_conditions', [])) / 5  # Normalized condition count
        ])
        
        # Symptom features
        symptoms = preprocessed_data.get('processed_symptoms', {})
        features.extend([
            symptoms.get('severity_analysis', {}).get('severity_score', 0) / 5,
            symptoms.get('severity_analysis', {}).get('urgency_score', 0) / 5,
            len(symptoms.get('entities', [])) / 10,  # Normalized entity count
            symptoms.get('temporal_info', {}).get('chronic_indicators', 0) / 3,
            symptoms.get('temporal_info', {}).get('acute_indicators', 0) / 3
        ])
        
        # Wearable data features
        wearable = preprocessed_data.get('processed_wearable', {})
        if wearable:
            features.extend([
                (wearable.get('heart_rate', 70) - 70) / 30,  # Normalized heart rate
                (wearable.get('spo2', 98) - 98) / 10,  # Normalized SpO2
                (wearable.get('sleep_hours', 8) - 8) / 4,  # Normalized sleep
                wearable.get('vitals_score', 1.0)
            ])
        else:
            features.extend([0, 0, 0, 1])  # Default values
        
        # Medical features
        medical_features = preprocessed_data.get('medical_features', {}).get('features', {})
        features.extend([
            medical_features.get('has_imaging', 0),
            medical_features.get('has_lab_results', 0),
            medical_features.get('abnormal_heart_rate', 0),
            medical_features.get('low_oxygen', 0),
            medical_features.get('sleep_disorder', 0)
        ])
        
        return np.array(features)
    
    def _predict_diabetes(self, feature_vector: np.ndarray, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict diabetes likelihood."""
        self.log_processing_step("Predicting diabetes")
        
        # Rule-based scoring for diabetes
        score = 0.0
        evidence = []
        
        patient_data = preprocessed_data.get('normalized_patient_data', {})
        symptoms = preprocessed_data.get('processed_symptoms', {})
        wearable = preprocessed_data.get('processed_wearable', {})
        
        # Age factor
        if patient_data.get('age', 0) > 45:
            score += 0.15
            evidence.append("Age over 45")
        
        # Existing diabetes
        if 'Diabetes' in patient_data.get('existing_conditions', []):
            score += 0.5
            evidence.append("Existing diabetes diagnosis")
        
        # Family history
        if 'diabetes' in patient_data.get('family_risk_factors', []):
            score += 0.1
            evidence.append("Family history of diabetes")
        
        # Symptom analysis
        symptom_text = symptoms.get('cleaned_text', '').lower()
        diabetes_symptoms = ['thirst', 'urination', 'hunger', 'fatigue', 'blurred vision']
        
        for symptom in diabetes_symptoms:
            if symptom in symptom_text:
                score += 0.05
                evidence.append(f"Symptom: {symptom}")
        
        # Medication check
        if 'diabetes' in patient_data.get('medication_categories', []):
            score += 0.3
            evidence.append("Taking diabetes medication")
        
        # Cap the score
        probability = min(score, 0.95)
        
        # Determine risk level
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'disease': 'Diabetes',
            'probability': probability,
            'risk_level': risk_level,
            'evidence': evidence,
            'model_type': 'rule_based'
        }
    
    def _predict_heart_disease(self, feature_vector: np.ndarray, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict heart disease likelihood."""
        self.log_processing_step("Predicting heart disease")
        
        score = 0.0
        evidence = []
        
        patient_data = preprocessed_data.get('normalized_patient_data', {})
        symptoms = preprocessed_data.get('processed_symptoms', {})
        wearable = preprocessed_data.get('processed_wearable', {})
        
        # Age and gender factors
        age = patient_data.get('age', 0)
        if age > 65:
            score += 0.2
            evidence.append("Age over 65")
        elif age > 45:
            score += 0.1
            evidence.append("Age over 45")
        
        if patient_data.get('gender_encoded') == 1:  # Male
            score += 0.05
            evidence.append("Male gender")
        
        # Existing conditions
        risk_conditions = ['Heart Disease', 'Hypertension', 'Diabetes']
        for condition in patient_data.get('existing_conditions', []):
            if condition in risk_conditions:
                score += 0.15
                evidence.append(f"Existing condition: {condition}")
        
        # Symptoms
        symptom_text = symptoms.get('cleaned_text', '').lower()
        heart_symptoms = ['chest pain', 'shortness of breath', 'palpitations', 'dizziness', 'fatigue']
        
        for symptom in heart_symptoms:
            if symptom in symptom_text:
                score += 0.08
                evidence.append(f"Symptom: {symptom}")
        
        # Wearable data
        if wearable:
            if wearable.get('heart_rate_category') in ['high', 'low']:
                score += 0.1
                evidence.append(f"Abnormal heart rate: {wearable.get('heart_rate_category')}")
            
            if wearable.get('spo2_category') == 'low':
                score += 0.1
                evidence.append("Low oxygen saturation")
        
        # Family history
        if 'heart disease' in patient_data.get('family_risk_factors', []):
            score += 0.1
            evidence.append("Family history of heart disease")
        
        probability = min(score, 0.95)
        
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'disease': 'Heart Disease',
            'probability': probability,
            'risk_level': risk_level,
            'evidence': evidence,
            'model_type': 'rule_based'
        }
    
    def _predict_stress(self, feature_vector: np.ndarray, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict stress/anxiety likelihood."""
        self.log_processing_step("Predicting stress/anxiety")
        
        score = 0.0
        evidence = []
        
        symptoms = preprocessed_data.get('processed_symptoms', {})
        wearable = preprocessed_data.get('processed_wearable', {})
        
        # Symptom analysis
        symptom_text = symptoms.get('cleaned_text', '').lower()
        stress_symptoms = ['anxiety', 'stress', 'nervous', 'worry', 'panic', 'tension', 'restless', 'insomnia']
        
        for symptom in stress_symptoms:
            if symptom in symptom_text:
                score += 0.1
                evidence.append(f"Symptom: {symptom}")
        
        # Physical symptoms of stress
        physical_stress_symptoms = ['headache', 'fatigue', 'muscle tension', 'sweating', 'palpitations']
        for symptom in physical_stress_symptoms:
            if symptom in symptom_text:
                score += 0.05
                evidence.append(f"Physical symptom: {symptom}")
        
        # Wearable indicators
        if wearable:
            if wearable.get('heart_rate_category') == 'high':
                score += 0.08
                evidence.append("Elevated heart rate")
            
            if wearable.get('sleep_category') == 'insufficient':
                score += 0.1
                evidence.append("Insufficient sleep")
        
        # Severity and urgency
        severity = symptoms.get('severity_analysis', {})
        if severity.get('urgency_level') in ['medium', 'high']:
            score += 0.05
            evidence.append("High symptom urgency")
        
        probability = min(score, 0.95)
        
        if probability > 0.6:
            risk_level = "High"
        elif probability > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'disease': 'Stress/Anxiety',
            'probability': probability,
            'risk_level': risk_level,
            'evidence': evidence,
            'model_type': 'rule_based'
        }
    
    def _predict_respiratory_conditions(self, feature_vector: np.ndarray, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict respiratory condition likelihood."""
        self.log_processing_step("Predicting respiratory conditions")
        
        score = 0.0
        evidence = []
        
        patient_data = preprocessed_data.get('normalized_patient_data', {})
        symptoms = preprocessed_data.get('processed_symptoms', {})
        wearable = preprocessed_data.get('processed_wearable', {})
        
        # Existing respiratory conditions
        if 'Asthma' in patient_data.get('existing_conditions', []):
            score += 0.3
            evidence.append("Existing asthma diagnosis")
        
        # Symptoms
        symptom_text = symptoms.get('cleaned_text', '').lower()
        respiratory_symptoms = ['cough', 'shortness of breath', 'wheezing', 'chest tightness', 'difficulty breathing']
        
        for symptom in respiratory_symptoms:
            if symptom in symptom_text:
                score += 0.1
                evidence.append(f"Symptom: {symptom}")
        
        # Wearable data
        if wearable and wearable.get('spo2_category') == 'low':
            score += 0.15
            evidence.append("Low oxygen saturation")
        
        # Medication check
        if 'respiratory' in patient_data.get('medication_categories', []):
            score += 0.2
            evidence.append("Taking respiratory medication")
        
        probability = min(score, 0.95)
        
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'disease': 'Respiratory Condition',
            'probability': probability,
            'risk_level': risk_level,
            'evidence': evidence,
            'model_type': 'rule_based'
        }
    
    def _get_huggingface_predictions(self, preprocessed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get predictions from HuggingFace models."""
        try:
            self.log_processing_step("Getting HuggingFace predictions")
            
            # Prepare text for analysis
            symptoms_text = preprocessed_data.get('processed_symptoms', {}).get('cleaned_text', '')
            
            if not symptoms_text:
                return []
            
            # Use HuggingFace client for medical text classification
            hf_results = self.hf_client.classify_medical_text(symptoms_text)
            
            predictions = []
            for result in hf_results:
                predictions.append({
                    'disease': result.get('label', 'Unknown'),
                    'probability': result.get('score', 0),
                    'risk_level': self._determine_risk_level(result.get('score', 0)),
                    'evidence': ['HuggingFace model prediction'],
                    'model_type': 'huggingface'
                })
            
            return predictions
            
        except Exception as e:
            self.logger.warning(f"HuggingFace prediction failed: {str(e)}")
            return []
    
    def _determine_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability."""
        if probability > 0.7:
            return "High"
        elif probability > 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_overall_risk(self, predictions: List[Dict[str, Any]], preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall health risk assessment."""
        if not predictions:
            return {
                'level': 'Low',
                'score': 0.0,
                'primary_concerns': [],
                'recommendation': 'Continue regular health monitoring'
            }
        
        # Calculate weighted risk score
        total_score = 0.0
        high_risk_diseases = []
        
        for pred in predictions:
            if pred['risk_level'] == 'High':
                total_score += pred['probability'] * 1.0
                high_risk_diseases.append(pred['disease'])
            elif pred['risk_level'] == 'Medium':
                total_score += pred['probability'] * 0.6
            else:
                total_score += pred['probability'] * 0.3
        
        # Normalize score
        overall_score = min(total_score / len(predictions), 1.0)
        
        # Determine overall risk level
        if overall_score > 0.7 or len(high_risk_diseases) >= 2:
            risk_level = 'High'
            recommendation = 'Seek immediate medical attention'
        elif overall_score > 0.4 or len(high_risk_diseases) >= 1:
            risk_level = 'Medium'
            recommendation = 'Schedule appointment with healthcare provider'
        else:
            risk_level = 'Low'
            recommendation = 'Continue regular health monitoring'
        
        return {
            'level': risk_level,
            'score': overall_score,
            'primary_concerns': high_risk_diseases,
            'recommendation': recommendation
        }
    
    def _perform_symptom_clustering(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform symptom clustering analysis."""
        try:
            self.log_processing_step("Performing symptom clustering analysis")
            
            # Extract symptoms data
            symptoms_data = preprocessed_data.get('processed_symptoms', {})
            symptoms_text = symptoms_data.get('raw_text', '')
            extracted_entities = preprocessed_data.get('preprocessing_metadata', {}).get('extracted_entities', [])
            
            if not symptoms_text:
                return {'cluster_analysis': {}, 'insights': [], 'confidence_summary': 'No symptoms provided'}
            
            # Perform clustering analysis
            cluster_analysis = self.symptom_analyzer.analyze_symptom_clusters(symptoms_text, extracted_entities)
            
            return cluster_analysis
            
        except Exception as e:
            self.logger.warning(f"Symptom clustering failed: {str(e)}")
            return {'cluster_analysis': {}, 'insights': [], 'confidence_summary': 'Analysis unavailable'}
    
    def _analyze_lab_reports(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze lab reports if available."""
        try:
            self.log_processing_step("Analyzing lab reports")
            
            # Extract processed files
            processed_files = preprocessed_data.get('processed_files', [])
            lab_analysis_results = {}
            
            for file_data in processed_files:
                if file_data.get('category') in ['lab_report', 'blood_test', 'medical_report']:
                    file_text = file_data.get('extracted_text', '')
                    if file_text:
                        lab_analysis = self.lab_analyzer.analyze_lab_report(file_text)
                        lab_analysis_results[file_data.get('filename', 'unknown')] = lab_analysis
            
            # Combine results from all lab reports
            if lab_analysis_results:
                combined_analysis = self._combine_lab_analyses(lab_analysis_results)
                return combined_analysis
            else:
                return {'extracted_values': {}, 'risk_adjustments': {}, 'lab_insights': []}
                
        except Exception as e:
            self.logger.warning(f"Lab report analysis failed: {str(e)}")
            return {'extracted_values': {}, 'risk_adjustments': {}, 'lab_insights': []}
    
    def _combine_lab_analyses(self, lab_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple lab analysis results."""
        combined_values = {}
        all_insights = []
        combined_risk_adjustments = {}
        
        for filename, analysis in lab_results.items():
            # Combine extracted values
            for category, values in analysis.get('extracted_values', {}).items():
                if category not in combined_values:
                    combined_values[category] = {}
                combined_values[category].update(values)
            
            # Combine insights
            insights = analysis.get('lab_insights', [])
            all_insights.extend([f"{filename}: {insight}" for insight in insights])
            
            # Combine risk adjustments (take maximum)
            risk_adjustments = analysis.get('risk_adjustments', {})
            for disease, adjustment in risk_adjustments.items():
                if disease not in combined_risk_adjustments:
                    combined_risk_adjustments[disease] = adjustment
                else:
                    combined_risk_adjustments[disease] = max(combined_risk_adjustments[disease], adjustment)
        
        return {
            'extracted_values': combined_values,
            'risk_adjustments': combined_risk_adjustments,
            'lab_insights': all_insights,
            'num_reports_analyzed': len(lab_results)
        }
    
    def _apply_lab_adjustments(self, predictions: List[Dict[str, Any]], lab_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply lab-based risk adjustments to predictions."""
        if not lab_analysis.get('risk_adjustments'):
            return predictions
        
        risk_adjustments = lab_analysis['risk_adjustments']
        adjusted_predictions = []
        
        for prediction in predictions:
            disease = prediction['disease'].lower()
            adjusted_prediction = prediction.copy()
            
            # Apply adjustment if available
            adjustment_factor = 1.0
            for disease_key, factor in risk_adjustments.items():
                if disease_key in disease or disease in disease_key:
                    adjustment_factor = factor
                    break
            
            # Adjust probability
            original_prob = prediction['probability']
            adjusted_prob = min(original_prob * adjustment_factor, 1.0)
            adjusted_prediction['probability'] = adjusted_prob
            adjusted_prediction['risk_level'] = self._determine_risk_level(adjusted_prob)
            
            # Add lab adjustment info
            if adjustment_factor != 1.0:
                adjusted_prediction['lab_adjusted'] = True
                adjusted_prediction['adjustment_factor'] = adjustment_factor
                adjusted_prediction['original_probability'] = original_prob
            
            adjusted_predictions.append(adjusted_prediction)
        
        # Re-sort by adjusted probability
        adjusted_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return adjusted_predictions
    
    def _generate_follow_up_questions(self, symptom_cluster_analysis: Dict[str, Any], predictions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate follow-up questions based on analysis."""
        try:
            # Get questions from symptom clustering
            cluster_questions = self.symptom_analyzer.get_follow_up_questions(symptom_cluster_analysis)
            
            # Add prediction-specific questions
            prediction_questions = []
            if predictions:
                top_prediction = predictions[0]
                if top_prediction['probability'] > 0.5:
                    disease = top_prediction['disease'].lower()
                    
                    if 'diabetes' in disease:
                        prediction_questions.append({
                            "question": "Do you have a family history of diabetes?",
                            "context": "Family history is a significant risk factor for diabetes",
                            "type": "yes_no"
                        })
                    elif 'heart' in disease:
                        prediction_questions.append({
                            "question": "Do you smoke or have you smoked in the past?",
                            "context": "Smoking is a major risk factor for heart disease",
                            "type": "yes_no"
                        })
            
            # Combine and limit questions
            all_questions = cluster_questions + prediction_questions
            return all_questions[:3]  # Return max 3 questions
            
        except Exception as e:
            self.logger.warning(f"Follow-up question generation failed: {str(e)}")
            return []
    
    def _calculate_overall_risk_enhanced(self, predictions: List[Dict[str, Any]], preprocessed_data: Dict[str, Any], lab_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall health risk assessment with lab adjustments."""
        if not predictions:
            return {
                'level': 'Low',
                'score': 0.0,
                'primary_concerns': [],
                'recommendation': 'Continue regular health monitoring'
            }
        
        # Calculate weighted risk score
        total_score = 0.0
        high_risk_diseases = []
        
        for pred in predictions:
            if pred['risk_level'] == 'High':
                total_score += pred['probability'] * 1.0
                high_risk_diseases.append(pred['disease'])
            elif pred['risk_level'] == 'Medium':
                total_score += pred['probability'] * 0.6
            else:
                total_score += pred['probability'] * 0.3
        
        # Apply lab-based risk multiplier
        lab_multiplier = 1.0
        if lab_analysis and lab_analysis.get('risk_adjustments'):
            max_adjustment = max(lab_analysis['risk_adjustments'].values())
            lab_multiplier = min(max_adjustment, 1.5)  # Cap at 1.5x
        
        # Normalize score
        overall_score = min((total_score / len(predictions)) * lab_multiplier, 1.0)
        
        # Determine overall risk level
        if overall_score > 0.7 or len(high_risk_diseases) >= 2:
            risk_level = 'High'
            recommendation = 'Seek immediate medical attention'
        elif overall_score > 0.4 or len(high_risk_diseases) >= 1:
            risk_level = 'Medium'
            recommendation = 'Schedule appointment with healthcare provider'
        else:
            risk_level = 'Low'
            recommendation = 'Continue regular health monitoring'
        
        result = {
            'level': risk_level,
            'score': overall_score,
            'primary_concerns': high_risk_diseases,
            'recommendation': recommendation
        }
        
        # Add lab adjustment info if applicable
        if lab_analysis and lab_analysis.get('risk_adjustments'):
            result['lab_adjusted'] = True
            result['lab_multiplier'] = lab_multiplier
        
        return result
