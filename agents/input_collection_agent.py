"""
Input Collection Agent
Responsible for collecting and organizing user inputs including symptoms,
medical history, uploaded documents, and wearable data.
"""

import json
import base64
from typing import Dict, Any, List
from .base_agent import BaseAgent

class InputCollectionAgent(BaseAgent):
    """
    Agent responsible for collecting and organizing all user inputs.
    Validates input data and prepares it for downstream processing.
    """
    
    def __init__(self):
        super().__init__("InputCollectionAgent")
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate user input data.
        
        Args:
            data: Raw input data from user interface
            
        Returns:
            Organized and validated input data
        """
        try:
            self.log_processing_step("Starting input collection")
            
            # Validate required fields
            required_fields = ['symptoms', 'patient_info']
            if not self.validate_input(data, required_fields):
                return self.handle_error(
                    ValueError("Missing required fields"), 
                    "input validation"
                )
            
            # Organize symptoms data
            symptoms_data = self._process_symptoms(data['symptoms'])
            
            # Organize patient information
            patient_data = self._process_patient_info(data['patient_info'])
            
            # Process uploaded files
            files_data = self._process_uploaded_files(data.get('uploaded_files', []))
            
            # Process wearable data
            wearable_data = self._process_wearable_data(data.get('wearable_data', {}))
            
            # Compile organized data
            organized_data = {
                'symptoms': symptoms_data,
                'patient_info': patient_data,
                'uploaded_files': files_data,
                'wearable_data': wearable_data,
                'collection_metadata': {
                    'total_files': len(files_data),
                    'has_wearable_data': bool(wearable_data),
                    'symptom_keywords': self._extract_symptom_keywords(data['symptoms'])
                }
            }
            
            self.log_processing_step("Input collection completed successfully")
            
            return self.create_success_response({
                'organized_data': organized_data,
                'data_quality_score': self._calculate_data_quality_score(organized_data)
            })
            
        except Exception as e:
            return self.handle_error(e, "input processing")
    
    def _process_symptoms(self, symptoms_text: str) -> Dict[str, Any]:
        """Process and structure symptoms text."""
        self.log_processing_step("Processing symptoms text")
        
        return {
            'raw_text': symptoms_text,
            'length': len(symptoms_text),
            'word_count': len(symptoms_text.split()),
            'processed_at': self.processing_history[-1]['timestamp']
        }
    
    def _process_patient_info(self, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate patient information."""
        self.log_processing_step("Processing patient information")
        
        processed_info = {
            'age': patient_info.get('age', 0),
            'gender': patient_info.get('gender', 'Unknown'),
            'existing_conditions': patient_info.get('existing_conditions', []),
            'medications': patient_info.get('medications', ''),
            'family_history': patient_info.get('family_history', '')
        }
        
        # Calculate risk factors
        risk_factors = []
        if processed_info['age'] > 65:
            risk_factors.append('advanced_age')
        if 'Diabetes' in processed_info['existing_conditions']:
            risk_factors.append('diabetes')
        if 'Hypertension' in processed_info['existing_conditions']:
            risk_factors.append('hypertension')
        if 'Heart Disease' in processed_info['existing_conditions']:
            risk_factors.append('heart_disease')
        
        processed_info['risk_factors'] = risk_factors
        
        return processed_info
    
    def _process_uploaded_files(self, uploaded_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process uploaded medical documents."""
        self.log_processing_step(f"Processing {len(uploaded_files)} uploaded files")
        
        processed_files = []
        
        for file_data in uploaded_files:
            processed_file = {
                'name': file_data.get('name', 'unknown'),
                'type': file_data.get('type', 'unknown'),
                'size': file_data.get('size', 0),
                'content': file_data.get('content', ''),
                'file_category': self._categorize_file(file_data.get('name', ''))
            }
            processed_files.append(processed_file)
        
        return processed_files
    
    def _process_wearable_data(self, wearable_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process wearable device data."""
        if not wearable_data:
            return {}
        
        self.log_processing_step("Processing wearable data")
        
        processed_data = {
            'heart_rate': wearable_data.get('heart_rate', 0),
            'spo2': wearable_data.get('spo2', 0),
            'sleep_hours': wearable_data.get('sleep_hours', 0)
        }
        
        # Add health indicators based on wearable data
        health_indicators = []
        
        if processed_data['heart_rate'] > 100:
            health_indicators.append('elevated_heart_rate')
        elif processed_data['heart_rate'] < 60:
            health_indicators.append('low_heart_rate')
        
        if processed_data['spo2'] < 95:
            health_indicators.append('low_oxygen_saturation')
        
        if processed_data['sleep_hours'] < 6:
            health_indicators.append('insufficient_sleep')
        elif processed_data['sleep_hours'] > 10:
            health_indicators.append('excessive_sleep')
        
        processed_data['health_indicators'] = health_indicators
        
        return processed_data
    
    def _categorize_file(self, filename: str) -> str:
        """Categorize uploaded files based on filename."""
        filename_lower = filename.lower()
        
        if any(keyword in filename_lower for keyword in ['xray', 'x-ray', 'radiograph']):
            return 'xray'
        elif any(keyword in filename_lower for keyword in ['blood', 'lab', 'test']):
            return 'blood_test'
        elif any(keyword in filename_lower for keyword in ['ecg', 'ekg', 'electrocardiogram']):
            return 'ecg'
        elif any(keyword in filename_lower for keyword in ['mri', 'ct', 'scan']):
            return 'imaging'
        else:
            return 'general_medical'
    
    def _extract_symptom_keywords(self, symptoms_text: str) -> List[str]:
        """Extract relevant medical keywords from symptoms text."""
        # Common symptom keywords for basic categorization
        symptom_keywords = [
            'pain', 'chest pain', 'headache', 'fatigue', 'shortness of breath',
            'nausea', 'vomiting', 'fever', 'cough', 'dizziness', 'weakness',
            'numbness', 'tingling', 'swelling', 'rash', 'itching', 'blurred vision',
            'palpitations', 'irregular heartbeat', 'sweating', 'chills'
        ]
        
        found_keywords = []
        symptoms_lower = symptoms_text.lower()
        
        for keyword in symptom_keywords:
            if keyword in symptoms_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _calculate_data_quality_score(self, organized_data: Dict[str, Any]) -> float:
        """Calculate a quality score for the collected data."""
        score = 0.0
        max_score = 100.0
        
        # Symptoms quality (30 points)
        symptoms = organized_data['symptoms']
        if symptoms['word_count'] >= 10:
            score += 30
        elif symptoms['word_count'] >= 5:
            score += 20
        else:
            score += 10
        
        # Patient info completeness (25 points)
        patient_info = organized_data['patient_info']
        completed_fields = sum([
            bool(patient_info['age']),
            bool(patient_info['gender'] != 'Unknown'),
            bool(patient_info['existing_conditions']),
            bool(patient_info['medications']),
            bool(patient_info['family_history'])
        ])
        score += (completed_fields / 5) * 25
        
        # File uploads (25 points)
        if organized_data['uploaded_files']:
            score += 25
        
        # Wearable data (20 points)
        if organized_data['wearable_data']:
            score += 20
        
        return min(score / max_score, 1.0)
