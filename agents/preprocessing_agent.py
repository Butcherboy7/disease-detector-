"""
Preprocessing Agent
Responsible for cleaning, extracting, and preparing data for analysis.
Handles document parsing, image processing, and data normalization.
"""

import json
import base64
import re
from typing import Dict, Any, List
from .base_agent import BaseAgent
from utils.file_processors import FileProcessor

class PreprocessingAgent(BaseAgent):
    """
    Agent responsible for preprocessing and cleaning collected data.
    Extracts meaningful information from documents and normalizes data.
    """
    
    def __init__(self):
        super().__init__("PreprocessingAgent")
        self.file_processor = FileProcessor()
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess collected data for analysis.
        
        Args:
            data: Organized data from InputCollectionAgent
            
        Returns:
            Preprocessed and cleaned data ready for analysis
        """
        try:
            self.log_processing_step("Starting data preprocessing")
            
            # Extract organized data
            if 'collection_result' in data:
                organized_data = data['collection_result']['organized_data']
            else:
                organized_data = data
            
            # Process symptoms text
            processed_symptoms = self._preprocess_symptoms(organized_data['symptoms'])
            
            # Process uploaded files
            processed_files = self._preprocess_files(organized_data['uploaded_files'])
            
            # Normalize patient data
            normalized_patient_data = self._normalize_patient_data(organized_data['patient_info'])
            
            # Process wearable data
            processed_wearable = self._preprocess_wearable_data(organized_data['wearable_data'])
            
            # Extract medical features
            medical_features = self._extract_medical_features(
                processed_symptoms, processed_files, normalized_patient_data, processed_wearable
            )
            
            # Compile preprocessed data
            preprocessed_data = {
                'processed_symptoms': processed_symptoms,
                'processed_files': processed_files,
                'normalized_patient_data': normalized_patient_data,
                'processed_wearable': processed_wearable,
                'medical_features': medical_features,
                'preprocessing_metadata': {
                    'extracted_entities': medical_features.get('entities', []),
                    'feature_count': len(medical_features.get('features', {})),
                    'data_completeness': self._calculate_completeness(medical_features)
                }
            }
            
            self.log_processing_step("Data preprocessing completed successfully")
            
            return self.create_success_response({
                'preprocessed_data': preprocessed_data
            })
            
        except Exception as e:
            return self.handle_error(e, "data preprocessing")
    
    def _preprocess_symptoms(self, symptoms_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess and extract information from symptoms text."""
        self.log_processing_step("Preprocessing symptoms text")
        
        raw_text = symptoms_data['raw_text']
        
        # Clean and normalize text
        cleaned_text = self._clean_text(raw_text)
        
        # Extract medical entities
        entities = self._extract_medical_entities(cleaned_text)
        
        # Analyze symptom severity
        severity_analysis = self._analyze_symptom_severity(cleaned_text)
        
        # Extract temporal information
        temporal_info = self._extract_temporal_information(cleaned_text)
        
        return {
            'original_text': raw_text,
            'cleaned_text': cleaned_text,
            'entities': entities,
            'severity_analysis': severity_analysis,
            'temporal_info': temporal_info,
            'word_count': len(cleaned_text.split()),
            'sentence_count': len(cleaned_text.split('.'))
        }
    
    def _preprocess_files(self, files_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess uploaded medical files."""
        self.log_processing_step(f"Preprocessing {len(files_data)} files")
        
        processed_files = []
        
        for file_data in files_data:
            try:
                processed_file = {
                    'name': file_data['name'],
                    'type': file_data['type'],
                    'category': file_data['file_category'],
                    'size': file_data['size']
                }
                
                # Process based on file type
                if file_data['type'].startswith('image/'):
                    processed_file.update(self._process_image_file(file_data))
                elif file_data['type'] == 'application/pdf':
                    processed_file.update(self._process_pdf_file(file_data))
                else:
                    processed_file['extracted_text'] = "File type not supported for text extraction"
                
                processed_files.append(processed_file)
                
            except Exception as e:
                self.logger.warning(f"Error processing file {file_data['name']}: {str(e)}")
                processed_files.append({
                    'name': file_data['name'],
                    'error': str(e),
                    'processed': False
                })
        
        return processed_files
    
    def _normalize_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize patient information for analysis."""
        self.log_processing_step("Normalizing patient data")
        
        normalized = {
            'age_group': self._categorize_age(patient_data['age']),
            'gender_encoded': self._encode_gender(patient_data['gender']),
            'risk_score': self._calculate_risk_score(patient_data),
            'condition_flags': self._create_condition_flags(patient_data['existing_conditions']),
            'medication_categories': self._categorize_medications(patient_data['medications']),
            'family_risk_factors': self._extract_family_risks(patient_data['family_history'])
        }
        
        # Keep original data as well
        normalized.update(patient_data)
        
        return normalized
    
    def _preprocess_wearable_data(self, wearable_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess wearable device data."""
        if not wearable_data:
            return {}
        
        self.log_processing_step("Preprocessing wearable data")
        
        processed = {
            'heart_rate_category': self._categorize_heart_rate(wearable_data['heart_rate']),
            'spo2_category': self._categorize_spo2(wearable_data['spo2']),
            'sleep_category': self._categorize_sleep(wearable_data['sleep_hours']),
            'vitals_score': self._calculate_vitals_score(wearable_data)
        }
        
        # Keep original values
        processed.update(wearable_data)
        
        return processed
    
    def _extract_medical_features(self, symptoms, files, patient_data, wearable_data) -> Dict[str, Any]:
        """Extract comprehensive medical features for analysis."""
        self.log_processing_step("Extracting medical features")
        
        features = {}
        entities = []
        
        # Symptom features
        if symptoms:
            features.update({
                'symptom_severity_score': symptoms['severity_analysis'].get('severity_score', 0),
                'symptom_urgency': symptoms['severity_analysis'].get('urgency_level', 'low'),
                'chronic_indicators': symptoms['temporal_info'].get('chronic_indicators', 0),
                'acute_indicators': symptoms['temporal_info'].get('acute_indicators', 0)
            })
            entities.extend(symptoms.get('entities', []))
        
        # Patient features
        if patient_data:
            features.update({
                'age_risk_factor': 1 if patient_data['age'] > 65 else 0,
                'comorbidity_count': len(patient_data.get('existing_conditions', [])),
                'family_history_risk': 1 if patient_data.get('family_risk_factors') else 0
            })
        
        # Wearable features
        if wearable_data:
            features.update({
                'abnormal_heart_rate': 1 if wearable_data.get('heart_rate_category') in ['high', 'low'] else 0,
                'low_oxygen': 1 if wearable_data.get('spo2_category') == 'low' else 0,
                'sleep_disorder': 1 if wearable_data.get('sleep_category') in ['insufficient', 'excessive'] else 0
            })
        
        # File-based features
        features['has_imaging'] = 1 if any(f.get('category') in ['xray', 'imaging'] for f in files) else 0
        features['has_lab_results'] = 1 if any(f.get('category') == 'blood_test' for f in files) else 0
        
        return {
            'features': features,
            'entities': entities,
            'feature_vector': list(features.values())
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep medical punctuation
        text = re.sub(r'[^\w\s\.\,\-\(\)]', '', text)
        return text.strip()
    
    def _extract_medical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical entities from text using rule-based approach."""
        entities = []
        
        # Common medical terms and patterns
        pain_patterns = re.findall(r'\b\w*pain\b|\bache\b|\baches\b|\baching\b', text, re.IGNORECASE)
        for match in pain_patterns:
            entities.append({'type': 'symptom', 'value': match, 'category': 'pain'})
        
        # Body parts
        body_parts = ['chest', 'head', 'back', 'stomach', 'leg', 'arm', 'neck', 'shoulder']
        for part in body_parts:
            if part in text.lower():
                entities.append({'type': 'body_part', 'value': part, 'category': 'anatomy'})
        
        # Severity indicators
        severity_words = ['severe', 'mild', 'moderate', 'intense', 'sharp', 'dull', 'chronic', 'acute']
        for word in severity_words:
            if word in text.lower():
                entities.append({'type': 'severity', 'value': word, 'category': 'intensity'})
        
        return entities
    
    def _analyze_symptom_severity(self, text: str) -> Dict[str, Any]:
        """Analyze severity of symptoms from text."""
        severity_keywords = {
            'severe': 5, 'intense': 5, 'excruciating': 5,
            'moderate': 3, 'noticeable': 3,
            'mild': 1, 'slight': 1, 'minor': 1
        }
        
        urgency_keywords = {
            'emergency': 5, 'urgent': 4, 'sudden': 4,
            'worsening': 3, 'increasing': 3,
            'chronic': 1, 'ongoing': 1
        }
        
        text_lower = text.lower()
        
        severity_score = 0
        urgency_score = 0
        
        for keyword, score in severity_keywords.items():
            if keyword in text_lower:
                severity_score = max(severity_score, score)
        
        for keyword, score in urgency_keywords.items():
            if keyword in text_lower:
                urgency_score = max(urgency_score, score)
        
        urgency_levels = {0: 'low', 1: 'low', 2: 'medium', 3: 'medium', 4: 'high', 5: 'high'}
        
        return {
            'severity_score': severity_score,
            'urgency_score': urgency_score,
            'urgency_level': urgency_levels.get(urgency_score, 'low')
        }
    
    def _extract_temporal_information(self, text: str) -> Dict[str, Any]:
        """Extract temporal information about symptoms."""
        chronic_indicators = len(re.findall(r'\b(months|years|chronic|ongoing|persistent)\b', text, re.IGNORECASE))
        acute_indicators = len(re.findall(r'\b(sudden|today|yesterday|hours|minutes)\b', text, re.IGNORECASE))
        
        return {
            'chronic_indicators': chronic_indicators,
            'acute_indicators': acute_indicators
        }
    
    def _process_image_file(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process image files (X-rays, etc.)."""
        return self.file_processor.process_image(file_data)
    
    def _process_pdf_file(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF files (reports, etc.)."""
        return self.file_processor.process_pdf(file_data)
    
    def _categorize_age(self, age: int) -> str:
        """Categorize age into groups."""
        if age < 18:
            return 'pediatric'
        elif age < 35:
            return 'young_adult'
        elif age < 55:
            return 'middle_aged'
        elif age < 75:
            return 'elderly'
        else:
            return 'geriatric'
    
    def _encode_gender(self, gender: str) -> int:
        """Encode gender for analysis."""
        gender_map = {'Male': 1, 'Female': 2, 'Other': 3}
        return gender_map.get(gender, 0)
    
    def _calculate_risk_score(self, patient_data: Dict[str, Any]) -> float:
        """Calculate overall risk score based on patient data."""
        score = 0.0
        
        # Age factor
        if patient_data['age'] > 65:
            score += 0.3
        elif patient_data['age'] > 45:
            score += 0.1
        
        # Existing conditions
        high_risk_conditions = ['Diabetes', 'Heart Disease', 'Hypertension']
        for condition in patient_data.get('existing_conditions', []):
            if condition in high_risk_conditions:
                score += 0.2
        
        return min(score, 1.0)
    
    def _create_condition_flags(self, conditions: List[str]) -> Dict[str, bool]:
        """Create boolean flags for medical conditions."""
        return {
            'has_diabetes': 'Diabetes' in conditions,
            'has_hypertension': 'Hypertension' in conditions,
            'has_heart_disease': 'Heart Disease' in conditions,
            'has_asthma': 'Asthma' in conditions
        }
    
    def _categorize_medications(self, medications_text: str) -> List[str]:
        """Categorize medications into therapeutic classes."""
        if not medications_text:
            return []
        
        categories = []
        text_lower = medications_text.lower()
        
        medication_categories = {
            'cardiovascular': ['lisinopril', 'metoprolol', 'amlodipine', 'atorvastatin'],
            'diabetes': ['metformin', 'insulin', 'glipizide'],
            'pain': ['ibuprofen', 'acetaminophen', 'aspirin'],
            'respiratory': ['albuterol', 'fluticasone']
        }
        
        for category, meds in medication_categories.items():
            if any(med in text_lower for med in meds):
                categories.append(category)
        
        return categories
    
    def _extract_family_risks(self, family_history: str) -> List[str]:
        """Extract family history risk factors."""
        if not family_history:
            return []
        
        risks = []
        text_lower = family_history.lower()
        
        risk_conditions = ['diabetes', 'heart disease', 'cancer', 'stroke', 'hypertension']
        
        for condition in risk_conditions:
            if condition in text_lower:
                risks.append(condition)
        
        return risks
    
    def _categorize_heart_rate(self, heart_rate: int) -> str:
        """Categorize heart rate."""
        if heart_rate < 60:
            return 'low'
        elif heart_rate > 100:
            return 'high'
        else:
            return 'normal'
    
    def _categorize_spo2(self, spo2: int) -> str:
        """Categorize oxygen saturation."""
        if spo2 < 95:
            return 'low'
        else:
            return 'normal'
    
    def _categorize_sleep(self, sleep_hours: float) -> str:
        """Categorize sleep duration."""
        if sleep_hours < 6:
            return 'insufficient'
        elif sleep_hours > 10:
            return 'excessive'
        else:
            return 'normal'
    
    def _calculate_vitals_score(self, wearable_data: Dict[str, Any]) -> float:
        """Calculate overall vitals score."""
        score = 1.0
        
        heart_rate = wearable_data.get('heart_rate', 70)
        if heart_rate < 60 or heart_rate > 100:
            score -= 0.3
        
        spo2 = wearable_data.get('spo2', 98)
        if spo2 < 95:
            score -= 0.4
        
        sleep_hours = wearable_data.get('sleep_hours', 8)
        if sleep_hours < 6 or sleep_hours > 10:
            score -= 0.2
        
        return max(score, 0.0)
    
    def _calculate_completeness(self, medical_features: Dict[str, Any]) -> float:
        """Calculate data completeness score."""
        total_features = len(medical_features.get('features', {}))
        non_zero_features = sum(1 for v in medical_features.get('features', {}).values() if v != 0)
        
        if total_features == 0:
            return 0.0
        
        return non_zero_features / total_features
