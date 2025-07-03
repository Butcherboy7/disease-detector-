"""
Simple Medical PDF Extractor
Basic but reliable PDF parsing for medical documents
"""

import re
import PyPDF2
from typing import Dict, Any, List
from io import BytesIO

class SimplePDFExtractor:
    """Simple and reliable medical PDF extractor"""
    
    def __init__(self):
        self.lab_patterns = {
            'tsh': r'tsh[:\s]*(\d+\.?\d*)',
            't4': r't4[:\s]*(\d+\.?\d*)',
            't3': r't3[:\s]*(\d+\.?\d*)',
            'hemoglobin': r'h(?:a|e)moglobin[:\s]*(\d+\.?\d*)',
            'ferritin': r'ferritin[:\s]*(\d+\.?\d*)',
            'glucose': r'glucose[:\s]*(\d+\.?\d*)',
            'hba1c': r'hba1c[:\s]*(\d+\.?\d*)',
            'vitamin_d': r'vitamin\s+d[:\s]*(\d+\.?\d*)',
            'testosterone': r'testosterone[:\s]*(\d+\.?\d*)',
            'cortisol': r'cortisol[:\s]*(\d+\.?\d*)',
            'cholesterol': r'cholesterol[:\s]*(\d+\.?\d*)'
        }
    
    def extract_from_pdf(self, pdf_file) -> Dict[str, Any]:
        """Extract medical data from PDF"""
        try:
            # Extract text
            text = self._extract_text(pdf_file)
            if not text:
                return {'error': 'Could not extract text from PDF'}
            
            # Extract components
            symptoms = self._extract_symptoms(text)
            lab_results = self._extract_labs(text)
            patient_info = self._extract_patient_info(text)
            
            return {
                'patient_info': patient_info,
                'symptoms': symptoms,
                'lab_results': lab_results,
                'extracted_text': text[:500] + '...' if len(text) > 500 else text
            }
            
        except Exception as e:
            return {'error': f'Extraction failed: {str(e)}'}
    
    def _extract_text(self, pdf_file) -> str:
        """Extract text from PDF"""
        try:
            if hasattr(pdf_file, 'read'):
                pdf_file.seek(0)
                reader = PyPDF2.PdfReader(pdf_file)
            else:
                reader = PyPDF2.PdfReader(BytesIO(pdf_file))
            
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except:
            return ""
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text"""
        symptoms = []
        text_lower = text.lower()
        
        # Common symptom keywords
        symptom_words = [
            'fatigue', 'tired', 'exhausted', 'weak', 'weakness',
            'dizziness', 'dizzy', 'lightheaded',
            'headache', 'head pain', 'migraine',
            'palpitations', 'heart racing', 'rapid heartbeat',
            'cold hands', 'cold feet', 'cold extremities',
            'brain fog', 'confusion', 'memory problems',
            'hair loss', 'hair thinning', 'balding',
            'dry skin', 'skin dryness',
            'weight gain', 'weight loss',
            'irregular periods', 'menstrual problems',
            'excessive thirst', 'frequent urination',
            'blurred vision', 'vision problems',
            'chest pain', 'shortness of breath',
            'anxiety', 'depression', 'mood changes',
            'insomnia', 'sleep problems',
            'constipation', 'digestive issues',
            'muscle pain', 'joint pain',
            'nausea', 'vomiting'
        ]
        
        # Find symptoms in text
        for symptom in symptom_words:
            if symptom in text_lower:
                symptoms.append(symptom)
        
        # Look for symptom patterns
        symptom_patterns = [
            r'symptoms?[:\s]*([^.]*)',
            r'complain(?:s|ts?)?[:\s]*([^.]*)',
            r'presenting with[:\s]*([^.]*)',
            r'reports?[:\s]*([^.]*)'
        ]
        
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                symptom_text = match.group(1).strip()
                if len(symptom_text) > 5 and len(symptom_text) < 200:
                    # Split by common delimiters
                    parts = re.split(r'[,;and&]', symptom_text)
                    for part in parts:
                        clean_part = part.strip()
                        if len(clean_part) > 3:
                            symptoms.append(clean_part)
        
        return list(set(symptoms))  # Remove duplicates
    
    def _extract_labs(self, text: str) -> Dict[str, str]:
        """Extract lab values from text"""
        lab_results = {}
        text_lower = text.lower()
        
        for lab_name, pattern in self.lab_patterns.items():
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                value = match.group(1)
                try:
                    # Validate numeric value
                    float_val = float(value)
                    if 0.001 <= float_val <= 10000:  # Reasonable range
                        lab_results[lab_name] = value
                        break  # Take first valid match
                except ValueError:
                    continue
        
        return lab_results
    
    def _extract_patient_info(self, text: str) -> Dict[str, Any]:
        """Extract basic patient information"""
        patient_info = {}
        text_lower = text.lower()
        
        # Age extraction
        age_pattern = r'age[:\s]*(\d+)'
        age_match = re.search(age_pattern, text_lower)
        if age_match:
            age = int(age_match.group(1))
            if 1 <= age <= 120:
                patient_info['age'] = age
        
        # Gender extraction
        if 'female' in text_lower or 'woman' in text_lower or 'mrs' in text_lower:
            patient_info['gender'] = 'female'
        elif 'male' in text_lower or 'man' in text_lower or 'mr' in text_lower:
            patient_info['gender'] = 'male'
        
        return patient_info