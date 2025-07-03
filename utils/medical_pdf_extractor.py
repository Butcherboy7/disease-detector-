"""
Medical PDF Extractor
Advanced PDF parsing system for extracting symptoms and lab results from medical reports
"""

import re
import json
from typing import Dict, Any, List, Tuple, Optional
import PyPDF2
from io import BytesIO

class MedicalPDFExtractor:
    """
    Professional medical PDF extraction system that identifies and extracts
    symptoms and lab results from medical reports and lab documents.
    """
    
    def __init__(self):
        self.symptom_patterns = self._initialize_symptom_patterns()
        self.lab_patterns = self._initialize_lab_patterns()
        self.section_markers = self._initialize_section_markers()
    
    def _initialize_symptom_patterns(self) -> List[str]:
        """Initialize regex patterns for symptom extraction"""
        return [
            r'symptoms?[:\s]*([^.]*?)(?:\n|$)',
            r'chief complaint[:\s]*([^.]*?)(?:\n|$)',
            r'presenting with[:\s]*([^.]*?)(?:\n|$)',
            r'patient reports?[:\s]*([^.]*?)(?:\n|$)',
            r'complaints?[:\s]*([^.]*?)(?:\n|$)',
            r'history of present illness[:\s]*([^.]*?)(?:\n|$)'
        ]
    
    def _initialize_lab_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for lab value extraction"""
        return {
            # Thyroid markers
            'tsh': r'tsh[:\s]*(\d+\.?\d*)\s*(?:uiu/ml|miu/l|\µiu/ml)',
            't4': r'(?:free\s+)?t4[:\s]*(\d+\.?\d*)\s*(?:µg/dl|ug/dl|pmol/l)',
            't3': r'(?:free\s+)?t3[:\s]*(\d+\.?\d*)\s*(?:ng/dl|pmol/l)',
            
            # Hematology
            'hemoglobin': r'h(?:a|e)moglobin[:\s]*(\d+\.?\d*)\s*(?:g/dl|gm/dl)',
            'hematocrit': r'hematocrit[:\s]*(\d+\.?\d*)\s*(?:%|percent)',
            'ferritin': r'ferritin[:\s]*(\d+\.?\d*)\s*(?:ng/ml|µg/l)',
            'mcv': r'mcv[:\s]*(\d+\.?\d*)\s*(?:fl|femtoliters)',
            
            # Metabolic
            'glucose': r'glucose[:\s]*(\d+\.?\d*)\s*(?:mg/dl|mmol/l)',
            'fasting_glucose': r'fasting\s+glucose[:\s]*(\d+\.?\d*)\s*(?:mg/dl|mmol/l)',
            'hba1c': r'hba1c[:\s]*(\d+\.?\d*)\s*(?:%|percent)',
            'random_glucose': r'random\s+glucose[:\s]*(\d+\.?\d*)\s*(?:mg/dl|mmol/l)',
            
            # Lipids
            'cholesterol': r'(?:total\s+)?cholesterol[:\s]*(\d+\.?\d*)\s*(?:mg/dl|mmol/l)',
            'ldl': r'ldl[:\s]*(\d+\.?\d*)\s*(?:mg/dl|mmol/l)',
            'hdl': r'hdl[:\s]*(\d+\.?\d*)\s*(?:mg/dl|mmol/l)',
            'triglycerides': r'triglycerides?[:\s]*(\d+\.?\d*)\s*(?:mg/dl|mmol/l)',
            
            # Hormones
            'testosterone': r'testosterone[:\s]*(\d+\.?\d*)\s*(?:ng/dl|nmol/l)',
            'cortisol': r'cortisol[:\s]*(\d+\.?\d*)\s*(?:µg/dl|nmol/l)',
            'insulin': r'insulin[:\s]*(\d+\.?\d*)\s*(?:µiu/ml|pmol/l)',
            
            # Vitamins
            'vitamin_d': r'vitamin\s+d[:\s]*(\d+\.?\d*)\s*(?:ng/ml|nmol/l)',
            'vitamin_b12': r'vitamin\s+b12[:\s]*(\d+\.?\d*)\s*(?:pg/ml|pmol/l)',
            'folate': r'folate[:\s]*(\d+\.?\d*)\s*(?:ng/ml|nmol/l)',
            
            # Kidney function
            'creatinine': r'creatinine[:\s]*(\d+\.?\d*)\s*(?:mg/dl|µmol/l)',
            'bun': r'bun[:\s]*(\d+\.?\d*)\s*(?:mg/dl|mmol/l)',
            
            # Liver function
            'alt': r'alt[:\s]*(\d+\.?\d*)\s*(?:u/l|iu/l)',
            'ast': r'ast[:\s]*(\d+\.?\d*)\s*(?:u/l|iu/l)',
            'bilirubin': r'bilirubin[:\s]*(\d+\.?\d*)\s*(?:mg/dl|µmol/l)'
        }
    
    def _initialize_section_markers(self) -> Dict[str, List[str]]:
        """Initialize section markers for document parsing"""
        return {
            'symptoms': [
                'chief complaint', 'symptoms', 'presenting complaint',
                'history of present illness', 'patient reports',
                'subjective', 'complaints'
            ],
            'lab_results': [
                'laboratory results', 'lab results', 'laboratory findings',
                'blood work', 'test results', 'laboratory values',
                'pathology', 'biochemistry'
            ],
            'assessment': [
                'assessment', 'impression', 'diagnosis', 'clinical impression'
            ],
            'plan': [
                'plan', 'recommendations', 'treatment plan', 'follow-up'
            ]
        }
    
    def extract_medical_data(self, pdf_file) -> Dict[str, Any]:
        """
        Extract medical data from PDF file
        
        Args:
            pdf_file: PDF file object or BytesIO
            
        Returns:
            Extracted medical data with symptoms and lab results
        """
        try:
            # Extract text from PDF
            pdf_text = self._extract_pdf_text(pdf_file)
            
            if not pdf_text:
                return {
                    'error': 'Could not extract text from PDF',
                    'extracted_text': '',
                    'symptoms': [],
                    'lab_results': {}
                }
            
            # Parse sections
            sections = self._parse_document_sections(pdf_text)
            
            # Extract symptoms
            symptoms = self._extract_symptoms(sections, pdf_text)
            
            # Extract lab results
            lab_results = self._extract_lab_results(sections, pdf_text)
            
            # Extract patient info if available
            patient_info = self._extract_patient_info(pdf_text)
            
            return {
                'extracted_text': pdf_text,
                'sections': sections,
                'symptoms': symptoms,
                'lab_results': lab_results,
                'patient_info': patient_info,
                'extraction_quality': self._assess_extraction_quality(symptoms, lab_results)
            }
            
        except Exception as e:
            return {
                'error': f'PDF extraction failed: {str(e)}',
                'extracted_text': '',
                'symptoms': [],
                'lab_results': {}
            }
    
    def _extract_pdf_text(self, pdf_file) -> str:
        """Extract text from PDF using PyPDF2"""
        try:
            if hasattr(pdf_file, 'read'):
                pdf_file.seek(0)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
            else:
                pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            # Fallback to basic text extraction
            try:
                if hasattr(pdf_file, 'read'):
                    pdf_file.seek(0)
                    return pdf_file.read().decode('utf-8', errors='ignore')
                else:
                    return str(pdf_file)
            except:
                return ""
    
    def _parse_document_sections(self, text: str) -> Dict[str, str]:
        """Parse document into logical sections"""
        text_lower = text.lower()
        sections = {}
        
        for section_type, markers in self.section_markers.items():
            section_content = ""
            
            for marker in markers:
                # Find section start
                pattern = rf'{re.escape(marker)}[:\s]*([^]*?)(?=\n\n|$)'
                match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
                
                if match:
                    section_content = match.group(1).strip()
                    break
            
            if section_content:
                sections[section_type] = section_content
        
        return sections
    
    def _extract_symptoms(self, sections: Dict[str, str], full_text: str) -> List[str]:
        """Extract symptoms from text using multiple strategies"""
        symptoms = []
        
        # Strategy 1: Extract from symptoms section
        symptoms_section = sections.get('symptoms', '')
        if symptoms_section:
            extracted = self._parse_symptom_text(symptoms_section)
            symptoms.extend(extracted)
        
        # Strategy 2: Use symptom patterns on full text
        for pattern in self.symptom_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                symptom_text = match.group(1).strip()
                if symptom_text:
                    parsed_symptoms = self._parse_symptom_text(symptom_text)
                    symptoms.extend(parsed_symptoms)
        
        # Strategy 3: Look for bullet points or numbered lists
        bullet_patterns = [
            r'[-•*]\s*([^-•*\n]+)',
            r'\d+\.\s*([^\d\n]+)',
            r'[a-z]\)\s*([^a-z\)\n]+)'
        ]
        
        for pattern in bullet_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                potential_symptom = match.group(1).strip()
                if self._is_likely_symptom(potential_symptom):
                    symptoms.append(potential_symptom.lower())
        
        # Clean and deduplicate
        return list(set([s.strip().lower() for s in symptoms if s.strip()]))
    
    def _parse_symptom_text(self, text: str) -> List[str]:
        """Parse symptom text into individual symptoms"""
        if not text:
            return []
        
        # Split by common delimiters
        delimiters = [',', ';', 'and', '&', '\n', '.']
        symptoms = [text]
        
        for delimiter in delimiters:
            new_symptoms = []
            for symptom in symptoms:
                if delimiter in symptom:
                    new_symptoms.extend(symptom.split(delimiter))
                else:
                    new_symptoms.append(symptom)
            symptoms = new_symptoms
        
        # Clean up symptoms
        cleaned_symptoms = []
        for symptom in symptoms:
            cleaned = re.sub(r'^[^\w]*|[^\w]*$', '', symptom.strip())
            if cleaned and len(cleaned) > 2:
                cleaned_symptoms.append(cleaned.lower())
        
        return cleaned_symptoms
    
    def _is_likely_symptom(self, text: str) -> bool:
        """Determine if text is likely a symptom"""
        if len(text) < 3 or len(text) > 100:
            return False
        
        # Common symptom keywords
        symptom_keywords = [
            'pain', 'ache', 'fatigue', 'tired', 'dizzy', 'nausea', 'vomit',
            'headache', 'fever', 'cough', 'short', 'breath', 'chest',
            'palpitation', 'sweat', 'cold', 'hot', 'weak', 'numbness',
            'tingling', 'rash', 'itch', 'constipat', 'diarrhea', 'bloat',
            'cramp', 'spasm', 'tremor', 'insomnia', 'sleep', 'anxiety',
            'depression', 'mood', 'memory', 'confusion', 'fog'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in symptom_keywords)
    
    def _extract_lab_results(self, sections: Dict[str, str], full_text: str) -> Dict[str, str]:
        """Extract lab results using pattern matching"""
        lab_results = {}
        
        # Focus on lab results section first
        lab_section = sections.get('lab_results', '')
        search_text = lab_section if lab_section else full_text
        
        # Apply lab patterns
        for lab_name, pattern in self.lab_patterns.items():
            matches = re.finditer(pattern, search_text, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                if self._is_valid_lab_value(value):
                    # Store with unit information if found in the match
                    unit_match = re.search(r'(\d+\.?\d*)\s*([a-zA-Z/%µ]+)', match.group(0))
                    if unit_match:
                        lab_results[lab_name] = f"{value} {unit_match.group(2)}"
                    else:
                        lab_results[lab_name] = value
                    break  # Take first valid match
        
        # Look for lab tables or structured data
        table_results = self._extract_lab_table_data(search_text)
        lab_results.update(table_results)
        
        return lab_results
    
    def _is_valid_lab_value(self, value: str) -> bool:
        """Validate if extracted value is a reasonable lab value"""
        try:
            num_value = float(value)
            # Basic sanity checks
            return 0.001 <= num_value <= 10000
        except ValueError:
            return False
    
    def _extract_lab_table_data(self, text: str) -> Dict[str, str]:
        """Extract lab data from table-like structures"""
        lab_results = {}
        
        # Look for table-like patterns
        lines = text.split('\n')
        
        for line in lines:
            # Pattern: Test Name    Value    Unit    Reference
            table_pattern = r'([a-zA-Z][a-zA-Z\s]{2,20})\s+(\d+\.?\d*)\s+([a-zA-Z/%µ]+)'
            match = re.search(table_pattern, line)
            
            if match:
                test_name = match.group(1).strip().lower().replace(' ', '_')
                value = match.group(2)
                unit = match.group(3)
                
                # Map common test names
                name_mappings = {
                    'thyroid_stimulating_hormone': 'tsh',
                    'thyroxine': 't4',
                    'triiodothyronine': 't3',
                    'haemoglobin': 'hemoglobin',
                    'blood_glucose': 'glucose',
                    'glycated_hemoglobin': 'hba1c'
                }
                
                mapped_name = name_mappings.get(test_name, test_name)
                lab_results[mapped_name] = f"{value} {unit}"
        
        return lab_results
    
    def _extract_patient_info(self, text: str) -> Dict[str, Any]:
        """Extract basic patient information"""
        patient_info = {}
        
        # Age extraction
        age_patterns = [
            r'age[:\s]*(\d+)',
            r'(\d+)\s*(?:year|yr|y)[\s\-]*old',
            r'(\d+)\s*(?:years?|yrs?)'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 1 <= age <= 120:
                    patient_info['age'] = age
                    break
        
        # Gender extraction
        gender_patterns = [
            r'gender[:\s]*(male|female|m|f)',
            r'sex[:\s]*(male|female|m|f)',
            r'\b(male|female)\b',
            r'\b(mr|mrs|ms)\b'
        ]
        
        for pattern in gender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gender_text = match.group(1).lower()
                if gender_text in ['male', 'm', 'mr']:
                    patient_info['gender'] = 'male'
                elif gender_text in ['female', 'f', 'mrs', 'ms']:
                    patient_info['gender'] = 'female'
                break
        
        # Name extraction (basic)
        name_patterns = [
            r'patient[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'name[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                patient_info['name'] = match.group(1).strip()
                break
        
        return patient_info
    
    def _assess_extraction_quality(self, symptoms: List[str], lab_results: Dict[str, str]) -> str:
        """Assess the quality of extraction"""
        if len(symptoms) >= 3 and len(lab_results) >= 2:
            return "Excellent"
        elif len(symptoms) >= 2 or len(lab_results) >= 1:
            return "Good"
        elif len(symptoms) >= 1 or lab_results:
            return "Fair"
        else:
            return "Poor"
    
    def create_structured_output(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert extraction result to structured medical format"""
        if 'error' in extraction_result:
            return extraction_result
        
        return {
            'patient_info': extraction_result.get('patient_info', {}),
            'symptoms': extraction_result.get('symptoms', []),
            'lab_results': extraction_result.get('lab_results', {}),
            'extraction_metadata': {
                'quality': extraction_result.get('extraction_quality', 'Unknown'),
                'text_length': len(extraction_result.get('extracted_text', '')),
                'sections_found': list(extraction_result.get('sections', {}).keys())
            }
        }