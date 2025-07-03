"""
Medical Analysis Engine
Realistic rule-based medical analysis that provides professional, medically accurate diagnoses
based on symptoms and lab values.
"""

import re
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime

class MedicalAnalysisEngine:
    """
    Professional medical analysis engine that uses realistic diagnostic criteria
    to provide medically accurate condition assessments.
    """
    
    def __init__(self):
        self.conditions = self._initialize_medical_conditions()
        self.lab_ranges = self._initialize_lab_reference_ranges()
        
    def _initialize_medical_conditions(self) -> Dict[str, Dict]:
        """Initialize comprehensive medical condition database with realistic criteria"""
        return {
            "hypothyroidism": {
                "name": "Hypothyroidism",
                "category": "Endocrine",
                "key_symptoms": [
                    "fatigue", "cold intolerance", "dry skin", "brain fog", 
                    "weight gain", "menstrual irregularities", "constipation",
                    "hair loss", "muscle weakness", "depression"
                ],
                "lab_criteria": {
                    "tsh": {"threshold": 4.5, "operator": ">"},
                    "t4": {"threshold": 5.0, "operator": "<"}
                },
                "symptom_weight": 0.4,
                "lab_weight": 0.6,
                "confirmation_tests": ["T3", "Thyroid Antibodies", "Reverse T3"],
                "specialist": "Endocrinologist"
            },
            
            "iron_deficiency_anemia": {
                "name": "Iron Deficiency Anemia",
                "category": "Hematologic",
                "key_symptoms": [
                    "fatigue", "dizziness", "hair loss", "cold hands and feet",
                    "palpitations", "restless legs", "brittle nails", "pale skin",
                    "shortness of breath", "cravings for ice or starch"
                ],
                "lab_criteria": {
                    "hemoglobin": {"threshold": 11.0, "operator": "<", "gender_specific": True},
                    "ferritin": {"threshold": 15.0, "operator": "<"},
                    "mcv": {"threshold": 80.0, "operator": "<"}
                },
                "symptom_weight": 0.3,
                "lab_weight": 0.7,
                "confirmation_tests": ["Iron Studies", "TIBC", "Transferrin Saturation"],
                "specialist": "Hematologist"
            },
            
            "type_2_diabetes": {
                "name": "Type 2 Diabetes",
                "category": "Endocrine",
                "key_symptoms": [
                    "excessive thirst", "frequent urination", "blurred vision",
                    "fatigue", "slow healing wounds", "frequent infections",
                    "weight loss", "increased hunger"
                ],
                "lab_criteria": {
                    "fasting_glucose": {"threshold": 126.0, "operator": ">"},
                    "hba1c": {"threshold": 6.5, "operator": ">"},
                    "random_glucose": {"threshold": 200.0, "operator": ">"}
                },
                "symptom_weight": 0.2,
                "lab_weight": 0.8,
                "confirmation_tests": ["Glucose Tolerance Test", "C-peptide", "Autoantibodies"],
                "specialist": "Endocrinologist"
            },
            
            "pcos": {
                "name": "Polycystic Ovary Syndrome (PCOS)",
                "category": "Endocrine/Gynecologic",
                "key_symptoms": [
                    "irregular periods", "hirsutism", "acne", "weight gain",
                    "hair loss", "insulin resistance", "mood changes",
                    "difficulty conceiving"
                ],
                "lab_criteria": {
                    "testosterone": {"threshold": 70.0, "operator": ">"},
                    "lh_fsh_ratio": {"threshold": 2.0, "operator": ">"},
                    "insulin": {"threshold": 25.0, "operator": ">"}
                },
                "symptom_weight": 0.5,
                "lab_weight": 0.5,
                "confirmation_tests": ["Pelvic Ultrasound", "DHEA-S", "17-OH Progesterone"],
                "specialist": "Gynecologist/Endocrinologist"
            },
            
            "food_poisoning": {
                "name": "Food Poisoning/Gastroenteritis",
                "category": "Gastrointestinal",
                "key_symptoms": [
                    "nausea", "vomiting", "diarrhea", "stomach pain", "abdominal cramps",
                    "fever", "chills", "fatigue", "dehydration", "loss of appetite"
                ],
                "lab_criteria": {},
                "symptom_weight": 0.9,
                "lab_weight": 0.1,
                "confirmation_tests": ["Stool Culture", "Blood Work", "Electrolyte Panel"],
                "specialist": "Gastroenterologist"
            },
            
            "migraine": {
                "name": "Migraine Headache",
                "category": "Neurological",
                "key_symptoms": [
                    "severe headache", "throbbing headache", "nausea", "vomiting",
                    "light sensitivity", "sound sensitivity", "visual disturbances",
                    "dizziness", "fatigue"
                ],
                "lab_criteria": {},
                "symptom_weight": 0.9,
                "lab_weight": 0.1,
                "confirmation_tests": ["CT Scan", "MRI", "Neurological Exam"],
                "specialist": "Neurologist"
            },
            
            "stress_burnout": {
                "name": "Chronic Stress/Burnout",
                "category": "Mental Health",
                "key_symptoms": [
                    "palpitations", "headache", "brain fog", "insomnia",
                    "anxiety", "irritability", "muscle tension", "digestive issues",
                    "fatigue", "mood swings"
                ],
                "lab_criteria": {
                    "cortisol": {"threshold": 25.0, "operator": ">", "time_specific": "morning"}
                },
                "symptom_weight": 0.8,
                "lab_weight": 0.2,
                "confirmation_tests": ["Cortisol Rhythm", "Stress Assessment", "Sleep Study"],
                "specialist": "Psychiatrist/Psychologist"
            },
            
            "vitamin_d_deficiency": {
                "name": "Vitamin D Deficiency",
                "category": "Nutritional",
                "key_symptoms": [
                    "bone pain", "muscle weakness", "fatigue", "depression",
                    "frequent infections", "delayed wound healing"
                ],
                "lab_criteria": {
                    "vitamin_d": {"threshold": 20.0, "operator": "<"}
                },
                "symptom_weight": 0.3,
                "lab_weight": 0.7,
                "confirmation_tests": ["PTH", "Calcium", "Phosphorus"],
                "specialist": "Primary Care/Endocrinologist"
            }
        }
    
    def _initialize_lab_reference_ranges(self) -> Dict[str, Dict]:
        """Initialize normal lab reference ranges"""
        return {
            "tsh": {"normal": (0.4, 4.0), "unit": "uIU/mL"},
            "t4": {"normal": (5.0, 12.0), "unit": "µg/dL"},
            "t3": {"normal": (80, 200), "unit": "ng/dL"},
            "hemoglobin": {"normal_female": (12.0, 15.5), "normal_male": (13.5, 17.5), "unit": "g/dL"},
            "ferritin": {"normal_female": (15, 200), "normal_male": (20, 300), "unit": "ng/mL"},
            "fasting_glucose": {"normal": (70, 99), "unit": "mg/dL"},
            "hba1c": {"normal": (4.0, 5.6), "unit": "%"},
            "vitamin_d": {"normal": (30, 100), "unit": "ng/mL"},
            "testosterone": {"normal_female": (15, 70), "normal_male": (300, 1000), "unit": "ng/dL"},
            "cortisol": {"normal_morning": (10, 20), "unit": "µg/dL"}
        }
    
    def analyze_medical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive medical analysis on patient data
        
        Args:
            data: Patient data with symptoms and lab results
            
        Returns:
            Professional medical analysis with likelihood assessments
        """
        try:
            # Extract and normalize data
            patient_info = data.get('patient_info', {})
            symptoms = self._normalize_symptoms(data.get('symptoms', []))
            lab_results = self._normalize_lab_results(data.get('lab_results', {}))
            
            # Analyze each condition
            condition_analyses = []
            for condition_key, condition_data in self.conditions.items():
                analysis = self._analyze_condition(
                    condition_key, condition_data, symptoms, lab_results, patient_info
                )
                condition_analyses.append(analysis)
            
            # Sort by clinical likelihood
            condition_analyses.sort(key=lambda x: x['clinical_score'], reverse=True)
            
            # Generate professional medical assessment
            assessment = self._generate_medical_assessment(
                condition_analyses, symptoms, lab_results, patient_info
            )
            
            return {
                'medical_assessment': assessment,
                'condition_analyses': condition_analyses,
                'lab_interpretation': self._interpret_lab_results(lab_results),
                'timestamp': datetime.now().isoformat(),
                'patient_summary': self._generate_patient_summary(patient_info, symptoms)
            }
            
        except Exception as e:
            return {
                'error': f"Medical analysis failed: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def _normalize_symptoms(self, symptoms: List[str]) -> List[str]:
        """Normalize symptom names for consistent matching"""
        normalized = []
        
        # Symptom mapping for consistent terminology
        symptom_mappings = {
            "tired": "fatigue",
            "exhausted": "fatigue",
            "weak": "fatigue",
            "cold extremities": "cold hands and feet",
            "cold feet": "cold hands and feet",
            "cold hands": "cold hands and feet",
            "memory problems": "brain fog",
            "confusion": "brain fog",
            "concentration problems": "brain fog",
            "heart racing": "palpitations",
            "rapid heartbeat": "palpitations",
            "irregular periods": "menstrual irregularities",
            "period problems": "menstrual irregularities",
            "hair thinning": "hair loss",
            "balding": "hair loss",
            "thirsty": "excessive thirst",
            "urinating frequently": "frequent urination",
            "peeing a lot": "frequent urination"
        }
        
        for symptom in symptoms:
            if isinstance(symptom, str):
                symptom_lower = symptom.lower().strip()
                normalized_symptom = symptom_mappings.get(symptom_lower, symptom_lower)
                if normalized_symptom not in normalized:
                    normalized.append(normalized_symptom)
        
        return normalized
    
    def _normalize_lab_results(self, lab_results: Dict[str, str]) -> Dict[str, float]:
        """Extract and normalize numeric lab values"""
        normalized = {}
        
        for lab_name, lab_value in lab_results.items():
            if not lab_value:
                continue
                
            # Normalize lab name
            lab_key = lab_name.lower().replace(' ', '_').replace('-', '_')
            lab_key = re.sub(r'[^\w]', '', lab_key)
            
            # Extract numeric value
            numeric_value = self._extract_numeric_value(str(lab_value))
            if numeric_value is not None:
                normalized[lab_key] = numeric_value
        
        return normalized
    
    def _extract_numeric_value(self, value_string: str) -> float:
        """Extract numeric value from lab result string"""
        if not value_string:
            return 0.0
        
        # Remove common units and extract first number
        cleaned = re.sub(r'[^\d.\s]', ' ', str(value_string))
        match = re.search(r'(\d+\.?\d*)', cleaned)
        
        if match:
            return float(match.group(1))
        return 0.0
    
    def _analyze_condition(self, condition_key: str, condition_data: Dict, 
                          symptoms: List[str], lab_results: Dict[str, float],
                          patient_info: Dict) -> Dict[str, Any]:
        """Analyze a specific medical condition using realistic clinical criteria"""
        
        # Calculate symptom match score
        symptom_score = self._calculate_symptom_match(symptoms, condition_data['key_symptoms'])
        
        # Calculate lab criteria match
        lab_score = self._evaluate_lab_criteria(lab_results, condition_data['lab_criteria'], patient_info)
        
        # Calculate weighted clinical score
        symptom_weight = condition_data['symptom_weight']
        lab_weight = condition_data['lab_weight']
        clinical_score = (symptom_score * symptom_weight) + (lab_score * lab_weight)
        
        # Determine likelihood based on realistic medical thresholds
        likelihood = self._determine_medical_likelihood(clinical_score, symptom_score, lab_score)
        
        # Get supporting evidence
        evidence = self._generate_clinical_evidence(
            symptoms, lab_results, condition_data, symptom_score, lab_score
        )
        
        return {
            'condition_key': condition_key,
            'condition_name': condition_data['name'],
            'category': condition_data['category'],
            'clinical_score': clinical_score,
            'symptom_score': symptom_score,
            'lab_score': lab_score,
            'likelihood': likelihood,
            'evidence': evidence,
            'confirmation_tests': condition_data['confirmation_tests'],
            'specialist': condition_data['specialist']
        }
    
    def _calculate_symptom_match(self, patient_symptoms: List[str], 
                                condition_symptoms: List[str]) -> float:
        """Calculate symptom match score with clinical relevance weighting"""
        if not condition_symptoms:
            return 0.0
        
        matches = 0
        total_weight = 0
        
        for condition_symptom in condition_symptoms:
            # Weight symptoms by clinical significance
            weight = self._get_symptom_weight(condition_symptom)
            total_weight += weight
            
            # Check for symptom match (fuzzy matching)
            if any(self._symptoms_match(condition_symptom, patient_symptom) 
                   for patient_symptom in patient_symptoms):
                matches += weight
        
        return matches / total_weight if total_weight > 0 else 0.0
    
    def _symptoms_match(self, condition_symptom: str, patient_symptom: str) -> bool:
        """Check if symptoms match using fuzzy logic"""
        condition_lower = condition_symptom.lower()
        patient_lower = patient_symptom.lower()
        
        # Exact match
        if condition_lower == patient_lower:
            return True
        
        # Partial match
        if condition_lower in patient_lower or patient_lower in condition_lower:
            return True
        
        # Synonym matching
        synonyms = {
            "fatigue": ["tired", "exhausted", "weakness"],
            "brain fog": ["confusion", "memory problems", "concentration"],
            "cold hands and feet": ["cold extremities", "cold hands", "cold feet"],
            "palpitations": ["heart racing", "rapid heartbeat"],
            "excessive thirst": ["thirsty", "polydipsia"],
            "frequent urination": ["urinating frequently", "polyuria"]
        }
        
        for main_symptom, synonym_list in synonyms.items():
            if condition_lower == main_symptom and any(syn in patient_lower for syn in synonym_list):
                return True
            if patient_lower == main_symptom and any(syn in condition_lower for syn in synonym_list):
                return True
        
        return False
    
    def _get_symptom_weight(self, symptom: str) -> float:
        """Get clinical weight for symptoms based on diagnostic significance"""
        high_weight_symptoms = [
            "excessive thirst", "frequent urination", "blurred vision",  # Diabetes
            "cold intolerance", "menstrual irregularities",  # Thyroid
            "cravings for ice or starch", "restless legs"  # Anemia
        ]
        
        if symptom in high_weight_symptoms:
            return 2.0
        return 1.0
    
    def _evaluate_lab_criteria(self, lab_results: Dict[str, float], 
                              lab_criteria: Dict[str, Dict], 
                              patient_info: Dict) -> float:
        """Evaluate lab criteria with realistic medical thresholds"""
        if not lab_criteria:
            return 0.5  # Neutral when no lab criteria
        
        total_score = 0
        criteria_count = 0
        
        for lab_name, criteria in lab_criteria.items():
            lab_value = lab_results.get(lab_name)
            if lab_value is None:
                continue
            
            threshold = criteria['threshold']
            operator = criteria['operator']
            
            # Apply gender-specific thresholds if needed
            if criteria.get('gender_specific') and patient_info.get('gender'):
                gender = patient_info['gender'].lower()
                if gender == 'male' and lab_name == 'hemoglobin':
                    threshold = 12.0  # Male hemoglobin threshold
            
            # Evaluate criteria
            if operator == '>' and lab_value > threshold:
                total_score += 1.0
            elif operator == '<' and lab_value < threshold:
                total_score += 1.0
            elif operator == '>=' and lab_value >= threshold:
                total_score += 1.0
            elif operator == '<=' and lab_value <= threshold:
                total_score += 1.0
            
            criteria_count += 1
        
        return total_score / criteria_count if criteria_count > 0 else 0.5
    
    def _determine_medical_likelihood(self, clinical_score: float, 
                                    symptom_score: float, lab_score: float) -> str:
        """Determine medical likelihood using realistic clinical thresholds"""
        # Definitive: Strong lab + symptom evidence
        if clinical_score >= 0.8 and lab_score >= 0.8:
            return "Definitive"
        
        # Most Likely: Good combination of labs and symptoms
        elif clinical_score >= 0.7 and (lab_score >= 0.6 or symptom_score >= 0.7):
            return "Most Likely"
        
        # Likely: Some evidence but incomplete picture
        elif clinical_score >= 0.5:
            return "Likely"
        
        # Possible: Minimal evidence
        elif clinical_score >= 0.3:
            return "Possible"
        
        # Unlikely: Very low evidence
        else:
            return "Unlikely"
    
    def _generate_clinical_evidence(self, symptoms: List[str], lab_results: Dict[str, float],
                                  condition_data: Dict, symptom_score: float, 
                                  lab_score: float) -> List[str]:
        """Generate clinical evidence supporting the diagnosis"""
        evidence = []
        
        # Lab evidence
        for lab_name, criteria in condition_data['lab_criteria'].items():
            lab_value = lab_results.get(lab_name)
            if lab_value is not None:
                threshold = criteria['threshold']
                operator = criteria['operator']
                
                if ((operator == '>' and lab_value > threshold) or 
                    (operator == '<' and lab_value < threshold)):
                    evidence.append(f"{lab_name.upper()} = {lab_value} (abnormal)")
        
        # Symptom evidence
        matching_symptoms = []
        for condition_symptom in condition_data['key_symptoms']:
            if any(self._symptoms_match(condition_symptom, patient_symptom) 
                   for patient_symptom in symptoms):
                matching_symptoms.append(condition_symptom)
        
        if matching_symptoms:
            evidence.append(f"Classic symptoms: {', '.join(matching_symptoms[:3])}")
        
        return evidence
    
    def _generate_medical_assessment(self, condition_analyses: List[Dict], 
                                   symptoms: List[str], lab_results: Dict[str, float],
                                   patient_info: Dict) -> Dict[str, Any]:
        """Generate professional medical assessment"""
        
        # Categorize conditions by likelihood
        definitive = [c for c in condition_analyses if c['likelihood'] == 'Definitive']
        most_likely = [c for c in condition_analyses if c['likelihood'] == 'Most Likely']
        likely = [c for c in condition_analyses if c['likelihood'] == 'Likely']
        possible_unlikely = [c for c in condition_analyses if c['likelihood'] in ['Possible', 'Unlikely']]
        
        # Generate primary diagnosis
        primary_diagnosis = None
        if definitive:
            primary_diagnosis = definitive[0]
        elif most_likely:
            primary_diagnosis = most_likely[0]
        elif likely:
            primary_diagnosis = likely[0]
        
        # Generate differential diagnoses
        differential = []
        if primary_diagnosis and most_likely:
            differential = [c for c in most_likely if c != primary_diagnosis][:2]
        elif likely:
            differential = likely[:2]
        
        # Generate unlikely conditions
        unlikely = [c['condition_name'] for c in possible_unlikely[:3]]
        
        return {
            'primary_diagnosis': primary_diagnosis,
            'differential_diagnoses': differential,
            'unlikely_conditions': unlikely,
            'symptom_summary': symptoms,
            'lab_summary': lab_results,
            'recommendations': self._generate_recommendations(primary_diagnosis, differential) if primary_diagnosis else {}
        }
    
    def _generate_recommendations(self, primary: Dict, differential: List[Dict]) -> Dict[str, List[str]]:
        """Generate medical recommendations"""
        recommendations = {
            'immediate_tests': [],
            'specialist_referral': [],
            'follow_up': [],
            'lifestyle': []
        }
        
        if primary:
            recommendations['immediate_tests'].extend(primary['confirmation_tests'])
            recommendations['specialist_referral'].append(f"Consult {primary['specialist']}")
        
        for diff in differential:
            recommendations['immediate_tests'].extend(diff['confirmation_tests'][:1])
        
        # Remove duplicates
        recommendations['immediate_tests'] = list(set(recommendations['immediate_tests']))
        recommendations['specialist_referral'] = list(set(recommendations['specialist_referral']))
        
        # Add general recommendations
        recommendations['follow_up'] = [
            "Schedule follow-up in 2-4 weeks",
            "Monitor symptoms daily",
            "Bring all medications to appointment"
        ]
        
        recommendations['lifestyle'] = [
            "Maintain regular sleep schedule",
            "Balanced nutrition",
            "Stress management techniques"
        ]
        
        return recommendations
    
    def _interpret_lab_results(self, lab_results: Dict[str, float]) -> Dict[str, str]:
        """Provide interpretation of lab results"""
        interpretations = {}
        
        for lab_name, value in lab_results.items():
            if lab_name in self.lab_ranges:
                normal_range = self.lab_ranges[lab_name].get('normal')
                if normal_range and isinstance(normal_range, tuple):
                    low, high = normal_range
                    if value < low:
                        interpretations[lab_name] = f"Low ({value})"
                    elif value > high:
                        interpretations[lab_name] = f"High ({value})"
                    else:
                        interpretations[lab_name] = f"Normal ({value})"
                else:
                    interpretations[lab_name] = f"Value: {value}"
        
        return interpretations
    
    def _generate_patient_summary(self, patient_info: Dict, symptoms: List[str]) -> str:
        """Generate patient summary"""
        age = patient_info.get('age', 'Unknown')
        gender = patient_info.get('gender', 'Unknown')
        
        return f"{age}-year-old {gender} presenting with {', '.join(symptoms[:3])}"
    
    def format_medical_report(self, analysis_result: Dict[str, Any]) -> str:
        """Format analysis into professional medical report"""
        if 'error' in analysis_result:
            return f"Analysis Error: {analysis_result['error']}"
        
        assessment = analysis_result['medical_assessment']
        lab_interpretation = analysis_result['lab_interpretation']
        patient_summary = analysis_result['patient_summary']
        
        report = []
        report.append("=" * 60)
        report.append("          MEDICAL SCREENING ASSESSMENT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Patient Summary
        report.append("PATIENT PRESENTATION")
        report.append("-" * 20)
        report.append(patient_summary.title())
        report.append("")
        
        # Lab Results
        if lab_interpretation:
            report.append("LABORATORY FINDINGS")
            report.append("-" * 18)
            for lab, interpretation in lab_interpretation.items():
                report.append(f"• {lab.upper()}: {interpretation}")
            report.append("")
        
        # Primary Diagnosis
        primary = assessment.get('primary_diagnosis')
        if primary:
            report.append("🔍 DIAGNOSIS")
            report.append("")
            
            # Generate direct statement based on likelihood
            if primary['likelihood'] == 'Definitive':
                report.append(f"🩺 **You have {primary['condition_name']}**")
            elif primary['likelihood'] == 'Most Likely':
                report.append(f"🩺 **You most likely have {primary['condition_name']}**")
            elif primary['likelihood'] == 'Likely':
                report.append(f"🩺 **You likely have {primary['condition_name']}**")
            else:
                report.append(f"🩺 **You possibly have {primary['condition_name']}**")
            
            report.append(f"- Category: {primary['category']}")
            if primary['evidence']:
                report.append(f"- Supporting Evidence: {'; '.join(primary['evidence'])}")
            report.append("")
        
        # Differential Diagnoses
        differential = assessment.get('differential_diagnoses', [])
        if differential:
            report.append("🤔 DIFFERENTIAL CONSIDERATIONS")
            report.append("")
            for i, diff in enumerate(differential, 1):
                report.append(f"🩺 **{i}. {diff['condition_name']}**")
                report.append(f"- Likelihood: {diff['likelihood']}")
                if diff['evidence']:
                    report.append(f"- Supporting Evidence: {'; '.join(diff['evidence'])}")
                report.append("")
        
        # Unlikely Conditions
        unlikely = assessment.get('unlikely_conditions', [])
        if unlikely:
            report.append("❌ CONDITIONS UNLIKELY")
            report.append("")
            for condition in unlikely:
                report.append(f"• {condition}")
            report.append("")
        
        # Recommendations
        recommendations = assessment.get('recommendations', {})
        if recommendations:
            report.append("🧪 RECOMMENDED NEXT STEPS")
            report.append("")
            
            if recommendations.get('immediate_tests'):
                report.append("Immediate Testing:")
                for test in recommendations['immediate_tests']:
                    report.append(f"• {test}")
                report.append("")
            
            if recommendations.get('specialist_referral'):
                report.append("Specialist Consultation:")
                for referral in recommendations['specialist_referral']:
                    report.append(f"• {referral}")
                report.append("")
            
            if recommendations.get('follow_up'):
                report.append("Follow-up Plan:")
                for follow in recommendations['follow_up']:
                    report.append(f"• {follow}")
                report.append("")
        
        # Disclaimer
        report.append("⚠️ MEDICAL DISCLAIMER")
        report.append("-" * 18)
        report.append("This screening tool is for educational purposes only.")
        report.append("Results do not constitute medical advice or diagnosis.")
        report.append("Consult healthcare professionals for proper evaluation.")
        
        return "\n".join(report)