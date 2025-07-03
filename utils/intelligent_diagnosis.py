"""
Intelligent Diagnosis Engine
Advanced rule-based diagnosis system with structured JSON input and natural language reasoning
"""

import json
import re
from typing import Dict, Any, List, Tuple
from datetime import datetime

class IntelligentDiagnosisEngine:
    """
    Advanced diagnosis engine that uses structured input and rule-based reasoning
    to provide natural language explanations and likelihood assessments.
    """
    
    def __init__(self):
        self.disease_rules = self._initialize_disease_rules()
        self.symptom_weights = self._initialize_symptom_weights()
        self.lab_reference_ranges = self._initialize_lab_ranges()
        
    def _initialize_disease_rules(self) -> Dict[str, Dict]:
        """Initialize disease-specific diagnostic rules"""
        return {
            "iron_deficiency_anemia": {
                "name": "Iron Deficiency Anemia",
                "key_symptoms": ["fatigue", "dizziness", "cold hands", "cold feet", "hair loss", "brittle nails", "restless legs"],
                "lab_indicators": {
                    "hemoglobin": {"female": 11.0, "male": 12.0},
                    "ferritin": {"female": 15.0, "male": 20.0},
                    "mcv": 80.0  # Mean corpuscular volume
                },
                "risk_factors": ["female", "vegetarian", "heavy_periods", "pregnancy"],
                "age_groups": ["20-50"],
                "description": "fatigue, hair thinning, cold extremities, and dizziness"
            },
            "hypothyroidism": {
                "name": "Hypothyroidism",
                "key_symptoms": ["fatigue", "brain fog", "cold intolerance", "weight gain", "constipation", "dry skin", "hair loss"],
                "lab_indicators": {
                    "tsh": 4.5,  # TSH > 4.5 indicates hypothyroidism
                    "t4": 0.8   # T4 < 0.8 indicates hypothyroidism
                },
                "risk_factors": ["female", "family_history", "autoimmune"],
                "age_groups": ["30-60"],
                "description": "brain fog, fatigue, cold intolerance, and elevated TSH"
            },
            "vitamin_d_deficiency": {
                "name": "Vitamin D Deficiency",
                "key_symptoms": ["fatigue", "bone pain", "muscle weakness", "depression", "frequent infections"],
                "lab_indicators": {
                    "vitamin_d": 20.0  # < 20 ng/mL indicates deficiency
                },
                "risk_factors": ["limited_sun_exposure", "dark_skin", "indoor_lifestyle"],
                "age_groups": ["all"],
                "description": "fatigue, memory issues, and low vitamin D levels"
            },
            "stress_anxiety": {
                "name": "Stress/Anxiety Disorder",
                "key_symptoms": ["fatigue", "palpitations", "anxiety", "headache", "insomnia", "muscle tension"],
                "lab_indicators": {},
                "risk_factors": ["high_stress_job", "life_changes", "family_history"],
                "age_groups": ["20-40"],
                "description": "fatigue, palpitations, anxiety, and headaches"
            },
            "diabetes": {
                "name": "Type 2 Diabetes",
                "key_symptoms": ["frequent urination", "excessive thirst", "fatigue", "blurred vision", "slow healing"],
                "lab_indicators": {
                    "glucose": 126.0,  # Fasting glucose > 126 mg/dL
                    "hba1c": 6.5      # HbA1c > 6.5%
                },
                "risk_factors": ["obesity", "family_history", "sedentary"],
                "age_groups": ["40+"],
                "description": "frequent urination, excessive thirst, and elevated blood sugar"
            },
            "heart_disease": {
                "name": "Heart Disease",
                "key_symptoms": ["chest pain", "shortness of breath", "fatigue", "palpitations", "swelling"],
                "lab_indicators": {
                    "cholesterol": 240.0,  # Total cholesterol > 240 mg/dL
                    "ldl": 160.0          # LDL > 160 mg/dL
                },
                "risk_factors": ["smoking", "high_blood_pressure", "diabetes", "family_history"],
                "age_groups": ["45+"],
                "description": "chest pain, shortness of breath, and cardiovascular risk factors"
            }
        }
    
    def _initialize_symptom_weights(self) -> Dict[str, float]:
        """Initialize symptom importance weights for different conditions"""
        return {
            "fatigue": 0.8,
            "dizziness": 0.7,
            "cold_hands": 0.6,
            "hair_loss": 0.5,
            "brain_fog": 0.7,
            "palpitations": 0.6,
            "headache": 0.5,
            "chest_pain": 0.9,
            "shortness_of_breath": 0.8,
            "frequent_urination": 0.8,
            "excessive_thirst": 0.7
        }
    
    def _initialize_lab_ranges(self) -> Dict[str, Any]:
        """Initialize normal lab reference ranges"""
        return {
            "hemoglobin": {"female": (12.0, 15.5), "male": (13.5, 17.5)},
            "ferritin": {"female": (15, 200), "male": (20, 300)},
            "tsh": (0.4, 4.0),
            "vitamin_d": (30, 100),
            "glucose": (70, 99),
            "cholesterol": (0, 200),
            "ldl": (0, 100)
        }
    
    def analyze_structured_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze structured JSON input and provide intelligent diagnosis
        
        Args:
            input_data: Structured JSON with patient_info, symptoms, lab_results
            
        Returns:
            Comprehensive diagnosis with likelihood tiers and explanations
        """
        try:
            # Extract components
            patient_info = input_data.get('patient_info', {})
            symptoms = input_data.get('symptoms', [])
            lab_results = input_data.get('lab_results', {})
            
            # Normalize symptoms
            normalized_symptoms = self._normalize_symptoms(symptoms)
            
            # Analyze each condition
            condition_analyses = []
            for condition_key, condition_data in self.disease_rules.items():
                analysis = self._analyze_condition(
                    condition_key, condition_data, 
                    normalized_symptoms, lab_results, patient_info
                )
                condition_analyses.append(analysis)
            
            # Sort by likelihood score
            condition_analyses.sort(key=lambda x: x['likelihood_score'], reverse=True)
            
            # Categorize conditions
            categorized_conditions = self._categorize_conditions(condition_analyses)
            
            # Generate natural language explanations
            explanations = self._generate_explanations(categorized_conditions, symptoms, lab_results)
            
            # Calculate urgency level
            urgency_level = self._calculate_urgency(condition_analyses, symptoms, lab_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(categorized_conditions, lab_results)
            
            return {
                'patient_info': patient_info,
                'symptoms_analyzed': normalized_symptoms,
                'lab_results': lab_results,
                'categorized_conditions': categorized_conditions,
                'explanations': explanations,
                'urgency_level': urgency_level,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': f"Diagnosis analysis failed: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def _normalize_symptoms(self, symptoms: List[str]) -> List[str]:
        """Normalize symptom names for consistent matching"""
        normalized = []
        symptom_mappings = {
            "tired": "fatigue",
            "exhausted": "fatigue",
            "weak": "fatigue",
            "dizzy": "dizziness",
            "lightheaded": "dizziness",
            "cold_extremities": "cold hands",
            "cold_feet": "cold hands",
            "hair_thinning": "hair loss",
            "memory_problems": "brain fog",
            "confusion": "brain fog",
            "heart_racing": "palpitations",
            "rapid_heartbeat": "palpitations"
        }
        
        for symptom in symptoms:
            symptom_lower = symptom.lower().replace(" ", "_")
            normalized.append(symptom_mappings.get(symptom_lower, symptom_lower))
        
        return list(set(normalized))  # Remove duplicates
    
    def _analyze_condition(self, condition_key: str, condition_data: Dict, 
                          symptoms: List[str], lab_results: Dict, 
                          patient_info: Dict) -> Dict[str, Any]:
        """Analyze a specific condition against patient data"""
        
        # Calculate symptom match score
        symptom_score = self._calculate_symptom_score(symptoms, condition_data['key_symptoms'])
        
        # Calculate lab score
        lab_score = self._calculate_lab_score(lab_results, condition_data['lab_indicators'], 
                                            patient_info.get('gender', 'unknown'))
        
        # Calculate risk factor score
        risk_score = self._calculate_risk_score(patient_info, condition_data['risk_factors'])
        
        # Calculate age appropriateness
        age_score = self._calculate_age_score(patient_info.get('age', 0), condition_data['age_groups'])
        
        # Combined likelihood score
        likelihood_score = (symptom_score * 0.4 + lab_score * 0.3 + 
                           risk_score * 0.2 + age_score * 0.1)
        
        # Determine likelihood tier
        likelihood_tier = self._determine_likelihood_tier(likelihood_score)
        
        return {
            'condition_key': condition_key,
            'condition_name': condition_data['name'],
            'likelihood_score': likelihood_score,
            'likelihood_tier': likelihood_tier,
            'symptom_score': symptom_score,
            'lab_score': lab_score,
            'risk_score': risk_score,
            'age_score': age_score,
            'matching_symptoms': [s for s in symptoms if s in condition_data['key_symptoms']],
            'description': condition_data['description'],
            'confirmation_tests': self._get_confirmation_tests(condition_key)
        }
    
    def _calculate_symptom_score(self, patient_symptoms: List[str], 
                               condition_symptoms: List[str]) -> float:
        """Calculate symptom match score"""
        if not condition_symptoms:
            return 0.0
        
        matches = sum(1 for symptom in patient_symptoms if symptom in condition_symptoms)
        return matches / len(condition_symptoms)
    
    def _calculate_lab_score(self, lab_results: Dict, lab_indicators: Dict, 
                           gender: str) -> float:
        """Calculate lab abnormality score"""
        if not lab_indicators:
            return 0.5  # Neutral score when no lab indicators
        
        scores = []
        for lab_name, threshold in lab_indicators.items():
            lab_value = self._extract_lab_value(lab_results.get(lab_name, ""))
            if lab_value > 0:  # Only process valid numeric values
                if isinstance(threshold, dict) and gender in threshold:
                    threshold_val = threshold[gender]
                elif isinstance(threshold, (int, float)):
                    threshold_val = threshold
                else:
                    continue
                
                if lab_name in ['tsh', 'glucose', 'cholesterol', 'ldl']:
                    # Higher values indicate abnormality
                    score = 1.0 if lab_value > threshold_val else 0.0
                else:
                    # Lower values indicate abnormality (hemoglobin, ferritin)
                    score = 1.0 if lab_value < threshold_val else 0.0
                scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _extract_lab_value(self, lab_string: str) -> float:
        """Extract numeric value from lab result string"""
        if not lab_string:
            return 0.0
        
        # Use regex to extract first number from string
        match = re.search(r'(\d+\.?\d*)', str(lab_string))
        return float(match.group(1)) if match else 0.0
    
    def _calculate_risk_score(self, patient_info: Dict, risk_factors: List[str]) -> float:
        """Calculate risk factor score"""
        if not risk_factors:
            return 0.5
        
        patient_age = patient_info.get('age', 0)
        patient_gender = patient_info.get('gender', '').lower()
        
        score = 0.0
        for factor in risk_factors:
            if factor == 'female' and patient_gender == 'female':
                score += 1.0
            elif factor == 'male' and patient_gender == 'male':
                score += 1.0
            elif factor in ['high_stress_job', 'sedentary', 'smoking']:
                score += 0.5  # Assume moderate risk for lifestyle factors
        
        return min(score / len(risk_factors), 1.0)
    
    def _calculate_age_score(self, patient_age: int, age_groups: List[str]) -> float:
        """Calculate age appropriateness score"""
        if not age_groups or 'all' in age_groups:
            return 1.0
        
        for age_group in age_groups:
            if '+' in age_group:
                min_age = int(age_group.replace('+', ''))
                if patient_age >= min_age:
                    return 1.0
            elif '-' in age_group:
                min_age, max_age = map(int, age_group.split('-'))
                if min_age <= patient_age <= max_age:
                    return 1.0
        
        return 0.3  # Lower score for inappropriate age
    
    def _determine_likelihood_tier(self, score: float) -> str:
        """Determine likelihood tier based on score"""
        if score >= 0.8:
            return "Very Likely"
        elif score >= 0.6:
            return "Likely"
        elif score >= 0.4:
            return "Moderate"
        elif score >= 0.2:
            return "Possible"
        else:
            return "Unlikely"
    
    def _categorize_conditions(self, analyses: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize conditions by likelihood"""
        categorized = {
            "most_likely": [],
            "possibly_present": [],
            "ruled_out": []
        }
        
        for analysis in analyses:
            tier = analysis['likelihood_tier']
            if tier in ["Very Likely", "Likely"]:
                categorized["most_likely"].append(analysis)
            elif tier in ["Moderate", "Possible"]:
                categorized["possibly_present"].append(analysis)
            else:
                categorized["ruled_out"].append(analysis)
        
        return categorized
    
    def _generate_explanations(self, categorized: Dict, symptoms: List[str], 
                             lab_results: Dict) -> Dict[str, str]:
        """Generate natural language explanations"""
        explanations = {}
        
        # Main explanation
        if categorized["most_likely"]:
            top_condition = categorized["most_likely"][0]
            explanations["main"] = (
                f"Based on the symptoms of {', '.join(symptoms[:3])}, "
                f"the system strongly suspects **{top_condition['condition_name']}**. "
                f"This condition matches the reported {top_condition['description']}. "
                f"Likelihood: **{top_condition['likelihood_tier']}**."
            )
        else:
            explanations["main"] = (
                "The symptoms reported don't strongly point to a single condition. "
                "Further evaluation may be needed to determine the underlying cause."
            )
        
        # Lab explanation
        if lab_results:
            lab_findings = []
            for lab_name, lab_value in lab_results.items():
                numeric_value = self._extract_lab_value(lab_value)
                if numeric_value is not None:
                    lab_findings.append(f"{lab_name}: {lab_value}")
            
            if lab_findings:
                explanations["lab"] = (
                    f"Lab results show: {', '.join(lab_findings)}. "
                    f"These values help support or rule out certain conditions."
                )
        
        return explanations
    
    def _calculate_urgency(self, analyses: List[Dict], symptoms: List[str], 
                          lab_results: Dict) -> str:
        """Calculate urgency level"""
        high_urgency_symptoms = ['chest_pain', 'shortness_of_breath', 'severe_headache']
        moderate_urgency_symptoms = ['palpitations', 'dizziness', 'fatigue']
        
        # Check for high urgency symptoms
        if any(symptom in high_urgency_symptoms for symptom in symptoms):
            return "High"
        
        # Check for very likely serious conditions
        serious_conditions = ['heart_disease', 'diabetes']
        for analysis in analyses:
            if (analysis['condition_key'] in serious_conditions and 
                analysis['likelihood_tier'] in ['Very Likely', 'Likely']):
                return "High"
        
        # Check for moderate urgency
        if any(symptom in moderate_urgency_symptoms for symptom in symptoms):
            return "Medium"
        
        return "Low"
    
    def _generate_recommendations(self, categorized: Dict, lab_results: Dict) -> List[str]:
        """Generate specific recommendations"""
        recommendations = []
        
        # Recommendations based on most likely conditions
        for condition in categorized["most_likely"]:
            test_recs = condition.get('confirmation_tests', [])
            if test_recs:
                recommendations.append(f"Schedule {', '.join(test_recs)} to confirm {condition['condition_name']}")
        
        # General recommendations
        if not lab_results:
            recommendations.append("Consider basic lab work including CBC, comprehensive metabolic panel, and thyroid function")
        
        recommendations.extend([
            "Track symptoms daily to monitor patterns",
            "Maintain a balanced diet and regular exercise",
            "Follow up with healthcare provider for proper evaluation"
        ])
        
        return recommendations
    
    def _get_confirmation_tests(self, condition_key: str) -> List[str]:
        """Get recommended confirmation tests for each condition"""
        test_mappings = {
            "iron_deficiency_anemia": ["CBC", "Ferritin", "Iron studies"],
            "hypothyroidism": ["TSH", "Free T3", "Free T4"],
            "vitamin_d_deficiency": ["25-OH Vitamin D"],
            "diabetes": ["Fasting glucose", "HbA1c", "Glucose tolerance test"],
            "heart_disease": ["EKG", "Echocardiogram", "Stress test"],
            "stress_anxiety": ["Psychological evaluation", "Cortisol levels"]
        }
        
        return test_mappings.get(condition_key, ["Comprehensive evaluation"])
    
    def format_diagnosis_report(self, diagnosis_result: Dict[str, Any]) -> str:
        """Format diagnosis result into a comprehensive report"""
        if 'error' in diagnosis_result:
            return f"Error: {diagnosis_result['error']}"
        
        report = []
        report.append("=" * 60)
        report.append("        AI-POWERED HEALTH ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {diagnosis_result['timestamp']}")
        report.append("")
        
        # Patient information
        patient_info = diagnosis_result['patient_info']
        report.append("PATIENT INFORMATION")
        report.append("-" * 20)
        report.append(f"Age: {patient_info.get('age', 'Not provided')}")
        report.append(f"Gender: {patient_info.get('gender', 'Not provided')}")
        report.append(f"Existing Conditions: {', '.join(patient_info.get('existing_conditions', ['None']))}")
        report.append("")
        
        # Symptoms
        report.append("SYMPTOMS REPORTED")
        report.append("-" * 17)
        for symptom in diagnosis_result['symptoms_analyzed']:
            report.append(f"• {symptom.replace('_', ' ').title()}")
        report.append(f"Urgency Level: {diagnosis_result['urgency_level']}")
        report.append("")
        
        # Most likely conditions
        categorized = diagnosis_result['categorized_conditions']
        if categorized['most_likely']:
            report.append("MOST LIKELY CONDITION(S)")
            report.append("-" * 24)
            for i, condition in enumerate(categorized['most_likely'], 1):
                report.append(f"{i}. {condition['condition_name']}")
                report.append(f"   Likelihood: {condition['likelihood_tier']}")
                report.append(f"   Why: {condition['description']}")
                report.append(f"   Confirm With: {', '.join(condition['confirmation_tests'])}")
                report.append("")
        
        # Possibly present conditions
        if categorized['possibly_present']:
            report.append("POSSIBLY PRESENT CONDITIONS")
            report.append("-" * 27)
            for condition in categorized['possibly_present']:
                report.append(f"• {condition['condition_name']} ({condition['likelihood_tier']})")
            report.append("")
        
        # Ruled out conditions
        if categorized['ruled_out']:
            report.append("CONDITIONS UNLIKELY")
            report.append("-" * 19)
            for condition in categorized['ruled_out']:
                report.append(f"• {condition['condition_name']}")
            report.append("")
        
        # AI explanation
        explanations = diagnosis_result['explanations']
        if explanations.get('main'):
            report.append("AI EXPLANATION")
            report.append("-" * 14)
            report.append(explanations['main'])
            if explanations.get('lab'):
                report.append("")
                report.append(explanations['lab'])
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 15)
        for i, rec in enumerate(diagnosis_result['recommendations'], 1):
            report.append(f"{i}. {rec}")
        report.append("")
        
        # Disclaimer
        report.append("DISCLAIMER")
        report.append("-" * 10)
        report.append("This tool provides screening insights and should not replace medical evaluation.")
        report.append("Always consult with healthcare professionals for proper diagnosis and treatment.")
        
        return "\n".join(report)