"""
Symptom Clustering Analysis
Provides intelligent clustering of symptoms to identify disease patterns and confidence scores.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import json

class SymptomClusterAnalyzer:
    """
    Analyzes symptoms to identify disease clusters and confidence scores.
    Uses predefined symptom patterns and clustering algorithms.
    """
    
    def __init__(self):
        self.disease_clusters = {
            'metabolic_syndrome': {
                'symptoms': [
                    'fatigue', 'weight_gain', 'excessive_thirst', 'frequent_urination',
                    'blurred_vision', 'slow_healing', 'increased_hunger', 'tingling',
                    'abdominal_weight', 'high_blood_pressure', 'insulin_resistance'
                ],
                'weight_factors': {
                    'fatigue': 0.8, 'weight_gain': 0.9, 'excessive_thirst': 0.95,
                    'frequent_urination': 0.9, 'blurred_vision': 0.85, 'slow_healing': 0.8,
                    'increased_hunger': 0.85, 'tingling': 0.7, 'abdominal_weight': 0.8,
                    'high_blood_pressure': 0.9, 'insulin_resistance': 0.95
                },
                'confidence_threshold': 0.6
            },
            'cardiovascular_syndrome': {
                'symptoms': [
                    'chest_pain', 'shortness_of_breath', 'palpitations', 'fatigue',
                    'dizziness', 'swelling', 'irregular_heartbeat', 'high_blood_pressure',
                    'chest_tightness', 'arm_pain', 'jaw_pain', 'nausea'
                ],
                'weight_factors': {
                    'chest_pain': 0.95, 'shortness_of_breath': 0.9, 'palpitations': 0.85,
                    'fatigue': 0.7, 'dizziness': 0.8, 'swelling': 0.85,
                    'irregular_heartbeat': 0.9, 'high_blood_pressure': 0.85,
                    'chest_tightness': 0.9, 'arm_pain': 0.8, 'jaw_pain': 0.8, 'nausea': 0.7
                },
                'confidence_threshold': 0.65
            },
            'respiratory_syndrome': {
                'symptoms': [
                    'cough', 'shortness_of_breath', 'wheezing', 'chest_tightness',
                    'sputum', 'fever', 'fatigue', 'difficulty_breathing',
                    'throat_irritation', 'nasal_congestion'
                ],
                'weight_factors': {
                    'cough': 0.9, 'shortness_of_breath': 0.95, 'wheezing': 0.9,
                    'chest_tightness': 0.85, 'sputum': 0.8, 'fever': 0.7,
                    'fatigue': 0.6, 'difficulty_breathing': 0.95,
                    'throat_irritation': 0.6, 'nasal_congestion': 0.5
                },
                'confidence_threshold': 0.7
            },
            'neurological_syndrome': {
                'symptoms': [
                    'headache', 'dizziness', 'confusion', 'memory_loss',
                    'numbness', 'tingling', 'weakness', 'tremors',
                    'seizures', 'vision_changes', 'speech_difficulty'
                ],
                'weight_factors': {
                    'headache': 0.7, 'dizziness': 0.8, 'confusion': 0.85,
                    'memory_loss': 0.8, 'numbness': 0.85, 'tingling': 0.8,
                    'weakness': 0.8, 'tremors': 0.9, 'seizures': 0.95,
                    'vision_changes': 0.85, 'speech_difficulty': 0.9
                },
                'confidence_threshold': 0.65
            },
            'gastrointestinal_syndrome': {
                'symptoms': [
                    'abdominal_pain', 'nausea', 'vomiting', 'diarrhea',
                    'constipation', 'bloating', 'loss_of_appetite',
                    'weight_loss', 'heartburn', 'blood_in_stool'
                ],
                'weight_factors': {
                    'abdominal_pain': 0.9, 'nausea': 0.8, 'vomiting': 0.85,
                    'diarrhea': 0.8, 'constipation': 0.7, 'bloating': 0.6,
                    'loss_of_appetite': 0.7, 'weight_loss': 0.8,
                    'heartburn': 0.6, 'blood_in_stool': 0.95
                },
                'confidence_threshold': 0.6
            },
            'mental_health_syndrome': {
                'symptoms': [
                    'anxiety', 'depression', 'insomnia', 'fatigue',
                    'irritability', 'mood_swings', 'panic_attacks',
                    'social_withdrawal', 'concentration_difficulty'
                ],
                'weight_factors': {
                    'anxiety': 0.9, 'depression': 0.9, 'insomnia': 0.8,
                    'fatigue': 0.7, 'irritability': 0.8, 'mood_swings': 0.85,
                    'panic_attacks': 0.95, 'social_withdrawal': 0.8,
                    'concentration_difficulty': 0.8
                },
                'confidence_threshold': 0.65
            },
            'iron_deficiency_anemia': {
                'symptoms': [
                    'fatigue', 'weakness', 'cold_hands', 'cold_feet', 'dizziness',
                    'palpitations', 'hair_thinning', 'brittle_nails', 'restless_legs',
                    'ice_cravings', 'pale_skin', 'shortness_of_breath', 'brain_fog'
                ],
                'weight_factors': {
                    'fatigue': 0.9, 'weakness': 0.85, 'cold_hands': 0.8, 'cold_feet': 0.8,
                    'dizziness': 0.8, 'palpitations': 0.75, 'hair_thinning': 0.85,
                    'brittle_nails': 0.9, 'restless_legs': 0.9, 'ice_cravings': 0.95,
                    'pale_skin': 0.9, 'shortness_of_breath': 0.8, 'brain_fog': 0.7
                },
                'confidence_threshold': 0.6
            },
            'hypothyroidism': {
                'symptoms': [
                    'fatigue', 'brain_fog', 'weight_gain', 'hair_loss', 'cold_intolerance',
                    'slow_pulse', 'constipation', 'dry_skin', 'memory_problems',
                    'depression', 'muscle_weakness', 'joint_pain', 'sleep_issues'
                ],
                'weight_factors': {
                    'fatigue': 0.85, 'brain_fog': 0.9, 'weight_gain': 0.9, 'hair_loss': 0.9,
                    'cold_intolerance': 0.95, 'slow_pulse': 0.85, 'constipation': 0.8,
                    'dry_skin': 0.8, 'memory_problems': 0.85, 'depression': 0.75,
                    'muscle_weakness': 0.8, 'joint_pain': 0.7, 'sleep_issues': 0.7
                },
                'confidence_threshold': 0.65
            },
            'vitamin_d_deficiency': {
                'symptoms': [
                    'fatigue', 'body_aches', 'muscle_weakness', 'bone_pain',
                    'poor_concentration', 'mood_changes', 'muscle_cramps',
                    'frequent_infections', 'joint_pain', 'sleep_problems'
                ],
                'weight_factors': {
                    'fatigue': 0.8, 'body_aches': 0.9, 'muscle_weakness': 0.9,
                    'bone_pain': 0.95, 'poor_concentration': 0.75, 'mood_changes': 0.8,
                    'muscle_cramps': 0.85, 'frequent_infections': 0.8, 'joint_pain': 0.85,
                    'sleep_problems': 0.7
                },
                'confidence_threshold': 0.6
            },
            'autonomic_dysfunction': {
                'symptoms': [
                    'dizziness_on_standing', 'palpitations', 'fatigue', 'shortness_of_breath',
                    'chest_pain', 'brain_fog', 'exercise_intolerance', 'rapid_heart_rate',
                    'nausea', 'sweating', 'headache', 'sleep_disturbances'
                ],
                'weight_factors': {
                    'dizziness_on_standing': 0.95, 'palpitations': 0.9, 'fatigue': 0.8,
                    'shortness_of_breath': 0.85, 'chest_pain': 0.8, 'brain_fog': 0.85,
                    'exercise_intolerance': 0.9, 'rapid_heart_rate': 0.9, 'nausea': 0.7,
                    'sweating': 0.75, 'headache': 0.7, 'sleep_disturbances': 0.75
                },
                'confidence_threshold': 0.7
            }
        }
        
        # Common symptom mappings for normalization
        self.symptom_mappings = {
            'tired': 'fatigue', 'exhausted': 'fatigue', 'weak': 'fatigue',
            'breathless': 'shortness_of_breath', 'sob': 'shortness_of_breath',
            'chest hurt': 'chest_pain', 'chest ache': 'chest_pain',
            'heart racing': 'palpitations', 'fast heart': 'palpitations',
            'dizzy': 'dizziness', 'lightheaded': 'dizziness',
            'sick': 'nausea', 'queasy': 'nausea',
            'throw up': 'vomiting', 'threw up': 'vomiting',
            'stomach pain': 'abdominal_pain', 'belly pain': 'abdominal_pain',
            'can\'t sleep': 'insomnia', 'sleepless': 'insomnia',
            'worried': 'anxiety', 'nervous': 'anxiety', 'stressed': 'anxiety',
            'sad': 'depression', 'down': 'depression', 'hopeless': 'depression',
            # New mappings for anemia
            'cold hands': 'cold_hands', 'cold feet': 'cold_feet',
            'hair falling out': 'hair_thinning', 'brittle nails': 'brittle_nails',
            'restless legs': 'restless_legs', 'craving ice': 'ice_cravings',
            'pale': 'pale_skin', 'brain fog': 'brain_fog',
            # New mappings for thyroid
            'cold all the time': 'cold_intolerance', 'slow heart': 'slow_pulse',
            'dry skin': 'dry_skin', 'memory issues': 'memory_problems',
            'hair loss': 'hair_loss', 'weight gain': 'weight_gain',
            # New mappings for vitamin D
            'body aches': 'body_aches', 'bone pain': 'bone_pain',
            'muscle cramps': 'muscle_cramps', 'poor concentration': 'poor_concentration',
            'mood swings': 'mood_changes', 'frequent colds': 'frequent_infections',
            # New mappings for autonomic/POTS
            'dizzy when standing': 'dizziness_on_standing',
            'can\'t exercise': 'exercise_intolerance', 'fast heart rate': 'rapid_heart_rate',
            'sweating': 'sweating', 'sleep problems': 'sleep_disturbances'
        }
    
    def analyze_symptom_clusters(self, symptoms_text: str, extracted_entities: List[Dict]) -> Dict[str, Any]:
        """
        Analyze symptoms to identify disease clusters and confidence scores.
        
        Args:
            symptoms_text: Raw symptom text from user
            extracted_entities: Medical entities extracted from preprocessing
            
        Returns:
            Dictionary containing cluster analysis results
        """
        # Extract and normalize symptoms
        normalized_symptoms = self._extract_and_normalize_symptoms(symptoms_text, extracted_entities)
        
        # Calculate cluster confidences
        cluster_confidences = self._calculate_cluster_confidences(normalized_symptoms)
        
        # Generate cluster insights
        cluster_insights = self._generate_cluster_insights(cluster_confidences, normalized_symptoms)
        
        return {
            'cluster_analysis': cluster_confidences,
            'top_clusters': sorted(cluster_confidences.items(), key=lambda x: x[1], reverse=True)[:3],
            'insights': cluster_insights,
            'normalized_symptoms': normalized_symptoms,
            'confidence_summary': self._generate_confidence_summary(cluster_confidences)
        }
    
    def _extract_and_normalize_symptoms(self, symptoms_text: str, extracted_entities: List[Dict]) -> List[str]:
        """Extract and normalize symptoms from text and entities."""
        symptoms = set()
        
        # Extract from entities
        for entity in extracted_entities:
            if entity.get('type') == 'symptom':
                symptoms.add(entity['text'].lower())
        
        # Extract from text using keyword matching
        text_lower = symptoms_text.lower()
        for symptom in self._get_all_known_symptoms():
            if symptom in text_lower:
                symptoms.add(symptom)
        
        # Apply symptom mappings
        normalized = set()
        for symptom in symptoms:
            normalized.add(self.symptom_mappings.get(symptom, symptom))
        
        return list(normalized)
    
    def _calculate_cluster_confidences(self, symptoms: List[str]) -> Dict[str, float]:
        """Calculate confidence scores for each disease cluster."""
        confidences = {}
        
        for cluster_name, cluster_data in self.disease_clusters.items():
            cluster_symptoms = cluster_data['symptoms']
            weight_factors = cluster_data['weight_factors']
            
            # Calculate weighted confidence
            total_weight = 0
            matched_weight = 0
            
            for symptom in symptoms:
                if symptom in cluster_symptoms:
                    weight = weight_factors.get(symptom, 0.5)
                    matched_weight += weight
                    total_weight += weight
            
            # Add penalty for missing critical symptoms
            for critical_symptom in cluster_symptoms:
                if critical_symptom not in symptoms:
                    weight = weight_factors.get(critical_symptom, 0.5)
                    total_weight += weight
            
            # Calculate confidence percentage
            if total_weight > 0:
                confidence = (matched_weight / total_weight) * 100
                confidences[cluster_name] = min(confidence, 100)
            else:
                confidences[cluster_name] = 0
        
        return confidences
    
    def _generate_cluster_insights(self, cluster_confidences: Dict[str, float], symptoms: List[str]) -> List[str]:
        """Generate human-readable insights about cluster analysis."""
        insights = []
        
        # Sort clusters by confidence
        sorted_clusters = sorted(cluster_confidences.items(), key=lambda x: x[1], reverse=True)
        
        for cluster_name, confidence in sorted_clusters[:3]:
            if confidence > 40:  # Only show meaningful confidences
                cluster_display = self._format_cluster_name(cluster_name)
                insights.append(f"Clustered symptoms point {confidence:.0f}% toward {cluster_display}")
        
        # Add symptom pattern insights
        if len(symptoms) >= 3:
            insights.append(f"Analyzed {len(symptoms)} symptom patterns across multiple disease categories")
        
        return insights
    
    def _generate_confidence_summary(self, cluster_confidences: Dict[str, float]) -> str:
        """Generate a summary of the confidence analysis."""
        if not cluster_confidences:
            return "No symptom clusters identified"
            
        max_confidence = max(cluster_confidences.values())
        max_cluster = max(cluster_confidences.items(), key=lambda x: x[1])[0]
        
        if max_confidence > 70:
            return f"Strong indication of {self._format_cluster_name(str(max_cluster))} ({max_confidence:.0f}% confidence)"
        elif max_confidence > 50:
            return f"Moderate indication of {self._format_cluster_name(str(max_cluster))} ({max_confidence:.0f}% confidence)"
        elif max_confidence > 30:
            return f"Possible indication of {self._format_cluster_name(str(max_cluster))} ({max_confidence:.0f}% confidence)"
        else:
            return "Symptoms do not strongly cluster into specific disease patterns"
    
    def _format_cluster_name(self, cluster_name: str) -> str:
        """Format cluster name for display."""
        return cluster_name.replace('_', ' ').title()
    
    def _get_all_known_symptoms(self) -> List[str]:
        """Get all known symptoms from all clusters."""
        all_symptoms = set()
        for cluster_data in self.disease_clusters.values():
            all_symptoms.update(cluster_data['symptoms'])
        return list(all_symptoms)
    
    def get_follow_up_questions(self, cluster_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate follow-up questions based on cluster analysis to improve accuracy.
        
        Args:
            cluster_analysis: Result from analyze_symptom_clusters
            
        Returns:
            List of follow-up questions with context
        """
        questions = []
        top_clusters = cluster_analysis['top_clusters']
        
        if not top_clusters:
            return []
        
        # Get top cluster
        top_cluster_name, top_confidence = top_clusters[0]
        
        # Generate questions based on top cluster
        if top_cluster_name == 'metabolic_syndrome':
            questions.extend([
                {
                    "question": "Have you noticed increased thirst or frequent urination recently?",
                    "context": "These are key indicators of metabolic dysfunction",
                    "type": "yes_no"
                },
                {
                    "question": "On a scale of 1-10, how would you rate your energy levels lately?",
                    "context": "Fatigue patterns help assess metabolic health",
                    "type": "scale"
                },
                {
                    "question": "Have you experienced any unexplained weight changes in the past 6 months?",
                    "context": "Weight fluctuations can indicate metabolic issues",
                    "type": "descriptive"
                }
            ])
        elif top_cluster_name == 'cardiovascular_syndrome':
            questions.extend([
                {
                    "question": "Do you experience chest discomfort during physical activity?",
                    "context": "Exercise-related symptoms are important for heart health assessment",
                    "type": "yes_no"
                },
                {
                    "question": "How often do you feel short of breath during daily activities?",
                    "context": "Breathing difficulties can indicate cardiovascular issues",
                    "type": "frequency"
                },
                {
                    "question": "Have you noticed any swelling in your legs, ankles, or feet?",
                    "context": "Fluid retention can be a sign of heart problems",
                    "type": "yes_no"
                }
            ])
        elif top_cluster_name == 'respiratory_syndrome':
            questions.extend([
                {
                    "question": "Is your cough dry or do you produce mucus/phlegm?",
                    "context": "Cough characteristics help identify respiratory conditions",
                    "type": "choice"
                },
                {
                    "question": "Do you wheeze or hear whistling sounds when breathing?",
                    "context": "Wheezing indicates airway narrowing",
                    "type": "yes_no"
                },
                {
                    "question": "How long have you been experiencing these respiratory symptoms?",
                    "context": "Duration helps distinguish between acute and chronic conditions",
                    "type": "duration"
                }
            ])
        elif top_cluster_name == 'iron_deficiency_anemia':
            questions.extend([
                {
                    "question": "Do you have unusually cold hands or feet, even in warm weather?",
                    "context": "Cold extremities are common signs of anemia",
                    "type": "yes_no"
                },
                {
                    "question": "Have you noticed any unusual cravings, especially for ice or starch?",
                    "context": "Unusual cravings can indicate iron deficiency",
                    "type": "yes_no"
                },
                {
                    "question": "Do you experience restless legs, especially at night?",
                    "context": "Restless leg syndrome is often linked to iron deficiency",
                    "type": "yes_no"
                }
            ])
        elif top_cluster_name == 'hypothyroidism':
            questions.extend([
                {
                    "question": "Do you feel cold all the time, even when others are comfortable?",
                    "context": "Cold intolerance is a key sign of hypothyroidism",
                    "type": "yes_no"
                },
                {
                    "question": "Have you experienced unexplained weight gain despite no changes in diet?",
                    "context": "Weight gain can indicate slowed metabolism from thyroid issues",
                    "type": "yes_no"
                },
                {
                    "question": "Have you noticed your hair becoming thinner or more brittle?",
                    "context": "Hair changes are common in thyroid disorders",
                    "type": "yes_no"
                }
            ])
        elif top_cluster_name == 'vitamin_d_deficiency':
            questions.extend([
                {
                    "question": "Do you experience muscle aches or bone pain, especially in your back?",
                    "context": "Bone and muscle pain are classic signs of vitamin D deficiency",
                    "type": "yes_no"
                },
                {
                    "question": "How much time do you spend outdoors in sunlight each week?",
                    "context": "Limited sun exposure increases vitamin D deficiency risk",
                    "type": "descriptive"
                },
                {
                    "question": "Do you get sick more often than you used to?",
                    "context": "Frequent infections can indicate immune system weakness from vitamin D deficiency",
                    "type": "yes_no"
                }
            ])
        elif top_cluster_name == 'autonomic_dysfunction':
            questions.extend([
                {
                    "question": "Do you feel dizzy or lightheaded when you stand up quickly?",
                    "context": "Orthostatic intolerance is a key sign of autonomic dysfunction",
                    "type": "yes_no"
                },
                {
                    "question": "Do you have difficulty with exercise or feel exhausted after minimal activity?",
                    "context": "Exercise intolerance can indicate autonomic nervous system issues",
                    "type": "yes_no"
                },
                {
                    "question": "Does your heart rate increase significantly when you change positions?",
                    "context": "Postural heart rate changes suggest possible POTS or autonomic issues",
                    "type": "yes_no"
                }
            ])
        
        # Add general questions if confidence is low
        if top_confidence < 60:
            questions.append({
                "question": "Are there any other symptoms you haven't mentioned that you've noticed recently?",
                "context": "Additional symptoms can help improve diagnostic accuracy",
                "type": "descriptive"
            })
        
        return questions[:3]  # Return max 3 questions