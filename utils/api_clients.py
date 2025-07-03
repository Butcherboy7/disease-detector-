"""
API Clients
Client classes for interacting with external APIs like OpenAI, HuggingFace, etc.
Handles authentication, rate limiting, and error handling for external services.
"""

import os
import json
import time
import requests
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI

class OpenAIClient:
    """
    Client for OpenAI API interactions.
    Handles GPT model requests for medical explanations and text generation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("openai_client")
        self.api_key = os.getenv("OPENAI_API_KEY", "default_openai_key")
        self.client = OpenAI(api_key=self.api_key)
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
    def _enforce_rate_limit(self):
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def generate_medical_explanation(self, prompt: str) -> str:
        """
        Generate medical explanation using OpenAI GPT.
        
        Args:
            prompt: The prompt for explanation generation
            
        Returns:
            Generated explanation text
        """
        try:
            self._enforce_rate_limit()
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant that provides clear, "
                        + "compassionate explanations of health analysis results. "
                        + "Use simple language, avoid medical jargon, and be reassuring "
                        + "while being accurate. Always emphasize that AI analysis is a "
                        + "screening tool and not a medical diagnosis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content
            self.logger.info("Successfully generated medical explanation")
            return explanation
            
        except Exception as e:
            self.logger.error(f"OpenAI explanation generation failed: {str(e)}")
            return self._generate_fallback_explanation()
    
    def analyze_symptoms_text(self, symptoms_text: str) -> Dict[str, Any]:
        """
        Analyze symptoms text for medical insights.
        
        Args:
            symptoms_text: Patient's symptom description
            
        Returns:
            Analysis results in JSON format
        """
        try:
            self._enforce_rate_limit()
            
            prompt = f"""
            Analyze the following patient symptoms and provide a structured analysis.
            Return your response as JSON with the following structure:
            {{
                "severity_score": number (1-5),
                "urgency_level": "low|medium|high",
                "key_symptoms": ["symptom1", "symptom2"],
                "potential_concerns": ["concern1", "concern2"],
                "recommendation": "immediate|routine|monitoring"
            }}
            
            Patient symptoms: "{symptoms_text}"
            """
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical AI that analyzes patient symptoms. "
                        + "Respond only with valid JSON in the specified format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=300,
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            self.logger.info("Successfully analyzed symptoms with OpenAI")
            return result
            
        except Exception as e:
            self.logger.error(f"OpenAI symptoms analysis failed: {str(e)}")
            return self._generate_fallback_symptoms_analysis(symptoms_text)
    
    def generate_health_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """
        Generate personalized health recommendations.
        
        Args:
            analysis_data: Complete analysis data
            
        Returns:
            List of health recommendations
        """
        try:
            self._enforce_rate_limit()
            
            # Extract key information for recommendations
            predictions = analysis_data.get('predictions', [])
            risk_level = analysis_data.get('overall_risk', {}).get('level', 'Unknown')
            patient_age = analysis_data.get('patient_info', {}).get('age', 'Unknown')
            
            prompt = f"""
            Based on the following health analysis, provide 5-7 specific, actionable health recommendations.
            
            Risk Level: {risk_level}
            Patient Age: {patient_age}
            Top Predictions: {[p.get('disease', '') for p in predictions[:3]]}
            
            Provide recommendations as a JSON array of strings:
            ["recommendation1", "recommendation2", ...]
            
            Focus on practical, actionable advice that a patient can follow.
            """
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You provide practical health recommendations. "
                        + "Respond with a JSON array of recommendation strings."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=400,
                temperature=0.6
            )
            
            result = json.loads(response.choices[0].message.content)
            recommendations = result.get('recommendations', [])
            
            self.logger.info("Successfully generated health recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"OpenAI recommendations generation failed: {str(e)}")
            return self._generate_fallback_recommendations(analysis_data)
    
    def _generate_fallback_explanation(self) -> str:
        """Generate fallback explanation when OpenAI is unavailable."""
        return """
        Based on your symptoms and health information, our AI analysis has identified several potential health considerations. 
        While this analysis provides useful insights, it's important to remember that this is a screening tool designed to help 
        guide your health decisions, not replace professional medical evaluation.
        
        We recommend discussing these results with your healthcare provider, who can provide a comprehensive evaluation 
        taking into account your complete medical history and perform any necessary examinations or tests.
        
        If you're experiencing severe or worsening symptoms, please seek medical attention promptly.
        """
    
    def _generate_fallback_symptoms_analysis(self, symptoms_text: str) -> Dict[str, Any]:
        """Generate fallback symptoms analysis."""
        # Simple keyword-based analysis
        severity_keywords = ['severe', 'intense', 'extreme', 'unbearable']
        urgency_keywords = ['sudden', 'acute', 'emergency', 'urgent']
        
        text_lower = symptoms_text.lower()
        
        severity_score = 3  # Default medium severity
        urgency_level = "medium"
        
        if any(keyword in text_lower for keyword in severity_keywords):
            severity_score = 5
            urgency_level = "high"
        elif any(keyword in text_lower for keyword in urgency_keywords):
            urgency_level = "high"
        
        return {
            "severity_score": severity_score,
            "urgency_level": urgency_level,
            "key_symptoms": ["symptom analysis unavailable"],
            "potential_concerns": ["requires professional evaluation"],
            "recommendation": "routine" if urgency_level == "low" else "immediate"
        }
    
    def _generate_fallback_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate fallback recommendations."""
        return [
            "Schedule an appointment with your healthcare provider",
            "Monitor your symptoms and note any changes",
            "Maintain a healthy diet and regular exercise routine",
            "Ensure adequate sleep (7-9 hours per night)",
            "Manage stress through relaxation techniques",
            "Stay hydrated and avoid smoking",
            "Keep a symptom diary to track patterns"
        ]


class HuggingFaceClient:
    """
    Client for HuggingFace API interactions.
    Handles model inference for medical text classification and analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("huggingface_client")
        self.api_key = os.getenv("HUGGINGFACE_API_KEY", "default_hf_key")
        self.base_url = "https://api-inference.huggingface.co/models"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.rate_limit_delay = 2.0  # seconds between requests
        self.last_request_time = 0
        
    def _enforce_rate_limit(self):
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def classify_medical_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Classify medical text using HuggingFace models.
        
        Args:
            text: Medical text to classify
            
        Returns:
            List of classification results
        """
        try:
            self._enforce_rate_limit()
            
            # Use a general text classification model
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            url = f"{self.base_url}/{model_name}"
            
            payload = {"inputs": text}
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                results = response.json()
                self.logger.info("Successfully classified medical text with HuggingFace")
                return self._process_classification_results(results)
            else:
                self.logger.warning(f"HuggingFace API error: {response.status_code}")
                return self._generate_fallback_classification(text)
                
        except Exception as e:
            self.logger.error(f"HuggingFace classification failed: {str(e)}")
            return self._generate_fallback_classification(text)
    
    def analyze_medical_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical entities from text using NER models.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities
        """
        try:
            self._enforce_rate_limit()
            
            # Use a named entity recognition model
            model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            url = f"{self.base_url}/{model_name}"
            
            payload = {"inputs": text}
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                results = response.json()
                self.logger.info("Successfully extracted entities with HuggingFace")
                return self._process_entity_results(results)
            else:
                self.logger.warning(f"HuggingFace NER API error: {response.status_code}")
                return self._generate_fallback_entities(text)
                
        except Exception as e:
            self.logger.error(f"HuggingFace entity extraction failed: {str(e)}")
            return self._generate_fallback_entities(text)
    
    def _process_classification_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process classification results into medical context."""
        processed_results = []
        
        for result in results:
            label = result.get('label', '').upper()
            score = result.get('score', 0)
            
            # Map general sentiment to medical risk
            if label == 'NEGATIVE' and score > 0.7:
                processed_results.append({
                    'label': 'High Concern',
                    'score': score,
                    'category': 'medical_risk'
                })
            elif label == 'POSITIVE' and score > 0.7:
                processed_results.append({
                    'label': 'Low Concern',
                    'score': score,
                    'category': 'medical_risk'
                })
        
        return processed_results
    
    def _process_entity_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process entity extraction results."""
        processed_entities = []
        
        for entity in results:
            entity_type = entity.get('entity_group', entity.get('entity', 'MISC'))
            word = entity.get('word', '')
            score = entity.get('score', 0)
            
            if score > 0.5:  # Only include high-confidence entities
                processed_entities.append({
                    'type': entity_type,
                    'text': word,
                    'confidence': score
                })
        
        return processed_entities
    
    def _generate_fallback_classification(self, text: str) -> List[Dict[str, Any]]:
        """Generate fallback classification when API is unavailable."""
        # Simple keyword-based classification
        concern_keywords = ['pain', 'severe', 'acute', 'emergency', 'urgent', 'bleeding']
        normal_keywords = ['mild', 'slight', 'occasional', 'minor']
        
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in concern_keywords):
            return [{
                'label': 'High Concern',
                'score': 0.8,
                'category': 'fallback_classification'
            }]
        elif any(keyword in text_lower for keyword in normal_keywords):
            return [{
                'label': 'Low Concern',
                'score': 0.7,
                'category': 'fallback_classification'
            }]
        else:
            return [{
                'label': 'Medium Concern',
                'score': 0.6,
                'category': 'fallback_classification'
            }]
    
    def _generate_fallback_entities(self, text: str) -> List[Dict[str, Any]]:
        """Generate fallback entities when API is unavailable."""
        # Simple keyword extraction
        medical_keywords = [
            'chest', 'heart', 'blood', 'pressure', 'diabetes', 'pain',
            'headache', 'fever', 'cough', 'fatigue', 'nausea'
        ]
        
        entities = []
        text_lower = text.lower()
        
        for keyword in medical_keywords:
            if keyword in text_lower:
                entities.append({
                    'type': 'MEDICAL',
                    'text': keyword,
                    'confidence': 0.7
                })
        
        return entities


class HealthAPIClient:
    """
    Client for health-related APIs and data sources.
    Handles integration with medical databases and health information APIs.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("health_api_client")
        self.api_key = os.getenv("HEALTH_API_KEY", "default_health_key")
        
    def get_disease_information(self, disease_name: str) -> Dict[str, Any]:
        """
        Get information about a specific disease.
        
        Args:
            disease_name: Name of the disease
            
        Returns:
            Disease information dictionary
        """
        try:
            # In a real implementation, this would query a medical database
            # For now, return structured information based on common diseases
            return self._get_disease_info_fallback(disease_name)
            
        except Exception as e:
            self.logger.error(f"Health API query failed: {str(e)}")
            return self._get_disease_info_fallback(disease_name)
    
    def _get_disease_info_fallback(self, disease_name: str) -> Dict[str, Any]:
        """Fallback disease information."""
        disease_info = {
            'diabetes': {
                'name': 'Diabetes',
                'description': 'A group of metabolic disorders characterized by high blood sugar',
                'common_symptoms': ['increased thirst', 'frequent urination', 'fatigue', 'blurred vision'],
                'risk_factors': ['family history', 'obesity', 'sedentary lifestyle', 'age over 45'],
                'prevention': ['healthy diet', 'regular exercise', 'weight management'],
                'when_to_see_doctor': 'If experiencing classic symptoms or risk factors are present'
            },
            'heart_disease': {
                'name': 'Heart Disease',
                'description': 'Conditions that affect the heart and blood vessels',
                'common_symptoms': ['chest pain', 'shortness of breath', 'palpitations', 'fatigue'],
                'risk_factors': ['high blood pressure', 'high cholesterol', 'smoking', 'diabetes', 'family history'],
                'prevention': ['heart-healthy diet', 'regular exercise', 'no smoking', 'stress management'],
                'when_to_see_doctor': 'If experiencing chest pain or other cardiac symptoms'
            },
            'hypertension': {
                'name': 'Hypertension',
                'description': 'High blood pressure that can lead to serious health problems',
                'common_symptoms': ['headaches', 'dizziness', 'vision problems'],
                'risk_factors': ['age', 'family history', 'obesity', 'high sodium diet', 'stress'],
                'prevention': ['healthy diet', 'regular exercise', 'limit sodium', 'manage stress'],
                'when_to_see_doctor': 'For regular blood pressure monitoring and if symptoms occur'
            }
        }
        
        disease_key = disease_name.lower().replace(' ', '_').replace('/', '_')
        
        return disease_info.get(disease_key, {
            'name': disease_name,
            'description': 'Information not available',
            'common_symptoms': [],
            'risk_factors': [],
            'prevention': [],
            'when_to_see_doctor': 'Consult healthcare provider for evaluation'
        })
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate all API keys.
        
        Returns:
            Dictionary showing which API keys are valid
        """
        validation_results = {
            'openai': False,
            'huggingface': False,
            'health_api': False
        }
        
        # Check OpenAI key
        try:
            openai_key = os.getenv("OPENAI_API_KEY", "")
            if openai_key and openai_key != "default_openai_key":
                validation_results['openai'] = True
        except:
            pass
        
        # Check HuggingFace key
        try:
            hf_key = os.getenv("HUGGINGFACE_API_KEY", "")
            if hf_key and hf_key != "default_hf_key":
                validation_results['huggingface'] = True
        except:
            pass
        
        # Check Health API key
        try:
            health_key = os.getenv("HEALTH_API_KEY", "")
            if health_key and health_key != "default_health_key":
                validation_results['health_api'] = True
        except:
            pass
        
        return validation_results
