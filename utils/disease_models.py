"""
Disease Models
Utility classes and functions for disease prediction models and algorithms.
Contains lightweight ML models and rule-based classifiers for disease detection.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os

class DiseaseModels:
    """
    Collection of disease prediction models and algorithms.
    Includes both ML models and rule-based classifiers.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("disease_models")
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize disease prediction models."""
        try:
            self.logger.info("Initializing disease prediction models")
            
            # Initialize rule-based models
            self._init_rule_based_models()
            
            # Try to load pre-trained models if available
            self._load_pretrained_models()
            
            # Initialize lightweight ML models if no pre-trained models exist
            if not self.models:
                self._init_lightweight_models()
            
            self.logger.info("Disease models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize disease models: {str(e)}")
            # Fall back to rule-based models only
            self._init_rule_based_models()
    
    def _init_rule_based_models(self):
        """Initialize rule-based disease prediction models."""
        self.rule_based_classifiers = {
            'diabetes': DiabetesRuleClassifier(),
            'heart_disease': HeartDiseaseRuleClassifier(),
            'hypertension': HypertensionRuleClassifier(),
            'stress': StressRuleClassifier(),
            'respiratory': RespiratoryRuleClassifier()
        }
    
    def _load_pretrained_models(self):
        """Load pre-trained models if available."""
        model_path = "models"
        if not os.path.exists(model_path):
            return
        
        try:
            # Look for saved models
            model_files = [f for f in os.listdir(model_path) if f.endswith('.joblib')]
            
            for model_file in model_files:
                model_name = model_file.replace('.joblib', '')
                try:
                    model = joblib.load(os.path.join(model_path, model_file))
                    self.models[model_name] = model
                    self.logger.info(f"Loaded pre-trained model: {model_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load model {model_name}: {str(e)}")
        
        except Exception as e:
            self.logger.warning(f"Error loading pre-trained models: {str(e)}")
    
    def _init_lightweight_models(self):
        """Initialize lightweight ML models with synthetic training data."""
        try:
            # Create lightweight models for each disease
            diseases = ['diabetes', 'heart_disease', 'hypertension']
            
            for disease in diseases:
                # Initialize a simple model with default parameters
                model = RandomForestClassifier(
                    n_estimators=10,
                    max_depth=5,
                    random_state=42
                )
                
                # Generate minimal synthetic training data
                X_train, y_train = self._generate_synthetic_training_data(disease)
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Create scaler
                scaler = StandardScaler()
                scaler.fit(X_train)
                
                self.models[disease] = model
                self.scalers[disease] = scaler
                
                self.logger.info(f"Initialized lightweight model for {disease}")
        
        except Exception as e:
            self.logger.warning(f"Failed to initialize ML models: {str(e)}")
    
    def _generate_synthetic_training_data(self, disease: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate minimal synthetic training data for model initialization."""
        np.random.seed(42)  # For reproducibility
        
        # Generate basic feature set (age, gender, symptoms, vitals)
        n_samples = 100
        n_features = 10
        
        X = np.random.rand(n_samples, n_features)
        
        # Create simple rules for label generation based on disease
        if disease == 'diabetes':
            # Higher probability if age > 0.5 (normalized) and glucose-like features high
            y = ((X[:, 0] > 0.5) & (X[:, 3] > 0.6)).astype(int)
        elif disease == 'heart_disease':
            # Higher probability if age > 0.6 and chest pain features
            y = ((X[:, 0] > 0.6) & (X[:, 2] > 0.5)).astype(int)
        elif disease == 'hypertension':
            # Higher probability if age > 0.4 and blood pressure features
            y = ((X[:, 0] > 0.4) & (X[:, 4] > 0.7)).astype(int)
        else:
            # Random labels for other diseases
            y = np.random.binomial(1, 0.3, n_samples)
        
        return X, y
    
    def predict_disease_probability(self, feature_vector: np.ndarray, disease: str) -> Dict[str, Any]:
        """
        Predict disease probability using available models.
        
        Args:
            feature_vector: Normalized feature vector
            disease: Disease name to predict
            
        Returns:
            Prediction results with probability and confidence
        """
        try:
            # Try ML model first if available
            if disease in self.models:
                return self._predict_with_ml_model(feature_vector, disease)
            
            # Fall back to rule-based classifier
            elif disease in self.rule_based_classifiers:
                return self._predict_with_rules(feature_vector, disease)
            
            else:
                return {
                    'probability': 0.0,
                    'confidence': 0.0,
                    'method': 'none',
                    'error': f'No model available for {disease}'
                }
        
        except Exception as e:
            self.logger.error(f"Prediction failed for {disease}: {str(e)}")
            return {
                'probability': 0.0,
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def _predict_with_ml_model(self, feature_vector: np.ndarray, disease: str) -> Dict[str, Any]:
        """Predict using ML model."""
        try:
            model = self.models[disease]
            scaler = self.scalers.get(disease)
            
            # Scale features if scaler is available
            if scaler:
                feature_vector_scaled = scaler.transform(feature_vector.reshape(1, -1))
            else:
                feature_vector_scaled = feature_vector.reshape(1, -1)
            
            # Get prediction probability
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feature_vector_scaled)[0]
                probability = proba[1] if len(proba) > 1 else proba[0]
            else:
                prediction = model.predict(feature_vector_scaled)[0]
                probability = float(prediction)
            
            # Calculate confidence based on decision function if available
            confidence = 0.8  # Default confidence for ML models
            if hasattr(model, 'decision_function'):
                decision_score = abs(model.decision_function(feature_vector_scaled)[0])
                confidence = min(0.9, 0.5 + decision_score * 0.1)
            
            return {
                'probability': float(probability),
                'confidence': float(confidence),
                'method': 'ml_model',
                'model_type': type(model).__name__
            }
        
        except Exception as e:
            self.logger.error(f"ML model prediction failed: {str(e)}")
            # Fall back to rule-based
            return self._predict_with_rules(feature_vector, disease)
    
    def _predict_with_rules(self, feature_vector: np.ndarray, disease: str) -> Dict[str, Any]:
        """Predict using rule-based classifier."""
        try:
            if disease not in self.rule_based_classifiers:
                return {
                    'probability': 0.0,
                    'confidence': 0.0,
                    'method': 'none',
                    'error': f'No rule-based classifier for {disease}'
                }
            
            classifier = self.rule_based_classifiers[disease]
            result = classifier.predict(feature_vector)
            
            return {
                'probability': result['probability'],
                'confidence': result['confidence'],
                'method': 'rule_based',
                'rules_applied': result.get('rules_applied', [])
            }
        
        except Exception as e:
            self.logger.error(f"Rule-based prediction failed: {str(e)}")
            return {
                'probability': 0.0,
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def get_feature_importance(self, disease: str) -> Dict[str, float]:
        """Get feature importance for a specific disease model."""
        try:
            if disease in self.models:
                model = self.models[disease]
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                    return dict(zip(feature_names, importances))
            
            # Return rule-based importance if available
            if disease in self.rule_based_classifiers:
                return self.rule_based_classifiers[disease].get_feature_importance()
            
            return {}
        
        except Exception as e:
            self.logger.error(f"Failed to get feature importance: {str(e)}")
            return {}


class BaseRuleClassifier:
    """Base class for rule-based disease classifiers."""
    
    def __init__(self, disease_name: str):
        self.disease_name = disease_name
        self.rules = []
        
    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Base prediction method to be overridden."""
        return {
            'probability': 0.0,
            'confidence': 0.0,
            'rules_applied': []
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for rule-based classifier."""
        return {}


class DiabetesRuleClassifier(BaseRuleClassifier):
    """Rule-based classifier for diabetes prediction."""
    
    def __init__(self):
        super().__init__("diabetes")
        
    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Predict diabetes probability using rules."""
        try:
            score = 0.0
            rules_applied = []
            
            # Extract features (assuming normalized 0-1 values)
            if len(feature_vector) >= 10:
                age_norm = feature_vector[0]
                gender_norm = feature_vector[1]
                risk_score = feature_vector[2]
                symptoms_severity = feature_vector[4]
                heart_rate_norm = feature_vector[6]
                
                # Age rule (higher risk with age)
                if age_norm > 0.5:  # Age > 45 (normalized)
                    score += 0.2
                    rules_applied.append("age_factor")
                
                # Risk score rule
                if risk_score > 0.3:
                    score += 0.3
                    rules_applied.append("existing_risk_factors")
                
                # Symptoms severity
                if symptoms_severity > 0.6:
                    score += 0.2
                    rules_applied.append("symptom_severity")
                
                # Combined factors
                if age_norm > 0.6 and risk_score > 0.4:
                    score += 0.1
                    rules_applied.append("combined_age_risk")
            
            probability = min(score, 0.95)
            confidence = 0.7 if len(rules_applied) > 1 else 0.5
            
            return {
                'probability': probability,
                'confidence': confidence,
                'rules_applied': rules_applied
            }
        
        except Exception as e:
            return {
                'probability': 0.0,
                'confidence': 0.0,
                'rules_applied': [],
                'error': str(e)
            }


class HeartDiseaseRuleClassifier(BaseRuleClassifier):
    """Rule-based classifier for heart disease prediction."""
    
    def __init__(self):
        super().__init__("heart_disease")
        
    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Predict heart disease probability using rules."""
        try:
            score = 0.0
            rules_applied = []
            
            if len(feature_vector) >= 10:
                age_norm = feature_vector[0]
                gender_norm = feature_vector[1]
                risk_score = feature_vector[2]
                symptoms_severity = feature_vector[4]
                heart_rate_norm = feature_vector[6]
                
                # Age rule
                if age_norm > 0.65:  # Age > 65
                    score += 0.25
                    rules_applied.append("advanced_age")
                elif age_norm > 0.45:  # Age > 45
                    score += 0.15
                    rules_applied.append("middle_age")
                
                # Gender rule (males slightly higher risk)
                if gender_norm == 1/3:  # Normalized male
                    score += 0.05
                    rules_applied.append("male_gender")
                
                # Risk factors
                if risk_score > 0.4:
                    score += 0.3
                    rules_applied.append("cardiac_risk_factors")
                
                # Abnormal heart rate
                if heart_rate_norm > 0.8 or heart_rate_norm < 0.2:
                    score += 0.15
                    rules_applied.append("abnormal_heart_rate")
                
                # Severe symptoms
                if symptoms_severity > 0.7:
                    score += 0.2
                    rules_applied.append("severe_symptoms")
            
            probability = min(score, 0.95)
            confidence = 0.8 if len(rules_applied) > 2 else 0.6
            
            return {
                'probability': probability,
                'confidence': confidence,
                'rules_applied': rules_applied
            }
        
        except Exception as e:
            return {
                'probability': 0.0,
                'confidence': 0.0,
                'rules_applied': [],
                'error': str(e)
            }


class HypertensionRuleClassifier(BaseRuleClassifier):
    """Rule-based classifier for hypertension prediction."""
    
    def __init__(self):
        super().__init__("hypertension")
        
    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Predict hypertension probability using rules."""
        try:
            score = 0.0
            rules_applied = []
            
            if len(feature_vector) >= 10:
                age_norm = feature_vector[0]
                risk_score = feature_vector[2]
                symptoms_severity = feature_vector[4]
                
                # Age factor
                if age_norm > 0.4:  # Age > 40
                    score += 0.2
                    rules_applied.append("age_risk")
                
                # Existing risk factors
                if risk_score > 0.3:
                    score += 0.4
                    rules_applied.append("hypertension_risk_factors")
                
                # Symptom indicators
                if symptoms_severity > 0.5:
                    score += 0.15
                    rules_applied.append("related_symptoms")
            
            probability = min(score, 0.95)
            confidence = 0.7 if len(rules_applied) > 1 else 0.5
            
            return {
                'probability': probability,
                'confidence': confidence,
                'rules_applied': rules_applied
            }
        
        except Exception as e:
            return {
                'probability': 0.0,
                'confidence': 0.0,
                'rules_applied': [],
                'error': str(e)
            }


class StressRuleClassifier(BaseRuleClassifier):
    """Rule-based classifier for stress/anxiety prediction."""
    
    def __init__(self):
        super().__init__("stress")
        
    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Predict stress/anxiety probability using rules."""
        try:
            score = 0.0
            rules_applied = []
            
            if len(feature_vector) >= 10:
                symptoms_severity = feature_vector[4]
                urgency_score = feature_vector[5]
                heart_rate_norm = feature_vector[6]
                sleep_norm = feature_vector[8]
                
                # High symptom severity
                if symptoms_severity > 0.6:
                    score += 0.3
                    rules_applied.append("high_symptom_severity")
                
                # Urgency indicators
                if urgency_score > 0.6:
                    score += 0.2
                    rules_applied.append("symptom_urgency")
                
                # Elevated heart rate
                if heart_rate_norm > 0.7:
                    score += 0.15
                    rules_applied.append("elevated_heart_rate")
                
                # Sleep disturbances
                if sleep_norm < 0.3 or sleep_norm > 0.8:
                    score += 0.2
                    rules_applied.append("sleep_disturbance")
                
                # Combined stress indicators
                if symptoms_severity > 0.5 and heart_rate_norm > 0.6:
                    score += 0.1
                    rules_applied.append("combined_stress_indicators")
            
            probability = min(score, 0.95)
            confidence = 0.7 if len(rules_applied) > 2 else 0.6
            
            return {
                'probability': probability,
                'confidence': confidence,
                'rules_applied': rules_applied
            }
        
        except Exception as e:
            return {
                'probability': 0.0,
                'confidence': 0.0,
                'rules_applied': [],
                'error': str(e)
            }


class RespiratoryRuleClassifier(BaseRuleClassifier):
    """Rule-based classifier for respiratory condition prediction."""
    
    def __init__(self):
        super().__init__("respiratory")
        
    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Predict respiratory condition probability using rules."""
        try:
            score = 0.0
            rules_applied = []
            
            if len(feature_vector) >= 10:
                risk_score = feature_vector[2]
                symptoms_severity = feature_vector[4]
                spo2_norm = feature_vector[7]
                
                # Existing respiratory conditions
                if risk_score > 0.3:
                    score += 0.3
                    rules_applied.append("existing_respiratory_risk")
                
                # Respiratory symptoms
                if symptoms_severity > 0.5:
                    score += 0.25
                    rules_applied.append("respiratory_symptoms")
                
                # Low oxygen saturation
                if spo2_norm < 0.5:  # Below normal
                    score += 0.4
                    rules_applied.append("low_oxygen_saturation")
                
                # Combined respiratory indicators
                if symptoms_severity > 0.6 and spo2_norm < 0.6:
                    score += 0.15
                    rules_applied.append("combined_respiratory_distress")
            
            probability = min(score, 0.95)
            confidence = 0.8 if len(rules_applied) > 1 else 0.6
            
            return {
                'probability': probability,
                'confidence': confidence,
                'rules_applied': rules_applied
            }
        
        except Exception as e:
            return {
                'probability': 0.0,
                'confidence': 0.0,
                'rules_applied': [],
                'error': str(e)
            }
