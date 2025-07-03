"""
Lab Report Analyzer
Extracts and analyzes lab values from medical reports to adjust risk scores.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

class LabReportAnalyzer:
    """
    Analyzes lab reports to extract values and adjust disease risk scores.
    Supports common lab tests like blood glucose, cholesterol, CBC, etc.
    """
    
    def __init__(self):
        # Define normal ranges for common lab tests
        self.normal_ranges = {
            'glucose': {
                'fasting': {'min': 70, 'max': 100, 'unit': 'mg/dL'},
                'random': {'min': 70, 'max': 140, 'unit': 'mg/dL'},
                'hba1c': {'min': 4.0, 'max': 5.6, 'unit': '%'}
            },
            'cholesterol': {
                'total': {'min': 0, 'max': 200, 'unit': 'mg/dL'},
                'ldl': {'min': 0, 'max': 100, 'unit': 'mg/dL'},
                'hdl': {'min': 40, 'max': 999, 'unit': 'mg/dL'},  # Higher is better
                'triglycerides': {'min': 0, 'max': 150, 'unit': 'mg/dL'}
            },
            'blood_pressure': {
                'systolic': {'min': 90, 'max': 120, 'unit': 'mmHg'},
                'diastolic': {'min': 60, 'max': 80, 'unit': 'mmHg'}
            },
            'kidney_function': {
                'creatinine': {'min': 0.6, 'max': 1.2, 'unit': 'mg/dL'},
                'bun': {'min': 7, 'max': 20, 'unit': 'mg/dL'},
                'egfr': {'min': 90, 'max': 999, 'unit': 'mL/min/1.73m²'}
            },
            'liver_function': {
                'alt': {'min': 7, 'max': 40, 'unit': 'U/L'},
                'ast': {'min': 10, 'max': 40, 'unit': 'U/L'},
                'bilirubin': {'min': 0.3, 'max': 1.2, 'unit': 'mg/dL'}
            },
            'cbc': {
                'hemoglobin': {'min': 12.0, 'max': 15.5, 'unit': 'g/dL'},
                'hematocrit': {'min': 36, 'max': 46, 'unit': '%'},
                'white_blood_cells': {'min': 4.5, 'max': 11.0, 'unit': 'K/uL'},
                'platelets': {'min': 150, 'max': 450, 'unit': 'K/uL'}
            },
            'thyroid': {
                'tsh': {'min': 0.4, 'max': 4.0, 'unit': 'mIU/L'},
                'free_t4': {'min': 0.9, 'max': 1.7, 'unit': 'ng/dL'}
            },
            'inflammatory': {
                'crp': {'min': 0, 'max': 3.0, 'unit': 'mg/L'},
                'esr': {'min': 0, 'max': 30, 'unit': 'mm/hr'}
            }
        }
        
        # Lab test patterns for extraction
        self.lab_patterns = {
            'glucose': [
                r'glucose[:\s]*(\d+\.?\d*)\s*mg/dl',
                r'blood glucose[:\s]*(\d+\.?\d*)',
                r'fasting glucose[:\s]*(\d+\.?\d*)',
                r'random glucose[:\s]*(\d+\.?\d*)',
                r'hba1c[:\s]*(\d+\.?\d*)',
                r'hemoglobin a1c[:\s]*(\d+\.?\d*)'
            ],
            'cholesterol': [
                r'total cholesterol[:\s]*(\d+\.?\d*)',
                r'ldl[:\s]*(\d+\.?\d*)',
                r'hdl[:\s]*(\d+\.?\d*)',
                r'triglycerides[:\s]*(\d+\.?\d*)'
            ],
            'blood_pressure': [
                r'blood pressure[:\s]*(\d+)/(\d+)',
                r'bp[:\s]*(\d+)/(\d+)',
                r'systolic[:\s]*(\d+)',
                r'diastolic[:\s]*(\d+)'
            ],
            'kidney_function': [
                r'creatinine[:\s]*(\d+\.?\d*)',
                r'bun[:\s]*(\d+\.?\d*)',
                r'egfr[:\s]*(\d+\.?\d*)'
            ],
            'liver_function': [
                r'alt[:\s]*(\d+\.?\d*)',
                r'ast[:\s]*(\d+\.?\d*)',
                r'bilirubin[:\s]*(\d+\.?\d*)'
            ],
            'cbc': [
                r'hemoglobin[:\s]*(\d+\.?\d*)',
                r'hematocrit[:\s]*(\d+\.?\d*)',
                r'white blood cells[:\s]*(\d+\.?\d*)',
                r'wbc[:\s]*(\d+\.?\d*)',
                r'platelets[:\s]*(\d+\.?\d*)'
            ],
            'thyroid': [
                r'tsh[:\s]*(\d+\.?\d*)',
                r'free t4[:\s]*(\d+\.?\d*)',
                r'ft4[:\s]*(\d+\.?\d*)'
            ],
            'inflammatory': [
                r'crp[:\s]*(\d+\.?\d*)',
                r'c-reactive protein[:\s]*(\d+\.?\d*)',
                r'esr[:\s]*(\d+\.?\d*)'
            ]
        }
        
        # Disease risk adjustments based on lab values
        self.risk_adjustments = {
            'diabetes': {
                'glucose': {'weight': 0.4, 'threshold_multiplier': 1.5},
                'hba1c': {'weight': 0.3, 'threshold_multiplier': 1.8},
                'cholesterol': {'weight': 0.2, 'threshold_multiplier': 1.3},
                'blood_pressure': {'weight': 0.1, 'threshold_multiplier': 1.2}
            },
            'heart_disease': {
                'cholesterol': {'weight': 0.3, 'threshold_multiplier': 1.6},
                'blood_pressure': {'weight': 0.3, 'threshold_multiplier': 1.7},
                'glucose': {'weight': 0.2, 'threshold_multiplier': 1.4},
                'inflammatory': {'weight': 0.2, 'threshold_multiplier': 1.5}
            },
            'kidney_disease': {
                'kidney_function': {'weight': 0.5, 'threshold_multiplier': 2.0},
                'blood_pressure': {'weight': 0.3, 'threshold_multiplier': 1.5},
                'glucose': {'weight': 0.2, 'threshold_multiplier': 1.3}
            },
            'liver_disease': {
                'liver_function': {'weight': 0.6, 'threshold_multiplier': 2.0},
                'inflammatory': {'weight': 0.2, 'threshold_multiplier': 1.4},
                'cholesterol': {'weight': 0.2, 'threshold_multiplier': 1.3}
            }
        }
    
    def analyze_lab_report(self, report_text: str) -> Dict[str, Any]:
        """
        Analyze lab report text and extract values.
        
        Args:
            report_text: Text content of the lab report
            
        Returns:
            Dictionary containing extracted lab values and analysis
        """
        # Extract lab values
        extracted_values = self._extract_lab_values(report_text)
        
        # Analyze abnormal values
        abnormal_analysis = self._analyze_abnormal_values(extracted_values)
        
        # Calculate risk adjustments
        risk_adjustments = self._calculate_risk_adjustments(extracted_values)
        
        # Generate lab insights
        lab_insights = self._generate_lab_insights(extracted_values, abnormal_analysis)
        
        return {
            'extracted_values': extracted_values,
            'abnormal_analysis': abnormal_analysis,
            'risk_adjustments': risk_adjustments,
            'lab_insights': lab_insights,
            'quality_score': self._calculate_extraction_quality(extracted_values)
        }
    
    def _extract_lab_values(self, text: str) -> Dict[str, Dict[str, float]]:
        """Extract lab values from text using regex patterns."""
        text_lower = text.lower()
        extracted = {}
        
        for category, patterns in self.lab_patterns.items():
            category_values = {}
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            # Handle blood pressure format (systolic/diastolic)
                            if category == 'blood_pressure':
                                category_values['systolic'] = float(match[0])
                                category_values['diastolic'] = float(match[1])
                        else:
                            # Determine test type based on pattern
                            test_type = self._determine_test_type(pattern, category)
                            try:
                                category_values[test_type] = float(match)
                            except ValueError:
                                continue
            
            if category_values:
                extracted[category] = category_values
        
        return extracted
    
    def _determine_test_type(self, pattern: str, category: str) -> str:
        """Determine the specific test type from the pattern."""
        pattern_lower = pattern.lower()
        
        if 'hba1c' in pattern_lower or 'hemoglobin a1c' in pattern_lower:
            return 'hba1c'
        elif 'fasting' in pattern_lower:
            return 'fasting'
        elif 'random' in pattern_lower:
            return 'random'
        elif 'total cholesterol' in pattern_lower:
            return 'total'
        elif 'ldl' in pattern_lower:
            return 'ldl'
        elif 'hdl' in pattern_lower:
            return 'hdl'
        elif 'triglycerides' in pattern_lower:
            return 'triglycerides'
        elif 'systolic' in pattern_lower:
            return 'systolic'
        elif 'diastolic' in pattern_lower:
            return 'diastolic'
        elif 'creatinine' in pattern_lower:
            return 'creatinine'
        elif 'bun' in pattern_lower:
            return 'bun'
        elif 'egfr' in pattern_lower:
            return 'egfr'
        elif 'alt' in pattern_lower:
            return 'alt'
        elif 'ast' in pattern_lower:
            return 'ast'
        elif 'bilirubin' in pattern_lower:
            return 'bilirubin'
        elif 'hemoglobin' in pattern_lower:
            return 'hemoglobin'
        elif 'hematocrit' in pattern_lower:
            return 'hematocrit'
        elif 'white blood cells' in pattern_lower or 'wbc' in pattern_lower:
            return 'white_blood_cells'
        elif 'platelets' in pattern_lower:
            return 'platelets'
        elif 'tsh' in pattern_lower:
            return 'tsh'
        elif 'free t4' in pattern_lower or 'ft4' in pattern_lower:
            return 'free_t4'
        elif 'crp' in pattern_lower or 'c-reactive protein' in pattern_lower:
            return 'crp'
        elif 'esr' in pattern_lower:
            return 'esr'
        else:
            return category
    
    def _analyze_abnormal_values(self, extracted_values: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze which values are abnormal based on normal ranges."""
        abnormal_values = {}
        severity_counts = {'mild': 0, 'moderate': 0, 'severe': 0}
        
        for category, values in extracted_values.items():
            if category not in self.normal_ranges:
                continue
                
            normal_range = self.normal_ranges[category]
            abnormal_in_category = {}
            
            for test_type, value in values.items():
                if test_type not in normal_range:
                    continue
                    
                range_info = normal_range[test_type]
                min_val, max_val = range_info['min'], range_info['max']
                
                if value < min_val or value > max_val:
                    deviation = self._calculate_deviation(value, min_val, max_val)
                    severity = self._determine_severity(deviation)
                    
                    abnormal_in_category[test_type] = {
                        'value': value,
                        'normal_range': f"{min_val}-{max_val} {range_info['unit']}",
                        'deviation': deviation,
                        'severity': severity,
                        'status': 'high' if value > max_val else 'low'
                    }
                    
                    severity_counts[severity] += 1
            
            if abnormal_in_category:
                abnormal_values[category] = abnormal_in_category
        
        return {
            'abnormal_values': abnormal_values,
            'severity_summary': severity_counts,
            'total_abnormal': sum(severity_counts.values())
        }
    
    def _calculate_deviation(self, value: float, min_val: float, max_val: float) -> float:
        """Calculate how much a value deviates from normal range."""
        if value < min_val:
            return (min_val - value) / min_val
        elif value > max_val:
            return (value - max_val) / max_val
        else:
            return 0.0
    
    def _determine_severity(self, deviation: float) -> str:
        """Determine severity based on deviation from normal range."""
        if deviation < 0.2:
            return 'mild'
        elif deviation < 0.5:
            return 'moderate'
        else:
            return 'severe'
    
    def _calculate_risk_adjustments(self, extracted_values: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate risk score adjustments based on lab values."""
        risk_adjustments = {}
        
        for disease, adjustment_config in self.risk_adjustments.items():
            total_adjustment = 0.0
            
            for lab_category, config in adjustment_config.items():
                if lab_category in extracted_values:
                    category_values = extracted_values[lab_category]
                    
                    # Calculate category-specific adjustment
                    category_adjustment = self._calculate_category_adjustment(
                        category_values, lab_category, config
                    )
                    
                    total_adjustment += category_adjustment * config['weight']
            
            risk_adjustments[disease] = min(total_adjustment, 2.0)  # Cap at 2x multiplier
        
        return risk_adjustments
    
    def _calculate_category_adjustment(self, values: Dict[str, float], category: str, config: Dict[str, float]) -> float:
        """Calculate adjustment for a specific lab category."""
        if category not in self.normal_ranges:
            return 1.0
            
        normal_range = self.normal_ranges[category]
        total_deviation = 0.0
        value_count = 0
        
        for test_type, value in values.items():
            if test_type in normal_range:
                range_info = normal_range[test_type]
                deviation = self._calculate_deviation(value, range_info['min'], range_info['max'])
                total_deviation += deviation
                value_count += 1
        
        if value_count == 0:
            return 1.0
            
        avg_deviation = total_deviation / value_count
        return 1.0 + (avg_deviation * config['threshold_multiplier'])
    
    def _generate_lab_insights(self, extracted_values: Dict[str, Dict[str, float]], abnormal_analysis: Dict[str, Any]) -> List[str]:
        """Generate human-readable insights from lab analysis."""
        insights = []
        
        # Summary of extracted values
        total_values = sum(len(values) for values in extracted_values.values())
        insights.append(f"Analyzed {total_values} lab values across {len(extracted_values)} test categories")
        
        # Abnormal values summary
        abnormal_count = abnormal_analysis['total_abnormal']
        if abnormal_count > 0:
            severity_summary = abnormal_analysis['severity_summary']
            insights.append(f"Found {abnormal_count} abnormal values: {severity_summary['severe']} severe, {severity_summary['moderate']} moderate, {severity_summary['mild']} mild")
        else:
            insights.append("All lab values appear to be within normal ranges")
        
        # Specific category insights
        for category, values in extracted_values.items():
            if category in abnormal_analysis['abnormal_values']:
                abnormal_in_category = abnormal_analysis['abnormal_values'][category]
                insights.append(f"{category.replace('_', ' ').title()} tests show {len(abnormal_in_category)} abnormal values")
        
        return insights
    
    def _calculate_extraction_quality(self, extracted_values: Dict[str, Dict[str, float]]) -> float:
        """Calculate quality score of lab value extraction."""
        total_values = sum(len(values) for values in extracted_values.values())
        
        if total_values == 0:
            return 0.0
        elif total_values < 3:
            return 0.3
        elif total_values < 6:
            return 0.6
        elif total_values < 10:
            return 0.8
        else:
            return 1.0
    
    def get_lab_based_recommendations(self, lab_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on lab analysis."""
        recommendations = []
        abnormal_values = lab_analysis['abnormal_analysis']['abnormal_values']
        
        # Glucose-related recommendations
        if 'glucose' in abnormal_values:
            glucose_abnormal = abnormal_values['glucose']
            if any(val['severity'] in ['moderate', 'severe'] for val in glucose_abnormal.values()):
                recommendations.append("Consider consulting with an endocrinologist about glucose control")
        
        # Cholesterol-related recommendations
        if 'cholesterol' in abnormal_values:
            cholesterol_abnormal = abnormal_values['cholesterol']
            if any(val['severity'] in ['moderate', 'severe'] for val in cholesterol_abnormal.values()):
                recommendations.append("Consider dietary modifications and discuss cholesterol management with your doctor")
        
        # Blood pressure recommendations
        if 'blood_pressure' in abnormal_values:
            bp_abnormal = abnormal_values['blood_pressure']
            if any(val['severity'] in ['moderate', 'severe'] for val in bp_abnormal.values()):
                recommendations.append("Blood pressure monitoring and cardiovascular risk assessment recommended")
        
        # Kidney function recommendations
        if 'kidney_function' in abnormal_values:
            kidney_abnormal = abnormal_values['kidney_function']
            if any(val['severity'] in ['moderate', 'severe'] for val in kidney_abnormal.values()):
                recommendations.append("Kidney function evaluation by a nephrologist may be warranted")
        
        # General recommendation if many abnormal values
        if lab_analysis['abnormal_analysis']['total_abnormal'] > 5:
            recommendations.append("Multiple abnormal values detected - comprehensive medical evaluation recommended")
        
        return recommendations