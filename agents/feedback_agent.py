"""
Feedback Agent
Responsible for collecting and processing user feedback on prediction accuracy
and overall system performance.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List
from .base_agent import BaseAgent

class FeedbackAgent(BaseAgent):
    """
    Agent responsible for collecting user feedback and improving system performance.
    Handles feedback collection, analysis, and storage for system improvement.
    """
    
    def __init__(self):
        super().__init__("FeedbackAgent")
        self.feedback_storage = []  # In production, this would be a database
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user feedback on predictions and system performance.
        
        Args:
            data: Feedback data from user
            
        Returns:
            Processed feedback confirmation and insights
        """
        try:
            self.log_processing_step("Starting feedback processing")
            
            # Validate feedback data
            if not self._validate_feedback_data(data):
                return self.handle_error(
                    ValueError("Invalid feedback data"),
                    "feedback validation"
                )
            
            # Process different types of feedback
            processed_feedback = self._process_feedback(data)
            
            # Store feedback for analysis
            self._store_feedback(processed_feedback)
            
            # Generate feedback insights
            insights = self._generate_feedback_insights(processed_feedback)
            
            # Generate recommendations for improvement
            improvement_suggestions = self._generate_improvement_suggestions(processed_feedback)
            
            self.log_processing_step("Feedback processing completed successfully")
            
            return self.create_success_response({
                'feedback_id': processed_feedback['feedback_id'],
                'feedback_summary': processed_feedback['summary'],
                'insights': insights,
                'improvement_suggestions': improvement_suggestions,
                'thank_you_message': self._generate_thank_you_message(processed_feedback)
            })
            
        except Exception as e:
            return self.handle_error(e, "feedback processing")
    
    def _validate_feedback_data(self, data: Dict[str, Any]) -> bool:
        """Validate incoming feedback data."""
        required_fields = ['feedback_type']
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required feedback field: {field}")
                return False
        
        valid_feedback_types = ['prediction_accuracy', 'system_usability', 'general', 'suggestion']
        if data['feedback_type'] not in valid_feedback_types:
            self.logger.error(f"Invalid feedback type: {data['feedback_type']}")
            return False
        
        return True
    
    def _process_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure feedback data."""
        self.log_processing_step("Processing feedback data")
        
        feedback_id = f"feedback_{int(time.time())}_{hash(str(data)) % 10000}"
        
        processed_feedback = {
            'feedback_id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'feedback_type': data['feedback_type'],
            'user_session_id': data.get('user_session_id', 'unknown'),
            'original_data': data
        }
        
        # Process based on feedback type
        if data['feedback_type'] == 'prediction_accuracy':
            processed_feedback.update(self._process_prediction_feedback(data))
        elif data['feedback_type'] == 'system_usability':
            processed_feedback.update(self._process_usability_feedback(data))
        elif data['feedback_type'] == 'general':
            processed_feedback.update(self._process_general_feedback(data))
        elif data['feedback_type'] == 'suggestion':
            processed_feedback.update(self._process_suggestion_feedback(data))
        
        # Generate summary
        processed_feedback['summary'] = self._generate_feedback_summary(processed_feedback)
        
        return processed_feedback
    
    def _process_prediction_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process feedback specifically about prediction accuracy."""
        self.log_processing_step("Processing prediction accuracy feedback")
        
        return {
            'category': 'prediction_accuracy',
            'accuracy_rating': data.get('accuracy_rating', 0),  # 1-5 scale
            'predicted_disease': data.get('predicted_disease', ''),
            'actual_diagnosis': data.get('actual_diagnosis', ''),
            'doctor_confirmed': data.get('doctor_confirmed', False),
            'helpful_rating': data.get('helpful_rating', 0),
            'comments': data.get('comments', ''),
            'false_positive': self._is_false_positive(data),
            'false_negative': self._is_false_negative(data)
        }
    
    def _process_usability_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process feedback about system usability."""
        self.log_processing_step("Processing usability feedback")
        
        return {
            'category': 'usability',
            'ease_of_use': data.get('ease_of_use', 0),  # 1-5 scale
            'interface_rating': data.get('interface_rating', 0),
            'speed_rating': data.get('speed_rating', 0),
            'clarity_rating': data.get('clarity_rating', 0),
            'overall_satisfaction': data.get('overall_satisfaction', 0),
            'problematic_features': data.get('problematic_features', []),
            'liked_features': data.get('liked_features', []),
            'suggestions': data.get('suggestions', ''),
            'would_recommend': data.get('would_recommend', False)
        }
    
    def _process_general_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process general feedback."""
        self.log_processing_step("Processing general feedback")
        
        return {
            'category': 'general',
            'overall_rating': data.get('overall_rating', 0),
            'feedback_text': data.get('feedback_text', ''),
            'sentiment': self._analyze_sentiment(data.get('feedback_text', '')),
            'topics': self._extract_topics(data.get('feedback_text', ''))
        }
    
    def _process_suggestion_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process improvement suggestions."""
        self.log_processing_step("Processing suggestion feedback")
        
        return {
            'category': 'suggestion',
            'suggestion_type': data.get('suggestion_type', 'general'),
            'suggestion_text': data.get('suggestion_text', ''),
            'priority': data.get('priority', 'medium'),
            'feasibility': self._assess_suggestion_feasibility(data.get('suggestion_text', '')),
            'category_tags': self._categorize_suggestion(data.get('suggestion_text', ''))
        }
    
    def _is_false_positive(self, data: Dict[str, Any]) -> bool:
        """Determine if prediction was a false positive."""
        predicted = data.get('predicted_disease', '').lower()
        actual = data.get('actual_diagnosis', '').lower()
        
        if predicted and actual and predicted != actual:
            if data.get('doctor_confirmed', False):
                return True
        
        return False
    
    def _is_false_negative(self, data: Dict[str, Any]) -> bool:
        """Determine if prediction missed a condition (false negative)."""
        predicted = data.get('predicted_disease', '').lower()
        actual = data.get('actual_diagnosis', '').lower()
        
        if actual and (not predicted or 'none' in predicted or 'low risk' in predicted):
            if data.get('doctor_confirmed', False):
                return True
        
        return False
    
    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis of feedback text."""
        if not text:
            return 'neutral'
        
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'helpful', 'accurate', 'useful', 'satisfied', 'recommend']
        negative_words = ['bad', 'poor', 'terrible', 'inaccurate', 'useless', 'frustrated', 'disappointed', 'wrong']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from feedback text."""
        if not text:
            return []
        
        text_lower = text.lower()
        topics = []
        
        topic_keywords = {
            'accuracy': ['accurate', 'accuracy', 'correct', 'wrong', 'prediction'],
            'interface': ['interface', 'ui', 'design', 'layout', 'navigation'],
            'speed': ['slow', 'fast', 'speed', 'performance', 'loading'],
            'explanation': ['explanation', 'understand', 'clear', 'confusing'],
            'features': ['feature', 'functionality', 'option', 'tool'],
            'recommendation': ['recommend', 'suggestion', 'advice', 'next steps']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _assess_suggestion_feasibility(self, suggestion_text: str) -> str:
        """Assess the feasibility of implementing a suggestion."""
        if not suggestion_text:
            return 'unknown'
        
        text_lower = suggestion_text.lower()
        
        # Simple feasibility assessment based on keywords
        if any(word in text_lower for word in ['simple', 'easy', 'quick', 'minor']):
            return 'high'
        elif any(word in text_lower for word in ['complex', 'difficult', 'major', 'complete']):
            return 'low'
        else:
            return 'medium'
    
    def _categorize_suggestion(self, suggestion_text: str) -> List[str]:
        """Categorize suggestions into implementation areas."""
        if not suggestion_text:
            return []
        
        text_lower = suggestion_text.lower()
        categories = []
        
        category_keywords = {
            'ui_improvement': ['interface', 'design', 'layout', 'user experience'],
            'new_feature': ['add', 'new', 'feature', 'include', 'support'],
            'accuracy_improvement': ['accurate', 'better prediction', 'improve model'],
            'performance': ['faster', 'speed', 'performance', 'optimize'],
            'explanation': ['explain', 'clarify', 'understand', 'reasoning'],
            'data_input': ['input', 'upload', 'data entry', 'form']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)
        
        return categories
    
    def _store_feedback(self, processed_feedback: Dict[str, Any]) -> None:
        """Store processed feedback for analysis."""
        self.log_processing_step("Storing feedback")
        
        # In a production system, this would store to a database
        self.feedback_storage.append(processed_feedback)
        
        # Keep only last 1000 feedback entries in memory
        if len(self.feedback_storage) > 1000:
            self.feedback_storage = self.feedback_storage[-1000:]
    
    def _generate_feedback_insights(self, processed_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from the feedback."""
        insights = {
            'feedback_category': processed_feedback['category'],
            'sentiment_detected': processed_feedback.get('sentiment', 'neutral'),
            'priority_level': self._determine_priority_level(processed_feedback),
            'action_required': self._determine_action_required(processed_feedback)
        }
        
        # Add category-specific insights
        if processed_feedback['category'] == 'prediction_accuracy':
            insights['prediction_insights'] = {
                'accuracy_score': processed_feedback.get('accuracy_rating', 0),
                'needs_model_review': processed_feedback.get('accuracy_rating', 0) < 3,
                'false_positive_detected': processed_feedback.get('false_positive', False),
                'false_negative_detected': processed_feedback.get('false_negative', False)
            }
        elif processed_feedback['category'] == 'usability':
            insights['usability_insights'] = {
                'overall_usability_score': processed_feedback.get('overall_satisfaction', 0),
                'main_pain_points': processed_feedback.get('problematic_features', []),
                'successful_features': processed_feedback.get('liked_features', [])
            }
        
        return insights
    
    def _determine_priority_level(self, processed_feedback: Dict[str, Any]) -> str:
        """Determine the priority level of the feedback."""
        category = processed_feedback['category']
        
        if category == 'prediction_accuracy':
            if processed_feedback.get('false_positive') or processed_feedback.get('false_negative'):
                return 'high'
            elif processed_feedback.get('accuracy_rating', 0) < 3:
                return 'medium'
            else:
                return 'low'
        
        elif category == 'usability':
            if processed_feedback.get('overall_satisfaction', 0) < 3:
                return 'medium'
            else:
                return 'low'
        
        elif category == 'suggestion':
            priority = processed_feedback.get('priority', 'medium')
            return priority
        
        else:
            sentiment = processed_feedback.get('sentiment', 'neutral')
            if sentiment == 'negative':
                return 'medium'
            else:
                return 'low'
    
    def _determine_action_required(self, processed_feedback: Dict[str, Any]) -> bool:
        """Determine if immediate action is required."""
        priority = self._determine_priority_level(processed_feedback)
        
        if priority == 'high':
            return True
        
        # Check for specific conditions requiring action
        if processed_feedback['category'] == 'prediction_accuracy':
            if processed_feedback.get('false_positive') or processed_feedback.get('false_negative'):
                return True
        
        return False
    
    def _generate_improvement_suggestions(self, processed_feedback: Dict[str, Any]) -> List[str]:
        """Generate suggestions for system improvement based on feedback."""
        suggestions = []
        
        category = processed_feedback['category']
        
        if category == 'prediction_accuracy':
            if processed_feedback.get('accuracy_rating', 0) < 3:
                suggestions.append("Review and retrain prediction models")
                suggestions.append("Collect more training data for specific conditions")
            
            if processed_feedback.get('false_positive'):
                suggestions.append("Improve specificity of prediction models")
                suggestions.append("Add additional validation steps for high-confidence predictions")
            
            if processed_feedback.get('false_negative'):
                suggestions.append("Improve sensitivity of prediction models")
                suggestions.append("Review feature extraction and preprocessing steps")
        
        elif category == 'usability':
            problematic_features = processed_feedback.get('problematic_features', [])
            for feature in problematic_features:
                suggestions.append(f"Improve user experience for {feature}")
            
            if processed_feedback.get('overall_satisfaction', 0) < 3:
                suggestions.append("Conduct comprehensive UX review")
                suggestions.append("Implement user testing sessions")
        
        elif category == 'suggestion':
            suggestion_text = processed_feedback.get('suggestion_text', '')
            if suggestion_text:
                suggestions.append(f"Consider implementing: {suggestion_text}")
        
        # Add general suggestions based on sentiment
        sentiment = processed_feedback.get('sentiment', 'neutral')
        if sentiment == 'negative':
            suggestions.append("Follow up with user to understand specific concerns")
            suggestions.append("Review overall user experience flow")
        
        return suggestions
    
    def _generate_feedback_summary(self, processed_feedback: Dict[str, Any]) -> str:
        """Generate a summary of the processed feedback."""
        category = processed_feedback['category']
        timestamp = processed_feedback['timestamp']
        
        summary = f"Feedback received on {timestamp[:10]} - Category: {category.replace('_', ' ').title()}"
        
        if category == 'prediction_accuracy':
            accuracy = processed_feedback.get('accuracy_rating', 0)
            summary += f" - Accuracy Rating: {accuracy}/5"
        elif category == 'usability':
            satisfaction = processed_feedback.get('overall_satisfaction', 0)
            summary += f" - Satisfaction: {satisfaction}/5"
        elif category == 'general':
            sentiment = processed_feedback.get('sentiment', 'neutral')
            summary += f" - Sentiment: {sentiment.title()}"
        
        return summary
    
    def _generate_thank_you_message(self, processed_feedback: Dict[str, Any]) -> str:
        """Generate a personalized thank you message."""
        category = processed_feedback['category']
        
        base_message = "Thank you for your valuable feedback! "
        
        if category == 'prediction_accuracy':
            return base_message + "Your input helps us improve our prediction accuracy and provide better health insights."
        elif category == 'usability':
            return base_message + "Your suggestions help us create a better user experience for everyone."
        elif category == 'suggestion':
            return base_message + "Your suggestions are important for our continuous improvement."
        else:
            return base_message + "We appreciate you taking the time to share your thoughts with us."
    
    def get_feedback_analytics(self) -> Dict[str, Any]:
        """Get analytics on collected feedback."""
        if not self.feedback_storage:
            return {'total_feedback': 0, 'message': 'No feedback collected yet'}
        
        total_feedback = len(self.feedback_storage)
        
        # Count by category
        category_counts = {}
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for feedback in self.feedback_storage:
            category = feedback.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
            
            sentiment = feedback.get('sentiment', 'neutral')
            sentiment_counts[sentiment] += 1
        
        # Calculate averages for prediction accuracy
        accuracy_ratings = [f.get('accuracy_rating', 0) for f in self.feedback_storage 
                           if f.get('category') == 'prediction_accuracy' and f.get('accuracy_rating')]
        avg_accuracy = sum(accuracy_ratings) / len(accuracy_ratings) if accuracy_ratings else 0
        
        return {
            'total_feedback': total_feedback,
            'category_breakdown': category_counts,
            'sentiment_breakdown': sentiment_counts,
            'average_accuracy_rating': round(avg_accuracy, 2),
            'last_feedback_date': self.feedback_storage[-1]['timestamp'] if self.feedback_storage else None
        }
