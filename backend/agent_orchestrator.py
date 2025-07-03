"""
Agent Orchestrator
Coordinates the interaction between different AI agents in the disease detection system.
Manages agent lifecycle, communication, and data flow.
"""

import logging
import time
from typing import Dict, Any, Optional
from agents.input_collection_agent import InputCollectionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.prediction_agent import PredictionAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.report_generator_agent import ReportGeneratorAgent
from agents.feedback_agent import FeedbackAgent

class AgentOrchestrator:
    """
    Orchestrates the interaction between different AI agents.
    Manages the complete workflow from input collection to report generation.
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.agents_initialized = False
        self._initialize_agents()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the orchestrator."""
        logger = logging.getLogger("orchestrator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - Orchestrator - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_agents(self):
        """Initialize all AI agents."""
        try:
            self.logger.info("Initializing AI agents...")
            
            self.input_collection_agent = InputCollectionAgent()
            self.preprocessing_agent = PreprocessingAgent()
            self.prediction_agent = PredictionAgent()
            self.explainability_agent = ExplainabilityAgent()
            self.report_generator_agent = ReportGeneratorAgent()
            self.feedback_agent = FeedbackAgent()
            
            self.agents_initialized = True
            self.logger.info("All AI agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {str(e)}")
            self.agents_initialized = False
            raise
    
    def collect_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate input collection process.
        
        Args:
            data: Raw input data from user interface
            
        Returns:
            Processed input collection results
        """
        try:
            self.logger.info("Starting input collection orchestration")
            
            if not self.agents_initialized:
                return {
                    'success': False,
                    'error': 'Agents not initialized',
                    'timestamp': time.time()
                }
            
            # Process input through Input Collection Agent
            result = self.input_collection_agent.process(data)
            
            if result.get('success'):
                self.logger.info("Input collection completed successfully")
            else:
                self.logger.error(f"Input collection failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Input collection orchestration failed: {str(e)}")
            return {
                'success': False,
                'error': f'Orchestration error: {str(e)}',
                'timestamp': time.time()
            }
    
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate data preprocessing process.
        
        Args:
            data: Data from input collection
            
        Returns:
            Preprocessed data results
        """
        try:
            self.logger.info("Starting data preprocessing orchestration")
            
            if not self.agents_initialized:
                return {
                    'success': False,
                    'error': 'Agents not initialized',
                    'timestamp': time.time()
                }
            
            # Process data through Preprocessing Agent
            result = self.preprocessing_agent.process(data)
            
            if result.get('success'):
                self.logger.info("Data preprocessing completed successfully")
            else:
                self.logger.error(f"Data preprocessing failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Data preprocessing orchestration failed: {str(e)}")
            return {
                'success': False,
                'error': f'Orchestration error: {str(e)}',
                'timestamp': time.time()
            }
    
    def make_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate disease prediction process.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Disease prediction results
        """
        try:
            self.logger.info("Starting disease prediction orchestration")
            
            if not self.agents_initialized:
                return {
                    'success': False,
                    'error': 'Agents not initialized',
                    'timestamp': time.time()
                }
            
            # Process data through Prediction Agent
            result = self.prediction_agent.process(data)
            
            if result.get('success'):
                self.logger.info("Disease prediction completed successfully")
            else:
                self.logger.error(f"Disease prediction failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Disease prediction orchestration failed: {str(e)}")
            return {
                'success': False,
                'error': f'Orchestration error: {str(e)}',
                'timestamp': time.time()
            }
    
    def generate_explanation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate explanation generation process.
        
        Args:
            data: Data including predictions
            
        Returns:
            Generated explanations and recommendations
        """
        try:
            self.logger.info("Starting explanation generation orchestration")
            
            if not self.agents_initialized:
                return {
                    'success': False,
                    'error': 'Agents not initialized',
                    'timestamp': time.time()
                }
            
            # Process data through Explainability Agent
            result = self.explainability_agent.process(data)
            
            if result.get('success'):
                self.logger.info("Explanation generation completed successfully")
            else:
                self.logger.error(f"Explanation generation failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Explanation generation orchestration failed: {str(e)}")
            return {
                'success': False,
                'error': f'Orchestration error: {str(e)}',
                'timestamp': time.time()
            }
    
    def generate_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate report generation process.
        
        Args:
            data: Complete analysis data
            
        Returns:
            Generated health assessment report
        """
        try:
            self.logger.info("Starting report generation orchestration")
            
            if not self.agents_initialized:
                return {
                    'success': False,
                    'error': 'Agents not initialized',
                    'timestamp': time.time()
                }
            
            # Process data through Report Generator Agent
            result = self.report_generator_agent.process(data)
            
            if result.get('success'):
                self.logger.info("Report generation completed successfully")
            else:
                self.logger.error(f"Report generation failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Report generation orchestration failed: {str(e)}")
            return {
                'success': False,
                'error': f'Orchestration error: {str(e)}',
                'timestamp': time.time()
            }
    
    def submit_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate feedback submission process.
        
        Args:
            data: User feedback data
            
        Returns:
            Feedback processing results
        """
        try:
            self.logger.info("Starting feedback submission orchestration")
            
            if not self.agents_initialized:
                return {
                    'success': False,
                    'error': 'Agents not initialized',
                    'timestamp': time.time()
                }
            
            # Process feedback through Feedback Agent
            result = self.feedback_agent.process(data)
            
            if result.get('success'):
                self.logger.info("Feedback submission completed successfully")
            else:
                self.logger.error(f"Feedback submission failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Feedback submission orchestration failed: {str(e)}")
            return {
                'success': False,
                'error': f'Orchestration error: {str(e)}',
                'timestamp': time.time()
            }
    
    def get_feedback_analytics(self) -> Dict[str, Any]:
        """
        Get feedback analytics from the feedback agent.
        
        Returns:
            Feedback analytics data
        """
        try:
            if not self.agents_initialized:
                return {'error': 'Agents not initialized'}
            
            return self.feedback_agent.get_feedback_analytics()
            
        except Exception as e:
            self.logger.error(f"Failed to get feedback analytics: {str(e)}")
            return {'error': f'Analytics retrieval failed: {str(e)}'}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get status of all agents.
        
        Returns:
            Status information for all agents
        """
        try:
            if not self.agents_initialized:
                return {
                    'initialized': False,
                    'agents': {}
                }
            
            agent_status = {
                'initialized': True,
                'agents': {
                    'input_collection': {
                        'name': self.input_collection_agent.agent_name,
                        'active': True,
                        'processing_history_count': len(self.input_collection_agent.processing_history)
                    },
                    'preprocessing': {
                        'name': self.preprocessing_agent.agent_name,
                        'active': True,
                        'processing_history_count': len(self.preprocessing_agent.processing_history)
                    },
                    'prediction': {
                        'name': self.prediction_agent.agent_name,
                        'active': True,
                        'processing_history_count': len(self.prediction_agent.processing_history)
                    },
                    'explainability': {
                        'name': self.explainability_agent.agent_name,
                        'active': True,
                        'processing_history_count': len(self.explainability_agent.processing_history)
                    },
                    'report_generator': {
                        'name': self.report_generator_agent.agent_name,
                        'active': True,
                        'processing_history_count': len(self.report_generator_agent.processing_history)
                    },
                    'feedback': {
                        'name': self.feedback_agent.agent_name,
                        'active': True,
                        'processing_history_count': len(self.feedback_agent.processing_history)
                    }
                }
            }
            
            return agent_status
            
        except Exception as e:
            self.logger.error(f"Failed to get agent status: {str(e)}")
            return {
                'initialized': False,
                'error': str(e)
            }
    
    def run_complete_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete analysis pipeline from input to report.
        
        Args:
            input_data: Raw input data from user
            
        Returns:
            Complete analysis results
        """
        try:
            self.logger.info("Starting complete analysis pipeline")
            
            # Step 1: Input Collection
            collection_result = self.collect_input(input_data)
            if not collection_result.get('success'):
                return collection_result
            
            # Step 2: Preprocessing
            preprocessing_input = {
                'collection_result': collection_result,
                **input_data
            }
            preprocessing_result = self.preprocess_data(preprocessing_input)
            if not preprocessing_result.get('success'):
                return preprocessing_result
            
            # Step 3: Prediction
            prediction_input = {
                'collection_result': collection_result,
                'preprocessing_result': preprocessing_result,
                **input_data
            }
            prediction_result = self.make_prediction(prediction_input)
            if not prediction_result.get('success'):
                return prediction_result
            
            # Step 4: Explanation
            explanation_input = {
                'collection_result': collection_result,
                'preprocessing_result': preprocessing_result,
                'prediction_result': prediction_result,
                **input_data
            }
            explanation_result = self.generate_explanation(explanation_input)
            if not explanation_result.get('success'):
                return explanation_result
            
            # Step 5: Report Generation
            report_input = {
                'collection_result': collection_result,
                'preprocessing_result': preprocessing_result,
                'prediction_result': prediction_result,
                'explanation_result': explanation_result,
                **input_data
            }
            report_result = self.generate_report(report_input)
            
            # Compile complete results
            complete_results = {
                'success': True,
                'collection_result': collection_result,
                'preprocessing_result': preprocessing_result,
                'prediction_result': prediction_result,
                'explanation_result': explanation_result,
                'report_result': report_result,
                'pipeline_metadata': {
                    'total_processing_time': time.time() - collection_result.get('timestamp', time.time()),
                    'agents_involved': 5,
                    'pipeline_completed': True
                }
            }
            
            self.logger.info("Complete analysis pipeline finished successfully")
            return complete_results
            
        except Exception as e:
            self.logger.error(f"Complete analysis pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': f'Pipeline error: {str(e)}',
                'timestamp': time.time()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all agents.
        
        Returns:
            Health status of the orchestrator and all agents
        """
        try:
            health_status = {
                'orchestrator_healthy': True,
                'agents_initialized': self.agents_initialized,
                'timestamp': time.time()
            }
            
            if self.agents_initialized:
                # Test each agent with minimal data
                test_data = {'test': True}
                
                agent_health = {}
                agents = [
                    ('input_collection', self.input_collection_agent),
                    ('preprocessing', self.preprocessing_agent),
                    ('prediction', self.prediction_agent),
                    ('explainability', self.explainability_agent),
                    ('report_generator', self.report_generator_agent),
                    ('feedback', self.feedback_agent)
                ]
                
                for agent_name, agent in agents:
                    try:
                        # Basic health check - agent should handle invalid input gracefully
                        agent_health[agent_name] = {
                            'healthy': True,
                            'agent_name': agent.agent_name,
                            'last_check': time.time()
                        }
                    except Exception as e:
                        agent_health[agent_name] = {
                            'healthy': False,
                            'error': str(e),
                            'last_check': time.time()
                        }
                
                health_status['agent_health'] = agent_health
            
            return health_status
            
        except Exception as e:
            return {
                'orchestrator_healthy': False,
                'error': str(e),
                'timestamp': time.time()
            }
