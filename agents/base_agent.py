"""
Base Agent class for all AI agents in the disease detection system.
Provides common functionality and interface for agent communication.
"""

import logging
import time
from typing import Dict, Any, List
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the disease detection system.
    Provides common functionality for logging, communication, and error handling.
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = self._setup_logger()
        self.processing_history = []
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the agent."""
        logger = logging.getLogger(f"agent.{self.agent_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.agent_name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method that each agent must implement.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dict containing processing results
        """
        pass
    
    def log_processing_step(self, step: str, details: str = ""):
        """Log a processing step with timestamp."""
        timestamp = time.time()
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'details': details,
            'agent': self.agent_name
        }
        self.processing_history.append(log_entry)
        self.logger.info(f"{step}: {details}")
    
    def validate_input(self, data: Dict[str, Any], required_fields: List[str]) -> bool:
        """
        Validate that required fields are present in input data.
        
        Args:
            data: Input data dictionary
            required_fields: List of required field names
            
        Returns:
            True if all required fields are present
        """
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            return False
        
        return True
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Handle errors consistently across all agents.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            
        Returns:
            Error response dictionary
        """
        error_msg = f"Error in {self.agent_name}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {str(error)}"
        
        self.logger.error(error_msg)
        
        return {
            'success': False,
            'error': error_msg,
            'agent': self.agent_name,
            'timestamp': time.time()
        }
    
    def create_success_response(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a standardized success response.
        
        Args:
            result_data: The successful processing results
            
        Returns:
            Standardized success response
        """
        return {
            'success': True,
            'agent': self.agent_name,
            'timestamp': time.time(),
            'processing_history': self.processing_history,
            **result_data
        }
    
    def communicate_with_agent(self, target_agent: 'BaseAgent', data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Communicate with another agent.
        
        Args:
            target_agent: The agent to send data to
            data: Data to send
            
        Returns:
            Response from the target agent
        """
        self.log_processing_step(f"Communicating with {target_agent.agent_name}")
        
        try:
            response = target_agent.process(data)
            self.log_processing_step(f"Received response from {target_agent.agent_name}")
            return response
        except Exception as e:
            return self.handle_error(e, f"communication with {target_agent.agent_name}")
