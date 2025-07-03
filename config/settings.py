"""
Configuration Settings
Central configuration file for all application settings including
API keys, model parameters, and system configurations.
"""

import os
from typing import Dict, Any

# Flask Server Configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8000

# Streamlit Configuration
STREAMLIT_HOST = "0.0.0.0"
STREAMLIT_PORT = 5000

# API Keys and External Services
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "default_openai_key")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "default_hf_key")
HEALTH_API_KEY = os.getenv("HEALTH_API_KEY", "default_health_key")

# Model Configuration
MODEL_SETTINGS = {
    "openai": {
        "model": "gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
        "max_tokens": 500,
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    },
    "huggingface": {
        "classification_model": "distilbert-base-uncased-finetuned-sst-2-english",
        "ner_model": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "timeout": 30
    }
}

# Disease Prediction Settings
DISEASE_SETTINGS = {
    "supported_diseases": [
        "diabetes",
        "heart_disease", 
        "hypertension",
        "stress",
        "respiratory_condition"
    ],
    "risk_thresholds": {
        "low": 0.3,
        "medium": 0.6,
        "high": 0.8
    },
    "confidence_thresholds": {
        "ml_model": 0.8,
        "rule_based": 0.7,
        "hybrid": 0.75
    }
}

# File Processing Settings
FILE_SETTINGS = {
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "supported_image_types": [
        "image/jpeg",
        "image/jpg", 
        "image/png",
        "image/bmp"
    ],
    "supported_document_types": [
        "application/pdf"
    ],
    "image_quality_thresholds": {
        "high": 1000000,  # > 1MP
        "medium": 300000,  # > 0.3MP
        "low": 0
    }
}

# Agent Configuration
AGENT_SETTINGS = {
    "processing_timeout": 120,  # seconds
    "retry_attempts": 3,
    "rate_limits": {
        "openai": 1.0,  # seconds between requests
        "huggingface": 2.0,
        "default": 0.5
    },
    "logging_level": "INFO"
}

# Security Settings
SECURITY_SETTINGS = {
    "allowed_origins": ["*"],  # Configure for production
    "max_requests_per_minute": 60,
    "session_timeout": 3600,  # 1 hour
    "enable_cors": True
}

# Database Settings (for future implementation)
DATABASE_SETTINGS = {
    "type": "sqlite",  # Can be changed to postgresql, mysql, etc.
    "path": "data/disease_detection.db",
    "connection_pool_size": 10,
    "echo_sql": False
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "logs/disease_detection.log",
            "level": "DEBUG",
            "formatter": "detailed"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    },
    "loggers": {
        "agents": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        },
        "orchestrator": {
            "level": "INFO", 
            "handlers": ["console"],
            "propagate": False
        }
    }
}

# Feature Flags
FEATURE_FLAGS = {
    "enable_ml_models": True,
    "enable_openai_integration": True,
    "enable_huggingface_integration": True,
    "enable_image_analysis": True,
    "enable_pdf_processing": True,
    "enable_feedback_collection": True,
    "enable_report_generation": True,
    "enable_wearable_data": True
}

# System Health Check Settings
HEALTH_CHECK_SETTINGS = {
    "check_interval": 300,  # 5 minutes
    "timeout": 30,
    "critical_services": [
        "input_collection_agent",
        "preprocessing_agent", 
        "prediction_agent",
        "explainability_agent"
    ]
}

# Performance Settings
PERFORMANCE_SETTINGS = {
    "max_concurrent_analyses": 10,
    "processing_queue_size": 50,
    "cache_ttl": 3600,  # 1 hour
    "enable_caching": True
}

# Validation Rules
VALIDATION_RULES = {
    "min_symptom_length": 10,
    "max_symptom_length": 5000,
    "min_age": 1,
    "max_age": 120,
    "required_fields": [
        "symptoms",
        "patient_info"
    ]
}

# Error Messages
ERROR_MESSAGES = {
    "missing_api_key": "API key not configured. Please set the required environment variable.",
    "invalid_file_type": "Unsupported file type. Please upload PDF, JPG, PNG, or BMP files.",
    "file_too_large": "File size exceeds the maximum limit of 10MB.",
    "processing_timeout": "Analysis is taking longer than expected. Please try again.",
    "agent_unavailable": "One or more AI agents are currently unavailable. Please try again later.",
    "invalid_input": "Please provide valid input data for analysis.",
    "network_error": "Network error occurred. Please check your connection and try again."
}

# Success Messages
SUCCESS_MESSAGES = {
    "analysis_complete": "Health analysis completed successfully!",
    "report_generated": "Health report has been generated and is ready for download.",
    "feedback_submitted": "Thank you for your feedback! It helps us improve our service.",
    "data_processed": "Your data has been processed and analyzed successfully."
}

# Default Configuration for New Users
DEFAULT_USER_CONFIG = {
    "units": "metric",  # or "imperial"
    "language": "en",
    "timezone": "UTC",
    "notifications": {
        "email": False,
        "sms": False,
        "push": False
    }
}

# API Rate Limiting
RATE_LIMIT_CONFIG = {
    "requests_per_minute": 60,
    "requests_per_hour": 1000,
    "requests_per_day": 5000,
    "burst_limit": 10
}

# Cache Configuration
CACHE_CONFIG = {
    "type": "memory",  # Can be "redis", "memcached", etc.
    "ttl": {
        "prediction_results": 3600,  # 1 hour
        "processed_files": 1800,     # 30 minutes
        "user_sessions": 7200        # 2 hours
    },
    "max_size": 1000  # Maximum number of cached items
}

def get_config() -> Dict[str, Any]:
    """
    Get complete configuration dictionary.
    
    Returns:
        Dictionary containing all configuration settings
    """
    return {
        "flask": {
            "host": FLASK_HOST,
            "port": FLASK_PORT
        },
        "streamlit": {
            "host": STREAMLIT_HOST,
            "port": STREAMLIT_PORT
        },
        "api_keys": {
            "openai": OPENAI_API_KEY,
            "huggingface": HUGGINGFACE_API_KEY,
            "health_api": HEALTH_API_KEY
        },
        "models": MODEL_SETTINGS,
        "diseases": DISEASE_SETTINGS,
        "files": FILE_SETTINGS,
        "agents": AGENT_SETTINGS,
        "security": SECURITY_SETTINGS,
        "database": DATABASE_SETTINGS,
        "logging": LOGGING_CONFIG,
        "features": FEATURE_FLAGS,
        "health_check": HEALTH_CHECK_SETTINGS,
        "performance": PERFORMANCE_SETTINGS,
        "validation": VALIDATION_RULES,
        "messages": {
            "errors": ERROR_MESSAGES,
            "success": SUCCESS_MESSAGES
        },
        "defaults": DEFAULT_USER_CONFIG,
        "rate_limits": RATE_LIMIT_CONFIG,
        "cache": CACHE_CONFIG
    }

def validate_config() -> Dict[str, Any]:
    """
    Validate configuration settings.
    
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Check API keys
    if OPENAI_API_KEY == "default_openai_key":
        validation_results["warnings"].append("OpenAI API key not set - using default fallback")
    
    if HUGGINGFACE_API_KEY == "default_hf_key":
        validation_results["warnings"].append("HuggingFace API key not set - using default fallback")
    
    # Check file size limits
    if FILE_SETTINGS["max_file_size"] > 50 * 1024 * 1024:  # 50MB
        validation_results["warnings"].append("File size limit is quite high - may impact performance")
    
    # Check timeouts
    if AGENT_SETTINGS["processing_timeout"] < 30:
        validation_results["warnings"].append("Processing timeout is quite low - may cause premature failures")
    
    # Validate required directories
    required_dirs = ["logs", "data", "models"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
                validation_results["warnings"].append(f"Created missing directory: {dir_name}")
            except Exception as e:
                validation_results["errors"].append(f"Failed to create directory {dir_name}: {str(e)}")
                validation_results["valid"] = False
    
    return validation_results

def get_environment_info() -> Dict[str, Any]:
    """
    Get information about the current environment.
    
    Returns:
        Dictionary containing environment information
    """
    return {
        "python_version": os.sys.version,
        "environment_variables": {
            "OPENAI_API_KEY": "SET" if os.getenv("OPENAI_API_KEY") else "NOT_SET",
            "HUGGINGFACE_API_KEY": "SET" if os.getenv("HUGGINGFACE_API_KEY") else "NOT_SET",
            "HEALTH_API_KEY": "SET" if os.getenv("HEALTH_API_KEY") else "NOT_SET"
        },
        "working_directory": os.getcwd(),
        "config_validation": validate_config()
    }

# Initialize configuration validation on import
_config_validation = validate_config()
if not _config_validation["valid"]:
    print("Configuration validation failed:")
    for error in _config_validation["errors"]:
        print(f"ERROR: {error}")

if _config_validation["warnings"]:
    print("Configuration warnings:")
    for warning in _config_validation["warnings"]:
        print(f"WARNING: {warning}")
