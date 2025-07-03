# AI Disease Detection System

## Overview

This is a comprehensive AI-powered disease detection web application that uses a multi-agent architecture to analyze symptoms, medical documents, and optional wearable data to provide health insights and recommendations. The system employs six specialized AI agents working together to deliver accurate disease predictions and explanations.

## System Architecture

### Frontend Architecture
- **Primary Interface**: Streamlit web application (`app.py`)
- **Communication**: Streamlit frontend communicates with Flask backend via HTTP requests
- **User Experience**: Multi-step wizard interface with progress tracking
- **File Upload**: Supports PDF documents, medical images (X-rays, scans), and text input

### Backend Architecture
- **Framework**: Flask web server with CORS enabled
- **Agent Orchestration**: Central orchestrator manages agent lifecycle and communication
- **API Structure**: RESTful endpoints for each agent operation
- **Threading**: Flask server runs in background thread from Streamlit

### Multi-Agent System
The application implements six specialized agents:

1. **Input Collection Agent**: Validates and organizes user inputs (symptoms, medical history, files)
2. **Preprocessing Agent**: Cleans and extracts information from documents and raw data
3. **Prediction Agent**: Uses ML models and APIs for disease likelihood prediction
4. **Explainability Agent**: Generates human-readable explanations using OpenAI GPT
5. **Report Generator Agent**: Creates comprehensive downloadable health reports
6. **Feedback Agent**: Collects user feedback for system improvement

## Key Components

### Agent Communication
- **Base Agent Class**: Abstract base class providing common functionality (logging, error handling, communication)
- **Agent Orchestrator**: Coordinates agent interactions and manages data flow between agents
- **Standardized Interface**: All agents implement consistent `process()` method for data handling

### External Service Integration
- **OpenAI API**: Used for generating medical explanations and natural language processing
- **HuggingFace API**: Provides access to pre-trained models for text analysis and classification
- **Rate Limiting**: Built-in rate limiting for external API calls to prevent quota exhaustion

### File Processing
- **PDF Processing**: Uses PyPDF2 for extracting text from medical reports
- **Image Processing**: PIL for handling medical images (X-rays, scans)
- **Base64 Encoding**: Secure file transfer between frontend and backend

### Disease Detection Models
- **Lightweight ML Models**: Scikit-learn based models for offline prediction capability
- **Rule-based Classifiers**: Fallback algorithms when ML models unavailable
- **Multiple Disease Support**: Diabetes, heart disease, hypertension, stress, respiratory conditions

## Data Flow

1. **Input Collection**: User provides symptoms, medical history, and optional files through Streamlit interface
2. **Validation**: Input Collection Agent validates and organizes all user data
3. **Preprocessing**: Raw data is cleaned, documents are parsed, and information is extracted
4. **Prediction**: ML models and APIs analyze processed data to predict disease likelihood
5. **Explanation**: OpenAI GPT generates human-readable explanations of predictions
6. **Report Generation**: Comprehensive health assessment report is created
7. **Feedback**: Optional user feedback collection for system improvement

## External Dependencies

### Required APIs
- **OpenAI API**: For medical explanations and natural language generation (GPT-4o model)
- **HuggingFace API**: For text classification and named entity recognition

### Python Packages
- **Web Framework**: Flask, Streamlit, Flask-CORS
- **AI/ML**: OpenAI, scikit-learn, numpy, pandas
- **File Processing**: PyPDF2, Pillow (PIL)
- **HTTP Requests**: requests library

### Configuration
- Environment variables for API keys (OPENAI_API_KEY, HUGGINGFACE_API_KEY)
- Centralized settings in `config/settings.py`
- Fallback to default values when API keys not provided

## Deployment Strategy

### Local Development
- Flask backend runs on port 8000
- Streamlit frontend runs on port 5000
- Background threading for Flask server initialization

### Production Considerations
- Replace in-memory storage with persistent database (feedback, processing history)
- Implement proper logging and monitoring
- Add authentication and user management
- Scale agent orchestration for concurrent users

### Offline Capability
- Lightweight ML models for offline prediction
- Rule-based fallback algorithms
- Local file processing without external dependencies

## Enhanced Features (Latest Updates)

### Symptom Clustering Analysis
- **Intelligent Pattern Recognition**: Analyzes symptoms across 6 disease categories (metabolic, cardiovascular, respiratory, neurological, mental health, gastrointestinal)
- **Confidence Scoring**: Provides percentage confidence for disease cluster matches
- **Example Output**: "Clustered symptoms point 72% toward metabolic syndrome"
- **Implementation**: New `utils/symptom_clustering.py` module with weighted clustering algorithms

### Lab Report Integration
- **Automatic Value Extraction**: Supports blood glucose, cholesterol panels, kidney function, liver function, CBC, thyroid, and inflammatory markers
- **Risk Score Adjustment**: Lab values dynamically modify disease prediction probabilities
- **Normal Range Validation**: Compares extracted values against medical reference ranges
- **Implementation**: New `utils/lab_report_analyzer.py` with comprehensive lab parsing

### Follow-up Question Model
- **Adaptive Questioning**: Generates 3 context-aware questions based on initial analysis
- **Question Types**: Yes/No, scale ratings, frequency assessments, duration tracking, and descriptive responses
- **Accuracy Improvement**: Questions designed to refine prediction confidence
- **Implementation**: Integrated into prediction agent with UI step 5 for question collection

### Enhanced User Interface
- **Step 5 Added**: Dedicated follow-up questions interface with multiple input types
- **Cluster Analysis Display**: Visual representation of symptom pattern matches
- **Lab Analysis Section**: Shows extracted values and risk adjustments
- **Confidence Indicators**: Clear display of lab-adjusted vs original predictions

## Changelog
```
Changelog:
- July 03, 2025: Enhanced with symptom clustering, lab integration, and follow-up questions
- July 03, 2025: Added comprehensive GitHub documentation (README, CONTRIBUTING)
- July 03, 2025: Initial multi-agent architecture setup
```

## User Preferences
```
Preferred communication style: Simple, everyday language.
```