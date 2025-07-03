# AI Disease Detection System

A comprehensive AI-powered disease detection web application that uses multiple specialized agents to analyze symptoms, medical documents, and optional wearable data to provide health insights and recommendations.

## 🏥 Features

### Multi-Agent Architecture
- **Input Collection Agent**: Gathers and validates user input including symptoms, medical history, and file uploads
- **Preprocessing Agent**: Cleans and extracts relevant information from text and medical documents
- **Prediction Agent**: Uses ML models and rule-based algorithms to predict disease likelihood
- **Explainability Agent**: Provides clear, understandable explanations using AI (OpenAI GPT)
- **Report Generator Agent**: Creates comprehensive, downloadable health assessment reports
- **Feedback Agent**: Collects user feedback to improve system performance

### Supported Analysis Types
- **Symptom Analysis**: Natural language processing of symptom descriptions
- **Medical Document Processing**: PDF reports, lab results, medical images
- **Image Analysis**: X-rays, blood test reports, ECG traces (basic analysis)
- **Wearable Data Integration**: Heart rate, SpO₂, sleep data
- **Risk Assessment**: Comprehensive health risk evaluation

### Disease Detection Capabilities
- Diabetes prediction and risk assessment
- Heart disease and cardiovascular conditions
- Hypertension detection
- Stress and anxiety analysis
- Respiratory condition evaluation
- Multi-condition risk analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required API keys (optional but recommended):
  - OpenAI API key for advanced explanations
  - HuggingFace API key for enhanced text analysis

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-disease-detection
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit flask flask-cors requests pandas numpy pillow PyPDF2 openai scikit-learn
   