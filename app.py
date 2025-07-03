import streamlit as st
import requests
import json
import time
import threading
from pathlib import Path
import base64
from io import BytesIO
from backend.flask_server import start_flask_server
from config.settings import FLASK_PORT

# Start Flask server in background thread
if 'flask_started' not in st.session_state:
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()
    st.session_state.flask_started = True
    time.sleep(2)  # Give Flask time to start

def main():
    st.set_page_config(
        page_title="AI Disease Detection System",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 AI-Powered Disease Detection System")
    st.markdown("**Powered by Multiple AI Agents for Comprehensive Health Analysis**")
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {}
    
    # Progress indicator
    progress_steps = ["Input Collection", "Processing", "Analysis", "Results"]
    cols = st.columns(len(progress_steps))
    
    for i, step in enumerate(progress_steps):
        with cols[i]:
            if i + 1 < st.session_state.current_step:
                st.success(f"✅ {step}")
            elif i + 1 == st.session_state.current_step:
                st.info(f"🔄 {step}")
            else:
                st.text(f"⏳ {step}")
    
    st.divider()
    
    # Step routing
    if st.session_state.current_step == 1:
        input_collection_step()
    elif st.session_state.current_step == 2:
        processing_step()
    elif st.session_state.current_step == 3:
        analysis_step()
    elif st.session_state.current_step == 4:
        results_step()

def input_collection_step():
    st.header("📝 Step 1: Input Collection")
    
    # Symptoms input
    st.subheader("Describe Your Symptoms")
    symptoms = st.text_area(
        "Please describe your symptoms in detail:",
        placeholder="e.g., I have been experiencing chest pain, shortness of breath, and fatigue for the past week...",
        height=100
    )
    
    # Medical history
    st.subheader("Medical History")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        existing_conditions = st.multiselect(
            "Existing Medical Conditions",
            ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "None"]
        )
    
    with col2:
        medications = st.text_area("Current Medications", height=80)
        family_history = st.text_area("Family Medical History", height=80)
    
    # Document uploads
    st.subheader("📄 Upload Medical Documents")
    uploaded_files = st.file_uploader(
        "Upload medical reports, X-rays, blood tests, etc.",
        accept_multiple_files=True,
        type=['pdf', 'jpg', 'jpeg', 'png', 'dcm']
    )
    
    # Wearable data (optional)
    st.subheader("📱 Wearable Data (Optional)")
    with st.expander("Add wearable device data"):
        col1, col2, col3 = st.columns(3)
        with col1:
            heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=30, max_value=200, value=70)
        with col2:
            spo2 = st.number_input("SpO₂ (%)", min_value=70, max_value=100, value=98)
        with col3:
            sleep_hours = st.number_input("Average Sleep (hours)", min_value=1, max_value=24, value=8)
    
    # Validate and proceed
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        if not symptoms.strip():
            st.error("Please describe your symptoms before proceeding.")
            return
        
        # Collect all input data
        input_data = {
            'symptoms': symptoms,
            'patient_info': {
                'age': age,
                'gender': gender,
                'existing_conditions': existing_conditions,
                'medications': medications,
                'family_history': family_history
            },
            'wearable_data': {
                'heart_rate': heart_rate,
                'spo2': spo2,
                'sleep_hours': sleep_hours
            },
            'uploaded_files': []
        }
        
        # Process uploaded files
        if uploaded_files:
            for file in uploaded_files:
                file_data = {
                    'name': file.name,
                    'type': file.type,
                    'size': file.size,
                    'content': base64.b64encode(file.read()).decode()
                }
                input_data['uploaded_files'].append(file_data)
        
        st.session_state.analysis_data = input_data
        st.session_state.current_step = 2
        st.rerun()

def processing_step():
    st.header("⚙️ Step 2: Processing Your Data")
    
    # Show processing status
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Call input collection agent
        status_text.text("🤖 Input Collection Agent: Organizing your data...")
        progress_bar.progress(20)
        
        response = requests.post(
            f"http://localhost:{FLASK_PORT}/api/collect_input",
            json=st.session_state.analysis_data,
            timeout=30
        )
        
        if response.status_code != 200:
            st.error(f"Input collection failed: {response.text}")
            return
        
        collection_result = response.json()
        st.session_state.analysis_data['collection_result'] = collection_result
        
        # Call preprocessing agent
        status_text.text("🔍 Preprocessing Agent: Extracting medical information...")
        progress_bar.progress(60)
        
        response = requests.post(
            f"http://localhost:{FLASK_PORT}/api/preprocess",
            json=st.session_state.analysis_data,
            timeout=60
        )
        
        if response.status_code != 200:
            st.error(f"Preprocessing failed: {response.text}")
            return
        
        preprocessing_result = response.json()
        st.session_state.analysis_data['preprocessing_result'] = preprocessing_result
        
        status_text.text("✅ Processing completed successfully!")
        progress_bar.progress(100)
        
        time.sleep(1)
        st.session_state.current_step = 3
        st.rerun()
        
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        st.info("Please ensure the backend service is running.")
    except Exception as e:
        st.error(f"Processing error: {str(e)}")

def analysis_step():
    st.header("🧠 Step 3: AI Analysis in Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Call prediction agent
        status_text.text("🎯 Prediction Agent: Analyzing for potential diseases...")
        progress_bar.progress(30)
        
        response = requests.post(
            f"http://localhost:{FLASK_PORT}/api/predict",
            json=st.session_state.analysis_data,
            timeout=120
        )
        
        if response.status_code != 200:
            st.error(f"Prediction failed: {response.text}")
            return
        
        prediction_result = response.json()
        st.session_state.analysis_data['prediction_result'] = prediction_result
        
        # Call explainability agent
        status_text.text("💡 Explainability Agent: Generating explanations...")
        progress_bar.progress(70)
        
        response = requests.post(
            f"http://localhost:{FLASK_PORT}/api/explain",
            json=st.session_state.analysis_data,
            timeout=90
        )
        
        if response.status_code != 200:
            st.error(f"Explanation generation failed: {response.text}")
            return
        
        explanation_result = response.json()
        st.session_state.analysis_data['explanation_result'] = explanation_result
        
        status_text.text("✅ Analysis completed successfully!")
        progress_bar.progress(100)
        
        time.sleep(1)
        st.session_state.current_step = 4
        st.rerun()
        
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")

def results_step():
    st.header("📊 Step 4: Your Health Analysis Results")
    
    if 'prediction_result' not in st.session_state.analysis_data:
        st.error("No analysis results available. Please restart the process.")
        return
    
    prediction = st.session_state.analysis_data['prediction_result']
    explanation = st.session_state.analysis_data.get('explanation_result', {})
    
    # Display predictions
    st.subheader("🎯 Disease Risk Assessment")
    
    if 'predictions' in prediction:
        for disease_pred in prediction['predictions']:
            disease = disease_pred.get('disease', 'Unknown')
            probability = disease_pred.get('probability', 0)
            risk_level = disease_pred.get('risk_level', 'Unknown')
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{disease}**")
            with col2:
                st.metric("Probability", f"{probability:.1%}")
            with col3:
                if risk_level.lower() == 'high':
                    st.error(f"🔴 {risk_level}")
                elif risk_level.lower() == 'medium':
                    st.warning(f"🟡 {risk_level}")
                else:
                    st.success(f"🟢 {risk_level}")
    
    # Display explanations
    if 'explanation' in explanation:
        st.subheader("💡 AI Explanation")
        st.info(explanation['explanation'])
    
    if 'recommendations' in explanation:
        st.subheader("📋 Recommendations")
        for rec in explanation['recommendations']:
            st.write(f"• {rec}")
    
    # Generate report
    st.subheader("📄 Generate Health Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download PDF Report", type="primary"):
            generate_report()
    
    with col2:
        if st.button("🔄 Start New Analysis"):
            reset_analysis()

def generate_report():
    try:
        response = requests.post(
            f"http://localhost:{FLASK_PORT}/api/generate_report",
            json=st.session_state.analysis_data,
            timeout=60
        )
        
        if response.status_code == 200:
            report_data = response.json()
            if 'report_content' in report_data:
                st.download_button(
                    label="📥 Download Report",
                    data=report_data['report_content'],
                    file_name=f"health_report_{int(time.time())}.txt",
                    mime="text/plain"
                )
            st.success("Report generated successfully!")
        else:
            st.error("Failed to generate report")
            
    except Exception as e:
        st.error(f"Report generation error: {str(e)}")

def reset_analysis():
    st.session_state.current_step = 1
    st.session_state.analysis_data = {}
    st.rerun()

if __name__ == "__main__":
    main()
