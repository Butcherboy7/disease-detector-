

























































import streamlit as st
import requests
import json
import time
import threading
from datetime import datetime
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
    elif st.session_state.current_step == 5:
        follow_up_questions_step()

def input_collection_step():
    st.header("📝 Step 1: Input Collection")
    
    # Input mode selection
    input_mode = st.radio(
        "Choose Input Method:",
        ["🖱️ Guided Form (Recommended)", "📋 JSON Input (Advanced)"],
        horizontal=True
    )
    
    if input_mode == "📋 JSON Input (Advanced)":
        st.subheader("JSON Structured Input")
        st.info("Use this format for structured medical data input:")
        
        example_json = {
            "patient_info": {
                "name": "Mrs. Anjali D.",
                "age": 38,
                "gender": "female",
                "existing_conditions": []
            },
            "symptoms": [
                "fatigue", "headache", "dizziness", "palpitations", "cold hands", "brain fog"
            ],
            "lab_results": {
                "Hemoglobin": "10.1 g/dL",
                "Ferritin": "8 ng/mL",
                "TSH": "5.8 uIU/mL",
                "Vitamin D": "18 ng/mL"
            }
        }
        
        with st.expander("📋 View Example JSON Format"):
            st.json(example_json)
        
        json_input = st.text_area(
            "Enter your structured data as JSON:",
            height=300,
            placeholder=json.dumps(example_json, indent=2)
        )
        
        if st.button("🧠 Analyze JSON Input", type="primary", use_container_width=True):
            try:
                # Parse JSON input
                structured_data = json.loads(json_input)
                
                # Validate required fields
                if not all(key in structured_data for key in ['patient_info', 'symptoms']):
                    st.error("JSON must contain 'patient_info' and 'symptoms' fields")
                    return
                
                # Process directly with medical analysis
                from utils.medical_analysis_engine import MedicalAnalysisEngine
                analysis_engine = MedicalAnalysisEngine()
                
                with st.spinner("Processing structured input..."):
                    medical_result = analysis_engine.analyze_medical_data(structured_data)
                    
                    # Store results and skip to results display
                    st.session_state.analysis_data = {
                        'input_data': structured_data,
                        'medical_analysis': medical_result,
                        'prediction_result': {
                            'medical_assessment': medical_result,
                            'predictions': [],
                            'overall_risk': {'level': 'Low', 'score': 0.1}
                        },
                        'explanation_result': {}
                    }
                    
                    st.session_state.current_step = 4  # Skip to results
                    st.success("✅ JSON input processed successfully!")
                    time.sleep(1)
                    st.rerun()
                    
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {str(e)}")
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
        
        return
    
    # Guided Form Input (Original)
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
    
    # Quick PDF Analysis
    if uploaded_files:
        st.subheader("🔍 Quick PDF Analysis")
        st.info("Upload a PDF medical report to automatically extract symptoms and lab results")
        
        if st.button("🧠 Analyze PDF Report", type="secondary"):
            for file in uploaded_files:
                if file.type == "application/pdf":
                    try:
                        from utils.simple_pdf_extractor import SimplePDFExtractor
                        extractor = SimplePDFExtractor()
                        
                        with st.spinner(f"Analyzing {file.name}..."):
                            extracted_data = extractor.extract_from_pdf(file)
                            
                            if 'error' in extracted_data:
                                st.error(f"Could not extract from {file.name}: {extracted_data['error']}")
                                continue
                            
                            # Process with medical analysis
                            from utils.medical_analysis_engine import MedicalAnalysisEngine
                            analysis_engine = MedicalAnalysisEngine()
                            medical_result = analysis_engine.analyze_medical_data(extracted_data)
                            
                            # Store results and go to analysis display
                            st.session_state.analysis_data = {
                                'input_data': extracted_data,
                                'medical_analysis': medical_result,
                                'pdf_extraction': extracted_data,
                                'prediction_result': {
                                    'medical_assessment': medical_result.get('medical_assessment', {}),
                                    'predictions': [],
                                    'overall_risk': {'level': 'Unknown', 'score': 0.0}
                                },
                                'explanation_result': {}
                            }
                            
                            st.session_state.current_step = 4  # Skip to results
                            st.success(f"✅ Successfully analyzed {file.name}")
                            time.sleep(1)
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"Analysis failed for {file.name}: {str(e)}")
                        continue
    
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
    
    # Display symptom clustering analysis with enhanced visualization
    if 'symptom_cluster_analysis' in prediction:
        cluster_analysis = prediction['symptom_cluster_analysis']
        
        st.subheader("🎯 Symptom Pattern Analysis")
        
        # Show confidence summary
        confidence_summary = cluster_analysis.get('confidence_summary', '')
        if confidence_summary:
            st.info(f"**Analysis Summary:** {confidence_summary}")
        
        # Create symptom clusters visualization
        if 'cluster_analysis' in cluster_analysis:
            cluster_data = cluster_analysis['cluster_analysis']
            
            # Create bar chart data for visualization
            cluster_viz_data = {}
            for cluster_name, confidence in cluster_data.items():
                if confidence > 20:  # Only show meaningful clusters
                    display_name = cluster_name.replace('_', ' ').title()
                    if 'syndrome' in display_name:
                        display_name = display_name.replace(' Syndrome', '')
                    cluster_viz_data[display_name] = round(confidence, 1)
            
            if cluster_viz_data:
                st.write("**Disease Cluster Confidence Scores:**")
                
                # Create columns for visualization
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display as progress bars
                    for cluster_name, confidence in sorted(cluster_viz_data.items(), key=lambda x: x[1], reverse=True):
                        # Color coding based on confidence
                        if confidence > 70:
                            color = "🔴"  # High confidence - red
                        elif confidence > 50:
                            color = "🟡"  # Medium confidence - yellow
                        else:
                            color = "🟢"  # Lower confidence - green
                        
                        st.write(f"{color} **{cluster_name}**: {confidence}%")
                        st.progress(confidence / 100)
                
                with col2:
                    # Summary box
                    st.markdown("### Summary")
                    top_cluster = max(cluster_viz_data.items(), key=lambda x: x[1])
                    st.write(f"**Primary Pattern:**")
                    st.write(f"{top_cluster[0]}")
                    st.write(f"**Confidence:** {top_cluster[1]}%")
        
        # Show top clusters
        if 'top_clusters' in cluster_analysis and cluster_analysis['top_clusters']:
            st.write("**Key Pattern Matches:**")
            for cluster_name, confidence in cluster_analysis['top_clusters']:
                if confidence > 30:  # Only show meaningful confidences
                    cluster_display = cluster_name.replace('_', ' ').title()
                    st.write(f"• Clustered symptoms point **{confidence:.0f}%** toward {cluster_display}")
        
        # Show insights
        if 'insights' in cluster_analysis and cluster_analysis['insights']:
            with st.expander("📋 Detailed Pattern Insights"):
                for insight in cluster_analysis['insights']:
                    st.write(f"• {insight}")
    
    # Display Professional Medical Analysis Results
    if 'medical_assessment' in prediction:
        medical_analysis = prediction['medical_assessment']
        
        st.subheader("🩺 Professional Medical Assessment")
        
        # Patient Summary
        if 'patient_summary' in medical_analysis:
            st.info(f"**Patient:** {medical_analysis['patient_summary']}")
        
        # Primary Diagnosis
        primary = medical_analysis.get('primary_diagnosis')
        if primary:
            st.markdown("### 🔍 Most Likely Condition")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**🩺 Primary Diagnosis: {primary['condition_name']}**")
                st.write(f"**Category:** {primary['category']}")
                if primary['evidence']:
                    st.write(f"**Clinical Evidence:** {'; '.join(primary['evidence'])}")
            
            with col2:
                likelihood = primary['likelihood']
                if likelihood in ['Very High', 'High']:
                    st.error(f"🔴 Likelihood: {likelihood}")
                elif likelihood == 'Moderate':
                    st.warning(f"🟡 Likelihood: {likelihood}")
                else:
                    st.info(f"🔵 Likelihood: {likelihood}")
            
            st.write(f"**Specialist Consultation:** {primary['specialist']}")
            st.write(f"**Recommended Tests:** {', '.join(primary['confirmation_tests'])}")
            st.divider()
        
        # Differential Diagnoses
        differential = medical_analysis.get('differential_diagnoses', [])
        if differential:
            st.markdown("### 🤔 Differential Considerations")
            for i, diff in enumerate(differential, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{i}. {diff['condition_name']}**")
                    if diff['evidence']:
                        st.caption(f"Evidence: {'; '.join(diff['evidence'])}")
                with col2:
                    st.write(f"**{diff['likelihood']}**")
        
        # Unlikely Conditions
        unlikely = medical_analysis.get('unlikely_conditions', [])
        if unlikely:
            with st.expander("❌ Conditions Ruled Out"):
                for condition in unlikely:
                    st.write(f"• {condition}")
        
        # Lab Interpretation
        if st.session_state.analysis_data.get('medical_analysis', {}).get('lab_interpretation'):
            lab_interp = st.session_state.analysis_data['medical_analysis']['lab_interpretation']
            st.markdown("### 🧪 Laboratory Findings")
            for lab, interpretation in lab_interp.items():
                st.write(f"• **{lab.upper()}:** {interpretation}")
        
        # Medical Recommendations
        recommendations = medical_analysis.get('recommendations', {})
        if recommendations:
            st.markdown("### 📋 Clinical Recommendations")
            
            if recommendations.get('immediate_tests'):
                st.write("**Immediate Testing:**")
                for test in recommendations['immediate_tests']:
                    st.write(f"• {test}")
            
            if recommendations.get('specialist_referral'):
                st.write("**Specialist Consultation:**")
                for referral in recommendations['specialist_referral']:
                    st.write(f"• {referral}")
            
            if recommendations.get('follow_up'):
                st.write("**Follow-up Plan:**")
                for follow in recommendations['follow_up']:
                    st.write(f"• {follow}")
    
    # Display Professional Medical Report
    elif 'medical_analysis' in st.session_state.analysis_data:
        medical_data = st.session_state.analysis_data['medical_analysis']
        
        if 'medical_assessment' in medical_data:
            assessment = medical_data['medical_assessment']
            
            st.subheader("🩺 Medical Screening Report")
            
            # Display formatted report
            from utils.medical_analysis_engine import MedicalAnalysisEngine
            engine = MedicalAnalysisEngine()
            formatted_report = engine.format_medical_report(medical_data)
            
            st.text_area("Full Medical Report", formatted_report, height=400)
            
            # Download report
            st.download_button(
                label="📄 Download Medical Report",
                data=formatted_report,
                file_name=f"medical_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    # Display lab analysis if available
    if 'lab_analysis' in prediction and prediction['lab_analysis'].get('extracted_values'):
        lab_analysis = prediction['lab_analysis']
        
        st.subheader("🔬 Lab Report Analysis")
        
        # Show lab insights
        if 'lab_insights' in lab_analysis and lab_analysis['lab_insights']:
            for insight in lab_analysis['lab_insights']:
                st.write(f"• {insight}")
        
        # Show risk adjustments
        if 'risk_adjustments' in lab_analysis and lab_analysis['risk_adjustments']:
            st.write("**Risk Adjustments Based on Lab Values:**")
            for disease, adjustment in lab_analysis['risk_adjustments'].items():
                if adjustment != 1.0:
                    adjustment_text = f"{adjustment:.1f}x" if adjustment > 1.0 else f"{adjustment:.1f}x"
                    st.write(f"• {disease.replace('_', ' ').title()}: {adjustment_text} risk multiplier")
    
    # Display predictions
    st.subheader("🎯 Disease Risk Assessment")
    
    if 'predictions' in prediction:
        # Filter out low probability predictions unless they're the new common diseases
        filtered_predictions = []
        for pred in prediction['predictions']:
            prob = pred.get('probability', 0)
            disease = pred.get('disease', '')
            
            # Always show new common diseases and predictions above 10%
            if prob > 0.1 or any(keyword in disease.lower() for keyword in ['anemia', 'thyroid', 'vitamin d', 'autonomic']):
                filtered_predictions.append(pred)
        
        # Sort by probability, but prioritize new conditions
        filtered_predictions.sort(key=lambda x: (x.get('probability', 0) + 
                                               (0.1 if any(keyword in x.get('disease', '').lower() 
                                                          for keyword in ['anemia', 'thyroid', 'vitamin d', 'autonomic']) 
                                                else 0)), reverse=True)
        
        for disease_pred in filtered_predictions:
            disease = disease_pred.get('disease', 'Unknown')
            probability = disease_pred.get('probability', 0)
            risk_level = disease_pred.get('risk_level', 'Unknown')
            severity_level = disease_pred.get('severity_level', 1)
            urgency_level = disease_pred.get('urgency_level', 'low')
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.write(f"**{disease}**")
                # Show if lab adjusted
                if disease_pred.get('lab_adjusted'):
                    original_prob = disease_pred.get('original_probability', probability)
                    st.caption(f"Lab-adjusted (was {original_prob:.1%})")
                
                # Show evidence
                evidence = disease_pred.get('evidence', [])
                if evidence and len(evidence) > 0:
                    key_evidence = evidence[:2]  # Show top 2 evidence points
                    st.caption(f"Key indicators: {', '.join(key_evidence)}")
            
            with col2:
                st.metric("Probability", f"{probability:.1%}")
            
            with col3:
                if risk_level.lower() == 'high':
                    st.error(f"🔴 {risk_level}")
                elif risk_level.lower() == 'medium':
                    st.warning(f"🟡 {risk_level}")
                else:
                    st.success(f"🟢 {risk_level}")
            
            with col4:
                # Show severity and urgency
                if urgency_level == 'moderate':
                    st.warning(f"⚠️ Urgency: {urgency_level.title()}")
                else:
                    st.info(f"ℹ️ Urgency: {urgency_level.title()}")
                
                st.caption(f"Severity Level: {severity_level}")
            
            st.divider()
    
    # Display follow-up questions
    if 'follow_up_questions' in prediction and prediction['follow_up_questions']:
        st.subheader("❓ Follow-up Questions")
        st.info("Answer these 3 quick questions to improve prediction accuracy:")
        
        follow_up_questions = prediction['follow_up_questions']
        if st.button("Answer Follow-up Questions", type="secondary"):
            st.session_state.follow_up_questions = follow_up_questions
            st.session_state.current_step = 5
            st.rerun()
    
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

def follow_up_questions_step():
    st.header("❓ Step 5: Follow-up Questions")
    st.info("Please answer these questions to improve the accuracy of your health analysis:")
    
    if 'follow_up_questions' not in st.session_state:
        st.error("No follow-up questions available.")
        if st.button("Back to Results"):
            st.session_state.current_step = 4
            st.rerun()
        return
    
    questions = st.session_state.follow_up_questions
    answers = {}
    
    for i, question_data in enumerate(questions):
        question = question_data.get('question', '')
        context = question_data.get('context', '')
        q_type = question_data.get('type', 'descriptive')
        
        st.subheader(f"Question {i+1}")
        st.write(f"**{question}**")
        if context:
            st.caption(f"Context: {context}")
        
        if q_type == 'yes_no':
            answer = st.radio(f"Answer {i+1}:", ["Yes", "No"], key=f"q_{i}")
        elif q_type == 'scale':
            answer = st.slider(f"Answer {i+1}:", 1, 10, 5, key=f"q_{i}")
        elif q_type == 'frequency':
            answer = st.selectbox(f"Answer {i+1}:", 
                                ["Never", "Rarely", "Sometimes", "Often", "Always"], 
                                key=f"q_{i}")
        elif q_type == 'choice':
            answer = st.selectbox(f"Answer {i+1}:", 
                                ["Dry cough", "Cough with mucus/phlegm"], 
                                key=f"q_{i}")
        elif q_type == 'duration':
            answer = st.selectbox(f"Answer {i+1}:", 
                                ["Less than 1 week", "1-2 weeks", "2-4 weeks", "1-3 months", "More than 3 months"], 
                                key=f"q_{i}")
        else:  # descriptive
            answer = st.text_area(f"Answer {i+1}:", key=f"q_{i}")
        
        answers[f"question_{i+1}"] = {
            'question': question,
            'answer': answer,
            'type': q_type
        }
        
        st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Update Analysis with Answers", type="primary"):
            # Store answers and trigger re-analysis
            st.session_state.follow_up_answers = answers
            st.info("Thank you! Your answers have been recorded to improve future analysis accuracy.")
            st.session_state.current_step = 4
            st.rerun()
    
    with col2:
        if st.button("⏩ Skip Questions"):
            st.session_state.current_step = 4
            st.rerun()

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
