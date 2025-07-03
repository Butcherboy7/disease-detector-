"""
Flask Server
Backend server for handling agent communication and API endpoints.
Orchestrates the interaction between different AI agents.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
from backend.agent_orchestrator import AgentOrchestrator
from config.settings import FLASK_HOST, FLASK_PORT

app = Flask(__name__)
CORS(app)

# Initialize agent orchestrator
orchestrator = AgentOrchestrator()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'agents_loaded': orchestrator.agents_initialized
    })

@app.route('/api/collect_input', methods=['POST'])
def collect_input():
    """Endpoint for input collection agent."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        result = orchestrator.collect_input(data)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Input collection failed: {str(e)}'
        }), 500

@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """Endpoint for preprocessing agent."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        result = orchestrator.preprocess_data(data)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Preprocessing failed: {str(e)}'
        }), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Endpoint for prediction agent."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        result = orchestrator.make_prediction(data)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/explain', methods=['POST'])
def generate_explanation():
    """Endpoint for explainability agent."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        result = orchestrator.generate_explanation(data)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Explanation generation failed: {str(e)}'
        }), 500

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """Endpoint for report generator agent."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        result = orchestrator.generate_report(data)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Report generation failed: {str(e)}'
        }), 500

@app.route('/api/submit_feedback', methods=['POST'])
def submit_feedback():
    """Endpoint for feedback agent."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        result = orchestrator.submit_feedback(data)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Feedback submission failed: {str(e)}'
        }), 500

@app.route('/api/feedback_analytics', methods=['GET'])
def get_feedback_analytics():
    """Endpoint to get feedback analytics."""
    try:
        analytics = orchestrator.get_feedback_analytics()
        return jsonify({
            'success': True,
            'analytics': analytics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Analytics retrieval failed: {str(e)}'
        }), 500

@app.route('/api/agent_status', methods=['GET'])
def get_agent_status():
    """Get status of all agents."""
    try:
        status = orchestrator.get_agent_status()
        return jsonify({
            'success': True,
            'agent_status': status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Status retrieval failed: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def start_flask_server():
    """Start the Flask server."""
    try:
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        print(f"Failed to start Flask server: {e}")

if __name__ == '__main__':
    start_flask_server()
