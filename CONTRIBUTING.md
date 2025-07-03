# Contributing to AI Disease Detection System

Thank you for your interest in contributing to the AI Disease Detection System! This document provides guidelines for contributing to the project.

## 🚀 Quick Start for Contributors

### Development Environment Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/ai-disease-detection.git
   cd ai-disease-detection
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Set Up Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## 🔧 Development Guidelines

### Code Style

We follow PEP 8 with some modifications:
- Line length: 120 characters
- Use type hints for all function parameters and return values
- Document all classes and public methods with docstrings
- Use meaningful variable and function names

### Project Structure

```
agents/                 # AI agents (core functionality)
├── base_agent.py      # Abstract base class
├── *_agent.py         # Specific agent implementations
└── __init__.py

utils/                 # Utility modules
├── symptom_clustering.py     # NEW: Symptom analysis
├── lab_report_analyzer.py    # NEW: Lab integration
├── disease_models.py         # ML models
└── api_clients.py           # External API clients

backend/               # Flask server and orchestration
config/                # Configuration management
tests/                 # Test suite
docs/                  # Documentation
```

### Agent Development Guidelines

When adding new agents or modifying existing ones:

1. **Inherit from BaseAgent**
   ```python
   from agents.base_agent import BaseAgent
   
   class NewAgent(BaseAgent):
       def __init__(self):
           super().__init__("NewAgent")
   ```

2. **Implement Required Methods**
   ```python
   def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
       """Main processing method."""
       try:
           # Your logic here
           return self.create_success_response(result_data)
       except Exception as e:
           return self.handle_error(e, "context")
   ```

3. **Use Logging**
   ```python
   self.log_processing_step("Starting important operation")
   ```

4. **Validate Input**
   ```python
   if not self.validate_input(data, ['required_field1', 'required_field2']):
       return self.handle_error(ValueError("Missing required fields"))
   ```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agents --cov=utils --cov=backend

# Run specific test categories
pytest tests/agents/           # Agent tests
pytest tests/integration/     # Integration tests
pytest tests/performance/     # Performance tests
```

### Writing Tests

1. **Agent Tests**
   ```python
   def test_agent_processing():
       agent = YourAgent()
       test_data = {"required_field": "value"}
       result = agent.process(test_data)
       
       assert result["success"] is True
       assert "result_data" in result
   ```

2. **Integration Tests**
   ```python
   def test_full_pipeline():
       # Test complete agent pipeline
       input_data = create_test_input()
       result = run_full_analysis(input_data)
       assert result["predictions"] is not None
   ```

### Test Data

- Use anonymized, synthetic medical data only
- Store test files in `tests/data/`
- Never include real patient information

## 🎯 Areas for Contribution

### High Priority

1. **New Disease Detection Models**
   - Implement detection for additional diseases
   - Improve existing prediction accuracy
   - Add specialized medical domain models

2. **Enhanced Symptom Clustering**
   - Expand disease pattern recognition
   - Add support for more medical specialties
   - Improve clustering confidence algorithms

3. **Lab Report Processing**
   - Support additional lab test types
   - Improve value extraction accuracy
   - Add international unit conversions

### Medium Priority

4. **User Experience**
   - Improve Streamlit interface design
   - Add data visualization features
   - Enhance accessibility compliance

5. **Performance Optimization**
   - Optimize agent communication
   - Reduce memory usage
   - Improve response times

6. **Documentation**
   - Add more usage examples
   - Create video tutorials
   - Expand API documentation

### Low Priority

7. **Infrastructure**
   - Add Docker support
   - Improve CI/CD pipeline
   - Add monitoring and alerting

## 📝 Pull Request Process

### Before Submitting

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write Tests**
   - Add unit tests for new functionality
   - Ensure existing tests still pass
   - Aim for >90% code coverage

3. **Update Documentation**
   - Update README.md if needed
   - Add docstrings to new functions
   - Update replit.md with architectural changes

4. **Lint Your Code**
   ```bash
   black .                    # Format code
   flake8                    # Check style
   mypy agents/ utils/       # Type checking
   ```

### PR Template

When submitting a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated Checks**
   - CI pipeline runs tests
   - Code style validation
   - Security scanning

2. **Manual Review**
   - Code quality assessment
   - Architecture review
   - Medical accuracy validation (if applicable)

3. **Approval Required**
   - At least one maintainer approval
   - All checks must pass
   - Documentation must be updated

## 🛡️ Security Guidelines

### Medical Data Handling

- **Never store real patient data**
- **Use synthetic data for testing**
- **Implement proper data anonymization**
- **Follow HIPAA-like principles**

### API Security

- **Validate all inputs**
- **Use environment variables for secrets**
- **Implement rate limiting**
- **Log security events**

### Code Security

- **No hardcoded credentials**
- **Sanitize file uploads**
- **Validate external API responses**
- **Use secure HTTP headers**

## 🐛 Bug Reports

### Before Reporting

1. **Search Existing Issues**
   - Check if the bug is already reported
   - Look for similar issues or discussions

2. **Reproduce the Bug**
   - Try to reproduce consistently
   - Note the exact steps
   - Check different environments if possible

### Bug Report Template

```markdown
## Bug Description
Clear description of what went wrong

## Steps to Reproduce
1. Go to...
2. Enter...
3. Click...
4. See error

## Expected Behavior
What should have happened

## Actual Behavior
What actually happened

## Environment
- OS: [e.g., Windows 10, macOS 12.0]
- Python version: [e.g., 3.11.0]
- Browser: [e.g., Chrome 96]

## Additional Context
Screenshots, logs, or other relevant information
```

## 🌟 Feature Requests

### Request Template

```markdown
## Feature Description
Clear description of the proposed feature

## Problem Solved
What problem does this solve?

## Proposed Solution
How should this work?

## Alternatives Considered
Other solutions you've considered

## Additional Context
Any other relevant information
```

### Evaluation Criteria

Features are evaluated based on:
- **Medical accuracy impact**
- **User experience improvement**
- **Implementation complexity**
- **Maintenance overhead**
- **Community benefit**

## 📚 Documentation Guidelines

### Code Documentation

- **Docstrings**: All public functions must have docstrings
- **Type Hints**: Use comprehensive type annotations
- **Comments**: Explain complex logic, not obvious code
- **Examples**: Include usage examples in docstrings

### User Documentation

- **Clear Language**: Write for non-technical healthcare users
- **Step-by-Step**: Break down complex processes
- **Screenshots**: Include visual guides where helpful
- **Medical Context**: Explain medical relevance when applicable

## 🎓 Learning Resources

### Medical Informatics
- [Healthcare Data Standards](https://www.hl7.org/)
- [Medical Terminology](https://www.nlm.nih.gov/research/umls/)
- [Clinical Decision Support](https://www.ahrq.gov/cds/index.html)

### AI/ML in Healthcare
- [Medical AI Ethics](https://www.who.int/publications/i/item/ethics-and-governance-of-artificial-intelligence-for-health)
- [Clinical ML Guidelines](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device)

### Technical Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenAI API Guide](https://platform.openai.com/docs/)

## 🤝 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be Respectful**: Treat all contributors with respect
- **Be Collaborative**: Work together towards common goals  
- **Be Patient**: Help newcomers learn and contribute
- **Be Professional**: Maintain professional standards

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code review and technical discussion
- **Email**: contact@ai-disease-detection.com for sensitive issues

### Recognition

Contributors are recognized through:
- **Contributors.md**: Listed in project contributors
- **Release Notes**: Acknowledged in version releases
- **GitHub**: Contributor statistics and badges

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the AI Disease Detection System! Your efforts help improve healthcare technology for everyone.