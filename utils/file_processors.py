"""
File Processors
Utility classes for processing different types of medical files including
PDFs, images, and other medical documents.
"""

import base64
import io
import logging
from typing import Dict, Any, List, Optional
from PIL import Image
import PyPDF2

class FileProcessor:
    """
    Handles processing of different file types including medical documents,
    images, and reports.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("file_processor")
        self.supported_image_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp']
        self.supported_document_types = ['application/pdf']
        
    def process_image(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process medical images (X-rays, scans, etc.).
        
        Args:
            file_data: Dictionary containing file information and content
            
        Returns:
            Dictionary with processed image information
        """
        try:
            file_type = file_data.get('type', '')
            file_content = file_data.get('content', '')
            
            if file_type not in self.supported_image_types:
                return {
                    'processed': False,
                    'error': f'Unsupported image type: {file_type}',
                    'extracted_text': ''
                }
            
            # Decode base64 image
            image_data = base64.b64decode(file_content)
            
            # Open image with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Extract basic image information
            image_info = {
                'width': image.width,
                'height': image.height,
                'format': image.format,
                'mode': image.mode,
                'size_bytes': len(image_data)
            }
            
            # Analyze image characteristics for medical context
            analysis = self._analyze_medical_image(image, file_data.get('category', 'general'))
            
            return {
                'processed': True,
                'image_info': image_info,
                'medical_analysis': analysis,
                'extracted_text': analysis.get('description', ''),
                'findings': analysis.get('findings', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return {
                'processed': False,
                'error': f'Image processing failed: {str(e)}',
                'extracted_text': ''
            }
    
    def process_pdf(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process PDF medical documents.
        
        Args:
            file_data: Dictionary containing file information and content
            
        Returns:
            Dictionary with extracted text and analysis
        """
        try:
            file_content = file_data.get('content', '')
            
            # Decode base64 PDF
            pdf_data = base64.b64decode(file_content)
            
            # Extract text from PDF
            extracted_text = self._extract_text_from_pdf(pdf_data)
            
            if not extracted_text:
                return {
                    'processed': True,
                    'extracted_text': 'No text could be extracted from the PDF',
                    'page_count': 0,
                    'medical_data': {}
                }
            
            # Analyze extracted text for medical information
            medical_analysis = self._analyze_medical_text(extracted_text)
            
            return {
                'processed': True,
                'extracted_text': extracted_text,
                'text_length': len(extracted_text),
                'medical_data': medical_analysis,
                'document_type': self._classify_medical_document(extracted_text)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            return {
                'processed': False,
                'error': f'PDF processing failed: {str(e)}',
                'extracted_text': ''
            }
    
    def _extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """Extract text content from PDF bytes."""
        try:
            pdf_file = io.BytesIO(pdf_data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = []
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())
            
            return '\n'.join(text_content).strip()
            
        except Exception as e:
            self.logger.warning(f"PDF text extraction failed: {str(e)}")
            return ""
    
    def _analyze_medical_image(self, image: Image.Image, category: str) -> Dict[str, Any]:
        """
        Analyze medical images for relevant characteristics.
        
        Args:
            image: PIL Image object
            category: Category of the medical image (xray, ct, mri, etc.)
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'category': category,
            'findings': [],
            'description': '',
            'quality_assessment': {},
            'recommendations': []
        }
        
        try:
            # Basic image quality assessment
            analysis['quality_assessment'] = {
                'resolution': f"{image.width}x{image.height}",
                'aspect_ratio': round(image.width / image.height, 2),
                'color_mode': image.mode,
                'estimated_quality': self._assess_image_quality(image)
            }
            
            # Category-specific analysis
            if category == 'xray':
                analysis.update(self._analyze_xray_image(image))
            elif category == 'blood_test':
                analysis.update(self._analyze_lab_report_image(image))
            elif category == 'ecg':
                analysis.update(self._analyze_ecg_image(image))
            else:
                analysis.update(self._analyze_general_medical_image(image))
            
        except Exception as e:
            self.logger.warning(f"Image analysis failed: {str(e)}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _assess_image_quality(self, image: Image.Image) -> str:
        """Assess the quality of the medical image."""
        # Simple quality assessment based on resolution and size
        total_pixels = image.width * image.height
        
        if total_pixels > 1000000:  # > 1MP
            return 'high'
        elif total_pixels > 300000:  # > 0.3MP
            return 'medium'
        else:
            return 'low'
    
    def _analyze_xray_image(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze X-ray images for medical insights."""
        return {
            'description': 'X-ray image uploaded for analysis. Professional radiological interpretation is recommended.',
            'findings': [
                'Image requires professional radiological review',
                'Automated analysis limited without specialized AI models'
            ],
            'recommendations': [
                'Consult with a radiologist for proper interpretation',
                'Ensure image quality is sufficient for diagnostic purposes'
            ]
        }
    
    def _analyze_lab_report_image(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze laboratory report images."""
        return {
            'description': 'Laboratory report image uploaded. Manual review of values is recommended.',
            'findings': [
                'Lab report image detected',
                'Text extraction may be needed for detailed analysis'
            ],
            'recommendations': [
                'Manually input critical lab values for better analysis',
                'Ensure all values are clearly visible in the image'
            ]
        }
    
    def _analyze_ecg_image(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze ECG/EKG images."""
        return {
            'description': 'ECG/EKG trace image uploaded. Cardiological interpretation required.',
            'findings': [
                'ECG trace image detected',
                'Professional cardiac interpretation needed'
            ],
            'recommendations': [
                'Have ECG reviewed by a cardiologist',
                'Consider uploading digital ECG data if available'
            ]
        }
    
    def _analyze_general_medical_image(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze general medical images."""
        return {
            'description': 'Medical image uploaded for analysis. Professional medical review recommended.',
            'findings': [
                'Medical image detected',
                'Specific analysis requires specialized tools'
            ],
            'recommendations': [
                'Share with appropriate medical specialist',
                'Ensure image quality meets diagnostic standards'
            ]
        }
    
    def _analyze_medical_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze extracted text from medical documents.
        
        Args:
            text: Extracted text content
            
        Returns:
            Dictionary with medical data analysis
        """
        analysis = {
            'detected_values': {},
            'medical_terms': [],
            'abnormal_findings': [],
            'normal_findings': [],
            'recommendations': []
        }
        
        try:
            # Extract numerical values that might be lab results
            analysis['detected_values'] = self._extract_lab_values(text)
            
            # Extract medical terminology
            analysis['medical_terms'] = self._extract_medical_terms(text)
            
            # Identify abnormal vs normal findings
            analysis['abnormal_findings'] = self._identify_abnormal_findings(text)
            analysis['normal_findings'] = self._identify_normal_findings(text)
            
        except Exception as e:
            self.logger.warning(f"Medical text analysis failed: {str(e)}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _extract_lab_values(self, text: str) -> Dict[str, Any]:
        """Extract laboratory values from text."""
        import re
        
        values = {}
        
        # Common lab value patterns
        patterns = {
            'glucose': r'glucose[:\s]*(\d+\.?\d*)',
            'cholesterol': r'cholesterol[:\s]*(\d+\.?\d*)',
            'blood_pressure': r'(?:bp|blood pressure)[:\s]*(\d+/\d+)',
            'heart_rate': r'(?:hr|heart rate)[:\s]*(\d+)',
            'temperature': r'(?:temp|temperature)[:\s]*(\d+\.?\d*)',
            'weight': r'weight[:\s]*(\d+\.?\d*)',
            'hemoglobin': r'(?:hb|hemoglobin)[:\s]*(\d+\.?\d*)'
        }
        
        text_lower = text.lower()
        
        for value_type, pattern in patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                values[value_type] = matches[0]
        
        return values
    
    def _extract_medical_terms(self, text: str) -> List[str]:
        """Extract medical terminology from text."""
        import re
        
        # Common medical terms to look for
        medical_terms = [
            'diabetes', 'hypertension', 'hyperlipidemia', 'coronary', 'cardiac',
            'pulmonary', 'respiratory', 'neurological', 'gastrointestinal',
            'endocrine', 'metabolic', 'inflammatory', 'infection', 'allergy',
            'chronic', 'acute', 'malignant', 'benign', 'abnormal', 'normal'
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in medical_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return list(set(found_terms))  # Remove duplicates
    
    def _identify_abnormal_findings(self, text: str) -> List[str]:
        """Identify abnormal findings in medical text."""
        abnormal_indicators = [
            'abnormal', 'elevated', 'high', 'low', 'decreased', 'increased',
            'irregular', 'atypical', 'concerning', 'significant'
        ]
        
        findings = []
        text_lower = text.lower()
        
        for indicator in abnormal_indicators:
            if indicator in text_lower:
                # Extract context around the indicator
                import re
                pattern = rf'.{{0,50}}{indicator}.{{0,50}}'
                matches = re.findall(pattern, text_lower)
                findings.extend(matches)
        
        return findings[:5]  # Limit to top 5 findings
    
    def _identify_normal_findings(self, text: str) -> List[str]:
        """Identify normal findings in medical text."""
        normal_indicators = [
            'normal', 'within normal limits', 'unremarkable', 'stable',
            'appropriate', 'regular', 'typical'
        ]
        
        findings = []
        text_lower = text.lower()
        
        for indicator in normal_indicators:
            if indicator in text_lower:
                findings.append(f"Normal finding: {indicator}")
        
        return list(set(findings))  # Remove duplicates
    
    def _classify_medical_document(self, text: str) -> str:
        """Classify the type of medical document based on content."""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['lab', 'laboratory', 'blood test', 'urine']):
            return 'laboratory_report'
        elif any(term in text_lower for term in ['radiology', 'x-ray', 'ct', 'mri', 'ultrasound']):
            return 'imaging_report'
        elif any(term in text_lower for term in ['ecg', 'ekg', 'electrocardiogram']):
            return 'cardiac_report'
        elif any(term in text_lower for term in ['discharge', 'admission', 'hospital']):
            return 'hospital_report'
        elif any(term in text_lower for term in ['consultation', 'visit', 'examination']):
            return 'consultation_note'
        else:
            return 'general_medical_document'
    
    def get_supported_file_types(self) -> Dict[str, List[str]]:
        """Get list of supported file types."""
        return {
            'images': self.supported_image_types,
            'documents': self.supported_document_types
        }
    
    def validate_file(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate uploaded file.
        
        Args:
            file_data: File data to validate
            
        Returns:
            Validation results
        """
        file_type = file_data.get('type', '')
        file_size = file_data.get('size', 0)
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check file type
        all_supported_types = self.supported_image_types + self.supported_document_types
        if file_type not in all_supported_types:
            validation['valid'] = False
            validation['errors'].append(f'Unsupported file type: {file_type}')
        
        # Check file size (limit to 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            validation['valid'] = False
            validation['errors'].append(f'File too large: {file_size} bytes (max {max_size} bytes)')
        
        # Check if file has content
        if not file_data.get('content'):
            validation['valid'] = False
            validation['errors'].append('File has no content')
        
        # Add warnings for image quality
        if file_type in self.supported_image_types and file_size < 50000:  # < 50KB
            validation['warnings'].append('Image file is quite small, quality may be insufficient for analysis')
        
        return validation
