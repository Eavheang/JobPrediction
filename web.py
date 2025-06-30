import os
import re
import base64
import tempfile
from io import BytesIO
import pickle
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import fitz  # PyMuPDF
import PyPDF2
import pytesseract
import spacy
import shutil
import platform

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
# Load the dataset
data = pd.read_csv(r'Dataset\resume_dataset - gpt_dataset.csv')

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
gemini_vision_model = genai.GenerativeModel('gemini-2.5-pro')
gemini_text_model = genai.GenerativeModel('gemini-2.5-pro')

# Enhanced text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s\+\#\.]+', ' ', text)  # Preserve +, #, . for C++, C#, etc.
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens).strip()

def extract_sections(text):
    sections = {'skills': [], 'experience': [], 'education': []}
    current_section = None
    section_patterns = {
        'skills': re.compile(r'\bskills?\b|\btechnical\s+skills?\b', re.I),
        'experience': re.compile(r'\b(work\s+)?experience\b|\bemployment\b', re.I),
        'education': re.compile(r'\beducation\b|\bacademic\b', re.I)
    }

    for line in text.split('\n'):
        line = line.strip().lower()
        if not line:
            continue
        for section, pattern in section_patterns.items():
            if pattern.search(line):
                current_section = section
                break
        else:
            if current_section:
                sections[current_section].append(line)

    for section in sections:
        sections[section] = [s for s in sections[section] if len(s.split()) > 2]
    return sections

# Apply cleaning and preprocessing first
resume_column = 'Resume'
if resume_column not in data.columns:
    raise KeyError(f"Column '{resume_column}' not found. Available columns: {data.columns.tolist()}")
data['cleaned_resume'] = data[resume_column].apply(clean_text)
data['sections'] = data['cleaned_resume'].apply(extract_sections)

# Encode labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['Category'])
num_classes = len(label_encoder.classes_)

# Handle class imbalance after cleaning
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(data['cleaned_resume']).toarray()
y = data['label']
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X, y)

# Define LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and DistilBERT model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
model.to(device)

# Initialize LSTM
input_dim = X_balanced.shape[1]
hidden_dim = 128
nn_model = LSTMClassifier(input_dim, hidden_dim, num_classes).to(device)

def check_poppler_installation():
    """Check if Poppler is installed and accessible."""
    print("\n=== Checking Poppler Installation ===")
    if platform.system() == "Windows":
        poppler_path = shutil.which("pdftoppm")  # This is a Poppler binary
        if not poppler_path:
            print("WARNING: Poppler is not installed or not in PATH!")
            print("To use Gemini AI extraction, please install Poppler:")
            print("1. Download Poppler for Windows from: https://github.com/oschwartz10612/poppler-windows/releases/")
            print("2. Extract the downloaded file")
            print("3. Add the extracted 'poppler-xx/Library/bin' directory to your system PATH")
            print("4. Restart your application")
            return False
        print("Poppler is installed and accessible")
        return True
    return True  # For non-Windows systems, assume it's handled by package manager

def extract_text_with_gemini(file_path):
    """Extract text from PDF or image using Gemini AI."""
    print(f"\n=== Starting Gemini AI Text Extraction ===")
    print(f"Processing file: {file_path}")
    try:
        if file_path.lower().endswith('.pdf'):
            print("Detected PDF file - converting to images...")
            # Convert PDF to images using PyMuPDF
            pdf_document = fitz.open(file_path)
            extracted_text = []
            
            for page_num in range(pdf_document.page_count):
                print(f"\nProcessing page {page_num + 1}/{pdf_document.page_count}")
                page = pdf_document[page_num]
                
                # Get the page as an image
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                
                # Create Gemini image input
                image_parts = [
                    {
                        "mime_type": "image/png",
                        "data": base64.b64encode(img_data).decode('utf-8')
                    }
                ]
                
                # Generate prompt for Gemini
                prompt = """Please extract all text content from this resume image. 
                Focus on:
                1. Contact information
                2. Professional summary
                3. Work experience
                4. Education
                5. Skills
                6. Certifications
                7. Projects
                
                Format the text in a clean, structured way and preserve all important information."""
                
                print("Sending request to Gemini AI...")
                # Get response from Gemini
                response = gemini_vision_model.generate_content([prompt, image_parts[0]])
                print(f"Received response from Gemini AI for page {page_num + 1}")
                print("Extracted text length:", len(response.text))
                print("First 200 characters of extracted text:", response.text[:200])
                extracted_text.append(response.text)
            
            pdf_document.close()
            final_text = "\n\n".join(extracted_text)
            print("\n=== Completed Gemini AI Text Extraction ===")
            print(f"Total extracted text length: {len(final_text)}")
            return final_text
            
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print("Detected image file")
            # Load and process image
            img = Image.open(file_path)
            
            # Convert to bytes
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create Gemini image input
            image_parts = [
                {
                    "mime_type": "image/png",
                    "data": base64.b64encode(img_byte_arr).decode('utf-8')
                }
            ]
            
            # Generate prompt for Gemini
            prompt = """Please extract all text content from this resume image. 
            Focus on:
            1. Contact information
            2. Professional summary
            3. Work experience
            4. Education
            5. Skills
            6. Certifications
            7. Projects
            
            Format the text in a clean, structured way and preserve all important information."""
            
            print("Sending request to Gemini AI...")
            # Get response from Gemini
            response = gemini_vision_model.generate_content([prompt, image_parts[0]])
            print("Received response from Gemini AI")
            print("Extracted text length:", len(response.text))
            print("First 200 characters of extracted text:", response.text[:200])
            return response.text
            
    except Exception as e:
        print(f"Error in Gemini extraction: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def enhance_text_with_gemini(text):
    """Use Gemini to enhance and structure the extracted text."""
    print("\n=== Starting Text Enhancement with Gemini ===")
    try:
        prompt = f"""Please analyze and enhance this resume text. 
        1. Structure the content clearly
        2. Identify and categorize key skills
        3. Highlight important achievements
        4. Extract years of experience
        5. Identify key technologies and tools
        
        Original text:
        {text}
        """
        
        print("Sending enhancement request to Gemini AI...")
        response = gemini_text_model.generate_content(prompt)
        print("Received enhanced text from Gemini AI")
        print("Enhanced text length:", len(response.text))
        print("First 200 characters of enhanced text:", response.text[:200])
        print("=== Text Enhancement Complete ===\n")
        return response.text
    except Exception as e:
        print(f"Error in Gemini enhancement: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return text

# Modify the process_uploaded_resume function to use Gemini
def process_uploaded_resume(file_path):
    print(f"\n=== Starting Resume Processing ===")
    print(f"Processing file: {file_path}")
    
    # Check Poppler installation before attempting Gemini extraction
    has_poppler = check_poppler_installation()
    
    # First try Gemini extraction
    print("Attempting Gemini AI extraction...")
    if has_poppler or not file_path.lower().endswith('.pdf'):
        gemini_text = extract_text_with_gemini(file_path)
    else:
        print("Skipping Gemini extraction due to missing Poppler installation")
        gemini_text = None
    
    if gemini_text and len(gemini_text.strip()) > 0:
        print("Gemini AI extraction successful")
        print("Enhancing extracted text with Gemini...")
        # Enhance the extracted text with Gemini
        enhanced_text = enhance_text_with_gemini(gemini_text)
        print("Text enhancement complete")
        print("Cleaning and extracting sections...")
        cleaned_text = clean_text(enhanced_text)
        sections = extract_sections(enhanced_text)
        print("=== Resume Processing Complete (Using Gemini) ===\n")
        return cleaned_text, sections
    
    # Fallback to original extraction methods if Gemini fails
    print("Gemini AI extraction failed or returned empty text, falling back to traditional methods...")
    text = None
    if file_path.lower().endswith('.pdf'):
        print("Attempting PDF extraction with PyPDF2/OCR...")
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        print("Attempting image extraction with Tesseract OCR...")
        text = extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file format. Upload PDF, JPG, JPEG, or PNG.")

    if text and len(text.strip()) > 0:
        print("Traditional extraction successful")
        print("Cleaning and extracting sections...")
        cleaned_text = clean_text(text)
        sections = extract_sections(cleaned_text)
        print("=== Resume Processing Complete (Using Traditional Methods) ===\n")
        return cleaned_text, sections
    else:
        print("Text extraction resulted in empty or whitespace-only text.")
        raise ValueError("Failed to extract text or extracted text is empty.")

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file with enhanced processing and fallback OCR method."""
    section_keywords = [
        'EDUCATION', 'EXPERIENCE', 'SKILLS', 'PROJECTS', 'CERTIFICATIONS',
        'ACHIEVEMENTS', 'PUBLICATIONS', 'REFERENCES', 'SUMMARY', 'OBJECTIVE',
        'WORK EXPERIENCE', 'PROFESSIONAL EXPERIENCE', 'TECHNICAL SKILLS',
        'EDUCATION AND TRAINING', 'PROFESSIONAL SUMMARY', 'CAREER OBJECTIVE'
    ]
    
    def clean_and_normalize_text(text):
        """Clean and normalize extracted text."""
        # Replace common unicode characters
        replacements = {
            '•': '* ',    # Bullet points
            '–': '-',     # En-dash
            '—': '-',     # Em-dash
            '"': '"',     # Smart quotes
            '"': '"',     # Smart quotes
            "'": "'",     # Smart apostrophes
            '…': '...',   # Ellipsis
            '\uf0b7': '*',  # Another bullet point variant
            '\u2022': '*',  # Another bullet point variant
            '\u2023': '*',  # Another bullet point variant
            '\u2043': '*',  # Another bullet point variant
            '\u204C': '*',  # Another bullet point variant
            '\u204D': '*',  # Another bullet point variant
            '\u2219': '*',  # Another bullet point variant
            '\u25E6': '*',  # Another bullet point variant
            '\u2043': '*',  # Another bullet point variant
            '\xa0': ' ',    # Non-breaking space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove multiple spaces and normalize whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR mistakes
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between camelCase
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Add space between letters and numbers
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Add space between numbers and letters
        
        return text

    def detect_sections(text):
        """Detect and format resume sections."""
        formatted_text = text
        for keyword in section_keywords:
            # Match section headers with flexible spacing and formatting
            pattern = f'([^\n])?({keyword})([^\n])?'
            replacement = f'\\1\n\n{keyword.upper()}\n\\3'
            formatted_text = re.sub(pattern, replacement, formatted_text, flags=re.IGNORECASE)
        
        # Clean up excessive newlines
        formatted_text = re.sub(r'\n\s*\n\s*\n', '\n\n', formatted_text)
        return formatted_text

    def extract_with_pdfplumber():
        """Extract text using pdfplumber for better layout analysis."""
        try:
            import pdfplumber
            text_parts = []
            
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    # Extract text with layout analysis
                    text = page.extract_text(
                        x_tolerance=3,  # Adjust for horizontal text alignment
                        y_tolerance=3,  # Adjust for vertical text alignment
                    )
                    if text:
                        text_parts.append(text)
            
            return '\n\n'.join(text_parts)
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
            return None

    try:
        # Try primary extraction method (PyPDF2)
        reader = PyPDF2.PdfReader(pdf_file)
        text_parts = []
        
        for page in reader.pages:
            page_text = page.extract_text()
            if not page_text:
                continue
            
            # Clean up common PDF extraction issues
            lines = [line.strip() for line in page_text.split('\n') if line.strip()]
            cleaned_lines = []
            
            for i, line in enumerate(lines):
                # Handle hyphenation at line breaks
                if cleaned_lines and cleaned_lines[-1].endswith('-'):
                    cleaned_lines[-1] = cleaned_lines[-1][:-1] + line
                    continue
                
                # Handle line continuation
                if i > 0 and not any(keyword.lower() in line.lower() for keyword in section_keywords):
                    prev_line = cleaned_lines[-1] if cleaned_lines else ""
                    # Check if this line continues the previous sentence
                    if (prev_line and 
                        not prev_line.endswith(('.', '!', '?', ':', ';')) and
                        not line[0].isupper() and
                        not line.startswith(('•', '*', '-', '–', '—'))):
                        cleaned_lines[-1] = f"{prev_line} {line}"
                        continue
                
                cleaned_lines.append(line)
            
            processed_text = '\n'.join(cleaned_lines)
            processed_text = clean_and_normalize_text(processed_text)
            text_parts.append(processed_text)
        
        final_text = '\n\n'.join(text_parts)
        
        # If PyPDF2 extraction yields poor results, try pdfplumber
        if len(final_text.strip()) < 100:  # Arbitrary threshold for "poor results"
            plumber_text = extract_with_pdfplumber()
            if plumber_text and len(plumber_text.strip()) > len(final_text.strip()):
                final_text = plumber_text
        
        # Post-processing
        final_text = detect_sections(final_text)
        
        # If both primary methods fail, try OCR
        if len(final_text.strip()) < 100:
            raise Exception("Primary extraction methods yielded insufficient text")
        
        return final_text.strip()
        
    except Exception as e:
        print(f"Primary extraction methods failed: {e}")
        print("Attempting OCR extraction method...")
        
        try:
            # Enhanced OCR method
            images = fitz.open(pdf_file)
            text_parts = []
            
            for page in images:
                # Convert to grayscale
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x zoom for better quality
                gray = pix.tobytes("png")
                
                # Enhanced image preprocessing
                # 1. Denoise
                denoised = cv2.fastNlMeansDenoising(gray)
                
                # 2. Increase contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                contrasted = clahe.apply(denoised)
                
                # 3. Thresholding
                thresh = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # 4. Remove small noise
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
                # Perform OCR with optimized settings
                custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                text = pytesseract.image_to_string(cleaned, config=custom_config)
                
                if text.strip():
                    # Clean and normalize OCR output
                    text = clean_and_normalize_text(text)
                    text_parts.append(text)
            
            final_text = '\n\n'.join(text_parts)
            final_text = detect_sections(final_text)
            
            return final_text.strip()
            
        except Exception as fallback_error:
            print(f"OCR extraction failed: {fallback_error}")
            return None

def extract_text_from_image(image_file):
    try:
        img = Image.open(image_file)
        pix = img.convert("L")  # Convert to grayscale
        text = pytesseract.image_to_string(pix, config='--psm 6')
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return None

def generate_pdf_preview(file_path):
    try:
        if file_path.lower().endswith('.pdf'):
            # Use PyMuPDF to generate preview
            pdf_document = fitz.open(file_path)
            if pdf_document.page_count > 0:
                page = pdf_document[0]  # Get first page
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                pdf_document.close()
                img_str = base64.b64encode(img_data).decode('utf-8')
                return f"data:image/png;base64,{img_str}"
        return None
    except Exception as e:
        print(f"Error generating PDF preview: {e}")
        return None

def load_trained_models(lstm_path='checkpoints/lstm_epoch_150.pt', distilbert_path='checkpoints/distilbert_epoch_150.pt'):
    
    # Load LSTM model
    lstm_model = LSTMClassifier(input_dim=5000, hidden_dim=128, output_dim=num_classes)
    lstm_checkpoint = torch.load(lstm_path, map_location=device)
    lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])
    lstm_model.to(device)
    lstm_model.eval()

    # Load DistilBERT model
    distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
    distilbert_checkpoint = torch.load(distilbert_path, map_location=device)
    distilbert_model.load_state_dict(distilbert_checkpoint['model_state_dict'])
    distilbert_model.to(device)
    distilbert_model.eval()

    return lstm_model, distilbert_model

def predict_job_role(resume_text, sections, distilbert_model, lstm_model, tokenizer, tfidf, label_encoder, device):
    try:
        # Process for DistilBERT
        encoding = tokenizer.encode_plus(
            resume_text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # DistilBERT prediction
        with torch.no_grad():
            outputs = distilbert_model(input_ids, attention_mask=attention_mask)
            distilbert_probs = torch.softmax(outputs.logits, dim=1)

        # LSTM prediction
        tfidf_vector = tfidf.transform([resume_text]).toarray()
        lstm_input = torch.FloatTensor(tfidf_vector).unsqueeze(0).to(device)
        with torch.no_grad():
            lstm_output = lstm_model(lstm_input)
            lstm_probs = torch.softmax(lstm_output, dim=1)
        
        # Weighted ensemble prediction
        distilbert_weight = 0.7
        lstm_weight = 0.3
        ensemble_probs = distilbert_weight * distilbert_probs + lstm_weight * lstm_probs
        predicted_idx = torch.argmax(ensemble_probs, dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
        confidence = ensemble_probs[0][predicted_idx].item()

        # Get top skills based on TF-IDF scores
        key_terms_with_scores = extract_key_skills(resume_text, tfidf)
        key_terms = [term for term, _ in key_terms_with_scores[:5]]  # Top 5 skills

        # Format sections for explanation
        formatted_sections = {
            "skills": [skill.strip() for skill in sections.get('skills', [])[:5]],
            "experience": [exp.strip() for exp in sections.get('experience', [])[:3]],
            "education": [edu.strip() for edu in sections.get('education', [])[:2]]
        }

        # Create explanation structure
        explanation = {
            "predicted_role": predicted_label,
            "confidence": confidence,
            "key_terms": key_terms,
            "sections": formatted_sections,
            "model_info": "DistilBERT + LSTM Ensemble"
        }

        return predicted_label, confidence, explanation, "DistilBERT + LSTM Ensemble"
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

def enhance_prediction_with_gemini(job_role, confidence, resume_text, sections, explanation):
    """
    Enhance the prediction results using Gemini AI to provide a concise, HR-friendly analysis.
    """
    print("\n=== Enhancing Prediction with Gemini AI ===")
    try:
        print("\n=== Starting Alternative Careers Analysis ===")
        print("Current Role Prediction:", job_role)
        print("Confidence Score:", f"{confidence*100:.1f}%")
        
        # Calculate remaining percentage for alternative roles
        remaining_percentage = (1 - confidence) * 100
        print(f"\nRemaining percentage for alternative roles: {remaining_percentage:.1f}%")

        # First, let's have Gemini suggest alternative careers
        alternative_careers_prompt = f"""Based on this resume content, suggest 2-3 alternative career paths that would be suitable for this candidate.

Resume Content:
{resume_text}

Current Skills: {', '.join(sections.get('skills', []))}
Current Experience: {', '.join(sections.get('experience', []))}
Education: {', '.join(sections.get('education', []))}

Important Note: The candidate has been predicted as {job_role} with {confidence*100:.1f}% confidence.
The remaining {remaining_percentage:.1f}% should be distributed among your alternative career suggestions.
Total of all alternative career matches should not exceed {remaining_percentage:.1f}%.

Format each suggestion EXACTLY as follows (with blank line between each role):

Role
Match: [percentage]%
Reasoning: [2-3 sentences explaining the fit based on specific skills/experience]

Only suggest roles that are genuinely suitable based on their background. Use precise percentages (e.g., 10.1%, 6.0%)."""

        print("\n=== Sending Alternative Careers Request to Gemini AI ===")
        print("Prompt length:", len(alternative_careers_prompt))
        
        alt_careers_response = gemini_text_model.generate_content(alternative_careers_prompt)
        alternative_careers = alt_careers_response.text
        
        print("\n=== Received Alternative Careers Response ===")
        print("Response length:", len(alternative_careers))
        print("\nAlternative Careers Suggestions:")
        print("-" * 50)
        print(alternative_careers)
        print("-" * 50)

        # Main analysis prompt
        prompt = f"""Analyze this resume for the role of {job_role} (predicted with {confidence*100:.1f}% confidence) and provide a brief analysis in the following format:

Resume Content:
{resume_text}

Key Sections:
Skills: {', '.join(sections.get('skills', [])[:10])}
Experience: {', '.join(sections.get('experience', [])[:3])}
Education: {', '.join(sections.get('education', [])[:2])}

Please provide your analysis in this exact format:

Overall Match:
[Write 2-3 sentences about why this candidate matches the role]

Alternative Career Paths:
Here are some alternative career paths that would be suitable based on your skills and experience:

Product Manager
Match: 14.1%
Reasoning: The candidate has demonstrated strong product-oriented thinking by leading projects from "concept to launch," managing user feedback, and winning entrepreneurial competitions like "Smart Spark+". Their versatile technical background in web, mobile, and IoT provides the breadth of knowledge needed to effectively manage and communicate with diverse development teams. This experience shows a clear ability to turn an "idea into an impactful solution."

UI/UX Designer
Match: 8.0%
Reasoning: The resume explicitly lists "UI/UX Design" as a key skill and details experience using Figma to develop interfaces for both mobile and web applications. This indicates a practical ability and interest in creating user-friendly products. Their background as a developer would allow them to create designs that are both aesthetically pleasing and technically feasible to implement.

IoT Engineer
Match: 8.0%
Reasoning: The candidate successfully led an "automate rover robot project," showcasing specific expertise in IoT devices, Arduino, and C++ programming. This hands-on experience in hardware integration and project management is a strong foundation for a career in the Internet of Things. This role directly leverages a unique and standout project on their resume.

Skills:
[Describe the candidate's relevant skills and how they align with the role]

Experience:
[Highlight the most relevant work experience and achievements and show the duration of the experience (Example: Frontend developer for 2 moths from Jan-Feb 2025)]

Education:
[Discuss the relevance of their educational background]

Questions for Candidate:
[Give 5 questions for HR to use for asking the candidate during interviews]

Keep each section concise and focused on what's most relevant for HR review. Use clear paragraph breaks between sections."""

        print("\n=== Sending Main Analysis Request to Gemini AI ===")
        print("Main prompt length:", len(prompt))
        
        response = gemini_text_model.generate_content(prompt)
        enhanced_analysis = response.text
        
        print("\n=== Received Main Analysis Response ===")
        print("Main analysis length:", len(enhanced_analysis))
        print("\nAnalysis Sections Found:")
        sections_found = [section.split(':')[0] for section in enhanced_analysis.split('\n\n') if ':' in section]
        print('\n'.join(f"- {section}" for section in sections_found))
        
        print("\n=== Enhancement Process Complete ===")
        return enhanced_analysis
        
    except Exception as e:
        print(f"\n=== ERROR in Gemini Enhancement ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Full traceback:")
        import traceback
        print(traceback.format_exc())
        return None

# Update the HTML/JavaScript part to handle the new format
def format_analysis_text(text):
    """Format the analysis text with proper styling"""
    if not text:
        return ""
    
    # Split the text into sections
    sections = text.split('\n\n')
    formatted_sections = []
    
    for section in sections:
        if ':' in section:
            # Get the title and content
            title, content = section.split(':', 1)
            formatted_sections.append(f"""
                <div class="mb-6">
                    <h4 class="text-xl font-semibold text-gray-900 mb-3">{title.strip()}</h4>
                    <p class="text-gray-700 leading-relaxed">{content.strip()}</p>
                </div>
            """)
    
    return '\n'.join(formatted_sections)

from flask import Flask, request, jsonify
import os
from pyngrok import ngrok
import traceback

app = Flask(__name__)

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Job Role Prediction</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            margin: 0;
            padding: 0;
        }

        .glass-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }

        .main-container {
            max-width: 1000px;
            margin: 2rem auto;
            padding: 3rem;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 32px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(16px);
        }

        .gradient-text {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .drop-zone {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
            border: 2px dashed rgba(102, 126, 234, 0.3);
        }

        .drop-zone:hover {
            transform: translateY(-4px);
            border-color: #667eea;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }

        .drop-zone.drag-over {
            border-color: #667eea;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
            transform: scale(1.02);
        }

        #preview, #pdfCanvas {
            max-height: 400px;
            object-fit: contain;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15);
            border: 4px solid rgba(255, 255, 255, 0.8);
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px 0 rgba(102, 126, 234, 0.5);
        }

        .btn-primary:active {
            transform: translateY(0);
        }

        .loading-spinner {
            border: 4px solid rgba(102, 126, 234, 0.2);
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 3rem;
            height: 3rem;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .result-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
            border: 1px solid rgba(102, 126, 234, 0.2);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.05);
        }

        .confidence-bar {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 999px;
            height: 8px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .feature-icon {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            margin-bottom: 1rem;
        }

        .floating-elements {
            position: absolute;
            width: 100%;
            height: 100%;
            overflow: hidden;
            pointer-events: none;
        }

        .floating-circle {
            position: absolute;
            border-radius: 50%;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            animation: float 6s ease-in-out infinite;
        }

        .floating-circle:nth-child(1) {
            width: 80px;
            height: 80px;
            top: 20%;
            left: 10%;
            animation-delay: 0s;
        }

        .floating-circle:nth-child(2) {
            width: 120px;
            height: 120px;
            top: 60%;
            right: 10%;
            animation-delay: 2s;
        }

        .floating-circle:nth-child(3) {
            width: 60px;
            height: 60px;
            top: 80%;
            left: 20%;
            animation-delay: 4s;
        }

        @keyframes float {
            0%, 100% {
                transform: translateY(0px);
            }
            50% {
                transform: translateY(-20px);
            }
        }

        .fade-in {
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
</head>
<body class="min-h-screen py-8 px-4 sm:px-6 lg:px-8 relative">
    <div class="floating-elements">
        <div class="floating-circle"></div>
        <div class="floating-circle"></div>
        <div class="floating-circle"></div>
    </div>

    <div class="main-container fade-in">
        <div class="text-center mb-16">
            <h1 class="text-6xl font-extrabold gradient-text mb-6">AI Job Role Predictor</h1>
            <p class="text-xl text-gray-600 max-w-2xl mx-auto leading-relaxed">
                Discover your perfect career path with cutting-edge AI analysis. Upload your resume and let our advanced machine learning models predict your ideal job role.
            </p>
        </div>

        <!-- Features Section -->
        <div class="grid md:grid-cols-3 gap-8 mb-12">
            <div class="text-center">
                <div class="feature-icon mx-auto">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                    </svg>
                </div>
                <h3 class="text-lg font-semibold text-gray-900 mb-2">Lightning Fast</h3>
                <p class="text-gray-600 text-sm">Get instant predictions powered by DistilBERT and LSTM models</p>
            </div>
            <div class="text-center">
                <div class="feature-icon mx-auto">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                </div>
                <h3 class="text-lg font-semibold text-gray-900 mb-2">High Accuracy</h3>
                <p class="text-gray-600 text-sm">Advanced AI models trained on thousands of resumes</p>
            </div>
            <div class="text-center">
                <div class="feature-icon mx-auto">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
                    </svg>
                </div>
                <h3 class="text-lg font-semibold text-gray-900 mb-2">Secure & Private</h3>
                <p class="text-gray-600 text-sm">Your data is processed securely and never stored</p>
            </div>
        </div>

        <div class="glass-container p-8">
            <form id="uploadForm" class="mb-8" enctype="multipart/form-data">
                <div class="space-y-8">
                    <div class="flex justify-center">
                        <div class="w-full max-w-2xl">
                            <label class="block text-lg font-semibold text-gray-800 mb-4 text-center">Upload Your Resume</label>
                            <div id="dropZone" class="drop-zone mt-1 flex justify-center px-8 pt-8 pb-8 rounded-2xl cursor-pointer">
                                <div class="space-y-4 text-center">
                                    <div class="mx-auto h-16 w-16 text-gray-400">
                                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                                        </svg>
                                    </div>
                                    <div class="text-lg text-gray-700">
                                        <label for="file-upload" class="relative cursor-pointer font-semibold text-indigo-600 hover:text-indigo-500 transition-colors">
                                            <span>Choose a file</span>
                                            <input id="file-upload" name="file" type="file" class="sr-only" accept="image/*,application/pdf">
                                        </label>
                                        <span class="text-gray-500"> or drag and drop</span>
                                    </div>
                                    <p class="text-sm text-gray-500">PDF, PNG, JPG • Up to 10MB</p>
                                </div>
                            </div>
                            <div id="previewContainer" class="mt-6 flex justify-center">
                                <img id="preview" class="hidden" alt="Preview">
                                <canvas id="pdfCanvas" class="hidden"></canvas>
                            </div>
                        </div>
                    </div>

                    <div class="flex justify-center">
                        <button type="submit" id="predictBtn" class="btn-primary inline-flex items-center px-8 py-4 border border-transparent text-lg font-semibold rounded-2xl text-white disabled:opacity-50 disabled:cursor-not-allowed">
                            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                            </svg>
                            Analyze Resume
                        </button>
                    </div>
                </div>
            </form>

            <div id="results" class="hidden fade-in">
                <div class="border-t border-gray-200 pt-8">
                    <h2 class="text-2xl font-bold text-gray-900 mb-6 text-center">Your Prediction Results</h2>
                    <div id="predictions" class="space-y-6"></div>
                </div>
            </div>

            <div id="error" class="hidden mt-6 p-6 rounded-2xl bg-red-50 border border-red-200">
                <div class="flex items-center">
                    <svg class="w-6 h-6 text-red-500 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <span class="text-red-700 font-medium"></span>
                </div>
            </div>

            <div id="loading" class="hidden">
                <div class="flex flex-col items-center justify-center py-12">
                    <div class="loading-spinner mb-4"></div>
                    <h3 class="text-xl font-semibold text-gray-700 mb-2">Analyzing Your Resume</h3>
                    <p class="text-gray-500 text-center max-w-md">Our AI is processing your resume using advanced machine learning models. This may take a few moments...</p>
                </div>
            </div>
        </div>

        <div class="text-center mt-12">
            <p class="text-gray-500 text-sm">Powered by DistilBERT & LSTM Neural Networks</p>
            <p class="text-gray-400 text-xs mt-1">© 2025 AI Job Role Prediction Platform</p>
        </div>
    </div>

    <script>
        const form = document.getElementById('uploadForm');
        const fileInput = document.getElementById('file-upload');
        const preview = document.getElementById('preview');
        const pdfCanvas = document.getElementById('pdfCanvas');
        const results = document.getElementById('results');
        const predictions = document.getElementById('predictions');
        const error = document.getElementById('error');
        const loading = document.getElementById('loading');
        const dropZone = document.getElementById('dropZone');

        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js';

        function showError(message) {
            error.querySelector('span').textContent = message;
            error.classList.remove('hidden');
            error.classList.add('fade-in');
        }

        function clearResults() {
            error.classList.add('hidden');
            results.classList.add('hidden');
            predictions.innerHTML = '';
        }

        function handleFile(file) {
            if (file && file.size > 10 * 1024 * 1024) {
                showError('File size exceeds 10MB limit.');
                fileInput.value = '';
                return;
            }
            if (file) {
                preview.classList.add('hidden');
                pdfCanvas.classList.add('hidden');

                if (file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.classList.remove('hidden');
                        preview.classList.add('fade-in');
                    };
                    reader.readAsDataURL(file);
                } else if (file.type === 'application/pdf') {
                    const reader = new FileReader();
                    reader.onload = async function(e) {
                        const typedArray = new Uint8Array(e.target.result);
                        const pdf = await pdfjsLib.getDocument(typedArray).promise;
                        const page = await pdf.getPage(1);
                        const viewport = page.getViewport({ scale: 1.0 });
                        pdfCanvas.height = viewport.height;
                        pdfCanvas.width = viewport.width;
                        await page.render({
                            canvasContext: pdfCanvas.getContext('2d'),
                            viewport: viewport
                        }).promise;
                        pdfCanvas.classList.remove('hidden');
                        pdfCanvas.classList.add('fade-in');
                    };
                    reader.readAsArrayBuffer(file);
                }
                clearResults();
            }
        }

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file) {
                fileInput.files = e.dataTransfer.files;
                handleFile(file);
            }
        });

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            handleFile(file);
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!fileInput.files.length) {
                showError('Please select a file');
                return;
            }

            const formData = new FormData(form);
            loading.classList.remove('hidden');
            loading.classList.add('fade-in');
            clearResults();

            try {
                const response = await fetch('/', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }

                const data = await response.json();

                if (data.error) {
                    showError(data.error);
                } else if (data.success) {
                    const pred = data.predictions[0];
                    displayPredictions(pred);
                }
            } catch (err) {
                showError('An error occurred while processing the resume: ' + err.message);
            } finally {
                loading.classList.add('hidden');
            }
        });

        function formatAnalysis(explanation) {
            if (!explanation || !explanation.text) {
                return '<div class="text-gray-700 leading-relaxed">No analysis available.</div>';
            }
            
            const sections = explanation.text.split('\\n\\n');
            return `
                <div class="space-y-6">
                    ${sections.map(section => {
                        if (!section.includes(':')) return '';
                        const [title, ...contentParts] = section.split('\\n');
                        const sectionTitle = title.split(':')[0].trim();
                        const content = contentParts.join('\\n').trim();
                        
                        if (sectionTitle === 'Alternative Career Paths') {
                            // Special formatting for alternative careers section
                            const careers = content.split('\\n\\n').filter(Boolean);
                            return `
                                <div class="bg-white p-6 rounded-xl border border-gray-200">
                                    <h4 class="text-xl font-semibold text-gray-900 mb-4">${sectionTitle}</h4>
                                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                                        ${careers.map(career => {
                                            const lines = career.split('\\n');
                                            const roleName = lines[0].trim();
                                            const matchLine = lines.find(l => l.startsWith('Match:'));
                                            const reasoningLine = lines.find(l => l.startsWith('Reasoning:'));
                                            
                                            const matchPercent = matchLine ? matchLine.split(':')[1].trim() : '';
                                            const reasoning = reasoningLine ? reasoningLine.split(':')[1].trim() : '';
                                            
                                            return roleName ? `
                                                <div class="bg-gray-50 p-4 rounded-lg hover:shadow-md transition-shadow duration-200">
                                                    <div class="flex justify-between items-center mb-2">
                                                        <span class="font-semibold text-gray-900">${roleName}</span>
                                                        <span class="text-sm text-indigo-600 font-medium">${matchPercent}</span>
                                                    </div>
                                                    <p class="text-gray-700 text-sm">${reasoning}</p>
                                                </div>
                                            ` : '';
                                        }).join('')}
                                    </div>
                                </div>
                            `;
                        }
                        
                        return `
                            <div class="bg-white p-6 rounded-xl border border-gray-200">
                                <h4 class="text-xl font-semibold text-gray-900 mb-3">${sectionTitle}</h4>
                                <div class="text-gray-700 leading-relaxed whitespace-pre-line">${content}</div>
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
        }
        
        function displayPredictions(pred) {
            predictions.innerHTML = `
                <div class="result-card fade-in">
                    <div class="flex items-start justify-between mb-6">
                        <div>
                            <h3 class="text-2xl font-bold text-gray-900 mb-2">Predicted Job Role</h3>
                            <span class="text-3xl font-extrabold gradient-text">${pred.label}</span>
                        </div>
                        <div class="bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm font-medium">
                            ${(pred.confidence * 100).toFixed(1)}% Match
                        </div>
                    </div>

                    <div class="mb-6">
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-sm font-medium text-gray-700">Confidence Level</span>
                            <span class="text-sm text-gray-600">${(pred.confidence * 100).toFixed(2)}%</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div class="confidence-bar h-2 rounded-full" style="width: ${(pred.confidence * 100)}%"></div>
                        </div>
                    </div>

                    <div class="mb-6">
                        ${formatAnalysis(pred.explanation)}
                    </div>

                    <div class="flex items-center justify-between pt-4 border-t border-gray-200">
                        <div class="flex items-center text-sm text-gray-600">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"></path>
                            </svg>
                            Model: ${pred.model}
                        </div>
                        <div class="text-sm text-gray-500">
                            Processed with AI
                        </div>
                    </div>
                </div>
            `;
            results.classList.remove('hidden');
            results.classList.add('fade-in');
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def serve_ui():
    print("Serving UI...")
    return HTML_CONTENT

# Update the predict route to use loaded models
@app.route('/', methods=['POST'])
def predict():
    print("Received prediction request...")
    if 'file' not in request.files:
        print("No file uploaded in request.")
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        print("No file selected.")
        return jsonify({"error": "No file selected"}), 400

    # Sanitize file name
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, safe_filename)
    print(f"Attempting to save file to: {file_path}")

    try:
        os.makedirs(temp_dir, exist_ok=True)
        file.save(file_path)
        print(f"File saved successfully to: {file_path}")

        # Load trained models
        lstm_model, distilbert_model = load_trained_models()
        
        # Process resume
        preview_url = generate_pdf_preview(file_path)
        resume_text, sections = process_uploaded_resume(file_path)
        
        # Make prediction
        job_role, confidence, explanation, model_used = predict_job_role(
            resume_text, sections, distilbert_model, lstm_model, tokenizer, tfidf, label_encoder, device
        )
        
        # Enhance prediction with Gemini AI
        enhanced_analysis = enhance_prediction_with_gemini(job_role, confidence, resume_text, sections, explanation)
        
        final_explanation = {
            "text": enhanced_analysis if enhanced_analysis else explanation,
            "is_enhanced": enhanced_analysis is not None,
            "alternative_roles": explanation.get("alternative_roles", [])
        }
        
        response = {
            "success": True,
            "predictions": [{
                "label": job_role,
                "confidence": confidence,
                "explanation": final_explanation,
                "model": f"{model_used} with Gemini AI Enhancement"
            }]
        }
        if preview_url:
            response["preview_url"] = preview_url
        return jsonify(response)
    except Exception as e:
        print(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        if os.path.exists(file_path):
            print(f"Cleaning up file: {file_path}")
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up file {file_path}: {e}")
                
def extract_key_skills(text, tfidf_vectorizer):
    """
    Extracts key terms from the cleaned resume text based on TF-IDF scores.

    Args:
        text (str): The cleaned resume text.
        tfidf_vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.

    Returns:
        list: A list of tuples, where each tuple contains a term and its TF-IDF score,
              sorted by score in descending order.
    """
    # Transform the text using the fitted TF-IDF vectorizer
    tfidf_matrix = tfidf_vectorizer.transform([text])

    # Get feature names (terms)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Get the TF-IDF scores for the text
    # The matrix is sparse, so convert to a dense array for easier processing
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Create a list of (term, score) tuples
    scored_terms = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names)) if tfidf_scores[i] > 0]

    # Sort the terms by score in descending order
    scored_terms = sorted(scored_terms, key=lambda item: item[1], reverse=True)

    # Return a reasonable number of top terms (e.g., top 10 or 15)
    return scored_terms[:15]

# Run Flask with ngrok
NGROK_AUTH_TOKEN = "2zBA5iHkCv7ApqDStrt9SqwSu1u_4NwhjBSY8J812iwmUP3V4"
try:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    ngrok.kill()
    public_url = ngrok.connect(5001)
    print(f"Public URL: {public_url}")
except Exception as e:
    print(f"Error setting up ngrok: {e}")
    raise

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error running Flask app: {e}")