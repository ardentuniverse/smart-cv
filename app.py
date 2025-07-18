import os
import re
import json
import logging
import secrets
from flask import Flask, request, render_template, session
from werkzeug.utils import secure_filename
from rapidfuzz import fuzz
import fitz  # PyMuPDF
import docx
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load keyword groups from config file
def load_keyword_groups():
    config_path = os.path.join(os.path.dirname(__file__), 'keywords.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Keyword groups configuration file not found.")
        return {}
    except json.JSONDecodeError:
        logger.error("Invalid JSON in keyword groups configuration.")
        return {}

KEYWORD_GROUPS = load_keyword_groups()

def extract_text_from_pdf(file_obj):
    try:
        with fitz.open(stream=file_obj, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text.strip()
    except fitz.FileDataError:
        raise ValueError("Invalid or corrupt PDF file.")

def extract_text_from_docx(file_obj):
    try:
        doc = docx.Document(file_obj)
        return " ".join(p.text for p in doc.paragraphs).strip()
    except docx.opc.exceptions.PackageNotFoundError:
        raise ValueError("Invalid or corrupt DOCX file.")

def calculate_similarity(cv_text, jd_text):
    return round(fuzz.token_set_ratio(cv_text, jd_text), 2)

def generate_profession_suggestions(cv_text, jd_text, profession):
    suggestions = []
    cv_text = cv_text.lower()
    jd_text = jd_text.lower()
    
    if profession not in KEYWORD_GROUPS:
        return []

    for keyword, message in KEYWORD_GROUPS.get(profession, []):
        if fuzz.partial_ratio(keyword, jd_text) > 90 and fuzz.partial_ratio(keyword, cv_text) < 90:
            suggestions.append(f"\u2b24 {message}")

    return suggestions[:10]

def dynamic_summary(score):
    if score >= 85:
        return "Strong fit. Just polish your CV for clarity and impact."
    elif score >= 70:
        return "Good match. You're close — refine with a few skill or tool mentions."
    elif score >= 50:
        return "Decent alignment, but some key skills or achievements are missing."
    else:
        return "Significant gaps found. Add missing tools, skills, or clearer impact."

def sanitize_input(text):
    # Basic sanitization to remove HTML tags and scripts
    return re.sub(r'<[^>]+>', '', text).strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    suggestions = []
    summary = ""
    error = None
    csrf_token = session.get('csrf_token', secrets.token_hex(16))
    session['csrf_token'] = csrf_token

    if request.method == 'POST':
        if request.form.get('csrf_token') != session.get('csrf_token'):
            error = "Invalid CSRF token."
        else:
            profession = request.form.get("profession")
            jd_text = sanitize_input(request.form.get('job_description', '').strip())
            file = request.files.get('resume')

            if not profession:
                error = "Please select a profession."
            elif not jd_text:
                error = "Please paste a job description."
            elif not file:
                error = "Please upload a CV file."
            elif file.content_length > app.config['MAX_CONTENT_LENGTH']:
                error = "File size exceeds 2MB limit."
            elif file.mimetype not in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                error = "Invalid file type. Upload PDF or DOCX."
            else:
                filename = secure_filename(file.filename)
                logger.info(f"Processing file: {filename}")
                if not filename.lower().endswith(('.pdf', '.docx')):
                    error = "Unsupported file format. Use PDF or DOCX."
                else:
                    try:
                        file_bytes = file.read()
                        if filename.lower().endswith('.pdf'):
                            cv_text = extract_text_from_pdf(BytesIO(file_bytes))
                        else:
                            cv_text = extract_text_from_docx(BytesIO(file_bytes))

                        score = calculate_similarity(cv_text, jd_text)
                        suggestions = generate_profession_suggestions(cv_text, jd_text, profession)
                        summary = dynamic_summary(score)
                    except ValueError as e:
                        error = str(e)
                    except Exception as e:
                        logger.error(f"Unexpected error processing file: {str(e)}")
                        error = f"Unexpected error processing file: {str(e)}"

    return render_template('index.html', score=score, suggestions=suggestions, summary=summary, error=error, 
                          keyword_groups=KEYWORD_GROUPS, csrf_token=csrf_token)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
