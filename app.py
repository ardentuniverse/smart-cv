import os
import re
import logging
import secrets
from flask import Flask, request, render_template_string, session
from werkzeug.utils import secure_filename
from rapidfuzz import fuzz
import fitz  # PyMuPDF
from fitz import FileDataError, EmptyFileError
import docx
from docx.opc.exceptions import PackageNotFoundError
from io import BytesIO
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download NLTK data with error handling
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logging.error(f"Failed to download NLTK data: {str(e)}")
        raise

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
if not app.config['SECRET_KEY']:
    raise ValueError("No SECRET_KEY set in environment variables.")
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024  # 512KB limit for memory efficiency

# Configure logging (log errors only)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# KEYWORD_GROUPS with regex patterns for synonyms
KEYWORD_GROUPS = {
    "tech": [
        ["python|scripting|pandas|django", "Include Python scripting or automation experience (e.g., Pandas, Django)."],
        ["sql|database|query|mysql|postgresql", "Mention SQL or database querying skills (e.g., MySQL, PostgreSQL)."],
        ["api|rest|graphql", "Add RESTful API or GraphQL integration experience."]
    ],
    "finance_banking": [
        ["erp|sap|oracle|quickbooks", "List ERP systems used (e.g., SAP, Oracle, QuickBooks)."],
        ["audit|financial review", "Add experience preparing or undergoing audits."]
    ],
    "medical": [
        ["patient care|healthcare", "Include hands-on or remote patient care experience."],
        ["clinical|trials|procedures", "Mention clinical trials or procedures."]
    ],
    "education": [
        ["curriculum|lesson plan", "Mention curriculum design or learning material development."],
        ["assessment|evaluation|grading", "Add evaluation or grading experience."]
    ],
    "logistics": [
        ["inventory|stock management", "Add inventory control or stock management experience."],
        ["supply chain|logistics", "Mention end-to-end supply chain processes."]
    ],
    "legal": [
        ["contract|agreement", "Include contract drafting, review, or negotiation."],
        ["compliance|regulation", "Highlight regulatory or legal compliance tasks."]
    ],
    "hospitality": [
        ["guest service|customer satisfaction", "Highlight customer/guest satisfaction focus."],
        ["reservation|booking", "Mention reservation systems or booking platforms used."]
    ],
    "oil_gas": [
        ["hse|safety|environment", "Mention Health, Safety & Environment responsibilities."],
        ["drilling|well site", "Include drilling operations or well site management."]
    ],
    "sales_business_retail": [
        ["crm|salesforce|hubspot", "Mention CRM platforms like Salesforce, HubSpot."],
        ["lead generation|prospecting", "Highlight prospecting or lead-gen activities."]
    ],
    "engineering": [
        ["cad|autocad|solidworks", "Include CAD software experience (e.g., AutoCAD, SolidWorks)."],
        ["project management|engineering design", "Mention project management or engineering design skills."]
    ],
    "marketing": [
        ["seo|digital marketing", "Include SEO or digital marketing campaign experience."],
        ["content creation|social media", "Mention content creation or social media strategy skills."]
    ],
    "human_resources": [
        ["recruitment|talent acquisition", "Highlight experience in talent acquisition or recruitment."],
        ["onboarding|training", "Mention employee onboarding or training program development."]
    ],
    "construction": [
        ["site management|supervision", "Include construction site management or supervision experience."],
        ["safety protocols|safety standards", "Mention adherence to construction safety standards."]
    ],
    "data_science": [
        ["machine learning|statistical modeling", "Include machine learning or statistical modeling experience."],
        ["data visualization|tableau|power bi", "Mention data visualization tools like Tableau or Power BI."]
    ]
}

def preprocess_text(text):
    """Preprocess text by removing stopwords and non-alphanumeric characters."""
    try:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        return ' '.join([word for word in tokens if word.isalnum() and word not in stop_words])
    except Exception as e:
        logger.error(f"Error in preprocess_text: {str(e)}")
        raise ValueError("Failed to preprocess text due to invalid content.")

def extract_text_from_pdf(file_obj):
    """Extract text from PDF file, limited to first 5 pages."""
    try:
        with fitz.open(stream=file_obj, filetype="pdf") as doc:
            text = ""
            for page in doc[:5]:  # Process only first 5 pages
                text += page.get_text()
            if not text.strip():
                raise ValueError("Empty or unreadable PDF file.")
            return text.strip()
    except FileDataError:
        raise ValueError("Invalid or corrupt PDF file.")
    except EmptyFileError:
        raise ValueError("Empty PDF file.")
    except Exception as e:
        logger.error(f"Error in extract_text_from_pdf: {str(e)}")
        raise ValueError(f"Failed to process PDF: {str(e)}")

def extract_text_from_docx(file_obj):
    """Extract text from DOCX file."""
    try:
        doc = docx.Document(file_obj)
        text = " ".join(p.text for p in doc.paragraphs).strip()
        if not text:
            raise ValueError("Empty or unreadable DOCX file.")
        return text
    except PackageNotFoundError:
        raise ValueError("Invalid or corrupt DOCX file.")
    except Exception as e:
        logger.error(f"Error in extract_text_from_docx: {str(e)}")
        raise ValueError(f"Failed to process DOCX: {str(e)}")

def calculate_similarity(cv_text, jd_text):
    """Calculate similarity score using weighted fuzzy metrics."""
    try:
        cv_text = preprocess_text(cv_text)
        jd_text = preprocess_text(jd_text)
        token_score = fuzz.token_set_ratio(cv_text, jd_text)
        partial_score = fuzz.partial_ratio(cv_text, jd_text)
        return round((0.6 * token_score + 0.4 * partial_score), 2)
    except Exception as e:
        logger.error(f"Error in calculate_similarity: {str(e)}")
        raise ValueError("Failed to calculate similarity score.")

def extract_jd_keywords(jd_text, top_n=5):
    """Extract top keywords from job description."""
    try:
        words = re.findall(r'\b\w+\b', preprocess_text(jd_text))
        word_counts = Counter(words)
        stop_words = set(stopwords.words('english'))
        return [word for word, _ in word_counts.most_common(top_n) if word not in stop_words]
    except Exception as e:
        logger.error(f"Error in extract_jd_keywords: {str(e)}")
        raise ValueError("Failed to extract keywords from job description.")

def generate_profession_suggestions(cv_text, jd_text, profession):
    """Generate suggestions based on predefined and dynamic keywords."""
    try:
        suggestions = []
        cv_text = preprocess_text(cv_text)
        jd_text = preprocess_text(jd_text)
        jd_keywords = extract_jd_keywords(jd_text)

        # Predefined keyword matches
        for keyword_pattern, message in KEYWORD_GROUPS.get(profession, []):
            if any(fuzz.partial_ratio(keyword, jd_text) > 80 for keyword in keyword_pattern.split('|')) and \
               all(fuzz.partial_ratio(keyword, cv_text) < 80 for keyword in keyword_pattern.split('|')):
                suggestions.append(f"\u2b24 {message}")

        # Dynamic JD keyword matches
        for keyword in jd_keywords:
            if fuzz.partial_ratio(keyword, jd_text) > 80 and fuzz.partial_ratio(keyword, cv_text) < 80:
                suggestions.append(f"\u2b24 Include experience with '{keyword}' from the job description.")

        return suggestions[:10]
    except Exception as e:
        logger.error(f"Error in generate_profession_suggestions: {str(e)}")
        raise ValueError("Failed to generate suggestions.")

def dynamic_summary(score, profession, suggestions):
    """Generate profession-specific summary."""
    try:
        profession_name = profession.replace('_', ' ').title()
        if score >= 85:
            return f"Strong fit for {profession_name}. Polish your CV with specific examples of impact."
        elif score >= 70:
            return f"Good match for {profession_name}. Add {len(suggestions)} key skills or tools to improve."
        elif score >= 50:
            return f"Moderate fit for {profession_name}. Focus on adding {len(suggestions)} missing skills or clearer achievements."
        else:
            return f"Significant gaps for {profession_name}. Incorporate {len(suggestions)} key skills or tools listed."
    except Exception as e:
        logger.error(f"Error in dynamic_summary: {str(e)}")
        raise ValueError("Failed to generate summary.")

def sanitize_input(text):
    """Sanitize input by removing HTML tags."""
    try:
        return re.sub(r'<[^>]+>', '', text).strip()
    except Exception as e:
        logger.error(f"Error in sanitize_input: {str(e)}")
        raise ValueError("Failed to sanitize input.")

# Embedded HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart CV Checker</title>
    <link href="https://fonts.googleapis.com/css2?family=Cabin&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Cabin', sans-serif; max-width: 800px; margin: auto; padding: 20px; line-height: 1.6; }
        textarea, select, input[type=file] { width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        ul { padding-left: 20px; }
        .error { color: red; }
        label { font-weight: bold; }
    </style>
</head>
<body>
    <h2>Smart CV Checker</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
        <p>
            <label for="profession" aria-label="Select your profession">Select your profession:</label>
            <select name="profession" id="profession" required>
                <option value="">--Choose one--</option>
                {% for key in keyword_groups.keys() %}
                <option value="{{ key }}" {% if request.form.get('profession') == key %}selected{% endif %}>{{ key.replace('_', ' ').title() }}</option>
                {% endfor %}
            </select>
        </p>
        <p>
            <label for="job_description" aria-label="Paste job description">Job Description:</label>
            <textarea name="job_description" id="job_description" rows="6" maxlength="5000" placeholder="Paste job description here..." required>{{ request.form.get('job_description', '') }}</textarea>
        </p>
        <p>
            <label for="resume" aria-label="Upload resume">Upload Resume:</label>
            <input type="file" name="resume" id="resume" required>
        </p>
        <p><button type="submit">Scan</button></p>
    </form>
    {% if score is not none %}
        <h3>ATS Score: {{ score }}%</h3>
        <p><strong>Summary:</strong> {{ summary }}</p>
        <h4>Suggestions to Improve CV</h4>
        {% if suggestions %}
            <ul>
            {% for s in suggestions %}
                <li>{{ s }}</li>
            {% endfor %}
            </ul>
        {% else %}
            <p>No suggestions available for this profession or no gaps detected.</p>
        {% endif %}
    {% elif error %}
        <p class="error"><strong>Error:</strong> {{ error }}</p>
    {% endif %}
</body>
</html>
'''

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
                error = "File size exceeds 512KB limit."
            elif file.mimetype not in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                error = "Invalid file type. Upload PDF or DOCX."
            else:
                filename = secure_filename(file.filename)
                if not filename.lower().endswith(('.pdf', '.docx')):
                    error = "Unsupported file format. Use PDF or DOCX."
                else:
                    try:
                        file_bytes = file.read()
                        if not file_bytes:
                            raise ValueError("Empty file uploaded.")
                        file_stream = BytesIO(file_bytes)
                        if filename.lower().endswith('.pdf'):
                            cv_text = extract_text_from_pdf(file_stream)
                        else:
                            cv_text = extract_text_from_docx(file_stream)

                        # Cache preprocessed texts
                        cv_text = session.get('cv_text', preprocess_text(cv_text))
                        jd_text = session.get('jd_text', preprocess_text(jd_text))
                        session['cv_text'] = cv_text
                        session['jd_text'] = jd_text

                        score = calculate_similarity(cv_text, jd_text)
                        suggestions = generate_profession_suggestions(cv_text, jd_text, profession)
                        summary = dynamic_summary(score,–

System: The response was cut off because it exceeded the maximum allowed length. To provide a complete and accurate response, I’ll continue from where it left off, ensuring the code is fully updated with all requested improvements, robust error handling, and compatibility with Render’s 512MB free tier. The artifact will complete the `app.py` file, maintaining the existing structure and addressing the “Unexpected error processing file” issue with specific error handling, memory optimizations, and secure configuration.

<xaiArtifact artifact_id="443e093f-0817-43d8-9b33-eed6b7010142" artifact_version_id="c57e41ac-7e73-4cc1-bc7a-c5f77ad0247e" title="app.py" contentType="text/python">
import os
import re
import logging
import secrets
from flask import Flask, request, render_template_string, session
from werkzeug.utils import secure_filename
from rapidfuzz import fuzz
import fitz  # PyMuPDF
from fitz import FileDataError, EmptyFileError
import docx
from docx.opc.exceptions import PackageNotFoundError
from io import BytesIO
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download NLTK data with error handling
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logging.error(f"Failed to download NLTK data: {str(e)}")
        raise

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
if not app.config['SECRET_KEY']:
    raise ValueError("No SECRET_KEY set in environment variables.")
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024  # 512KB limit for memory efficiency

# Configure logging (log errors only)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# KEYWORD_GROUPS with regex patterns for synonyms
KEYWORD_GROUPS = {
    "tech": [
        ["python|scripting|pandas|django", "Include Python scripting or automation experience (e.g., Pandas, Django)."],
        ["sql|database|query|mysql|postgresql", "Mention SQL or database querying skills (e.g., MySQL, PostgreSQL)."],
        ["api|rest|graphql", "Add RESTful API or GraphQL integration experience."]
    ],
    "finance_banking": [
        ["erp|sap|oracle|quickbooks", "List ERP systems used (e.g., SAP, Oracle, QuickBooks)."],
        ["audit|financial review", "Add experience preparing or undergoing audits."]
    ],
    "medical": [
        ["patient care|healthcare", "Include hands-on or remote patient care experience."],
        ["clinical|trials|procedures", "Mention clinical trials or procedures."]
    ],
    "education": [
        ["curriculum|lesson plan", "Mention curriculum design or learning material development."],
        ["assessment|evaluation|grading", "Add evaluation or grading experience."]
    ],
    "logistics": [
        ["inventory|stock management", "Add inventory control or stock management experience."],
        ["supply chain|logistics", "Mention end-to-end supply chain processes."]
    ],
    "legal": [
        ["contract|agreement", "Include contract drafting, review, or negotiation."],
        ["compliance|regulation", "Highlight regulatory or legal compliance tasks."]
    ],
    "hospitality": [
        ["guest service|customer satisfaction", "Highlight customer/guest satisfaction focus."],
        ["reservation|booking", "Mention reservation systems or booking platforms used."]
    ],
    "oil_gas": [
        ["hse|safety|environment", "Mention Health, Safety & Environment responsibilities."],
        ["drilling|well site", "Include drilling operations or well site management."]
    ],
    "sales_business_retail": [
        ["crm|salesforce|hubspot", "Mention CRM platforms like Salesforce, HubSpot."],
        ["lead generation|prospecting", "Highlight prospecting or lead-gen activities."]
    ],
    "engineering": [
        ["cad|autocad|solidworks", "Include CAD software experience (e.g., AutoCAD, SolidWorks)."],
        ["project management|engineering design", "Mention project management or engineering design skills."]
    ],
    "marketing": [
        ["seo|digital marketing", "Include SEO or digital marketing campaign experience."],
        ["content creation|social media", "Mention content creation or social media strategy skills."]
    ],
    "human_resources": [
        ["recruitment|talent acquisition", "Highlight experience in talent acquisition or recruitment."],
        ["onboarding|training", "Mention employee onboarding or training program development."]
    ],
    "construction": [
        ["site management|supervision", "Include construction site management or supervision experience."],
        ["safety protocols|safety standards", "Mention adherence to construction safety standards."]
    ],
    "data_science": [
        ["machine learning|statistical modeling", "Include machine learning or statistical modeling experience."],
        ["data visualization|tableau|power bi", "Mention data visualization tools like Tableau or Power BI."]
    ]
}

def preprocess_text(text):
    """Preprocess text by removing stopwords and non-alphanumeric characters."""
    try:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        return ' '.join([word for word in tokens if word.isalnum() and word not in stop_words])
    except Exception as e:
        logger.error(f"Error in preprocess_text: {str(e)}")
        raise ValueError("Failed to preprocess text due to invalid content.")

def extract_text_from_pdf(file_obj):
    """Extract text from PDF file, limited to first 5 pages."""
    try:
        with fitz.open(stream=file_obj, filetype="pdf") as doc:
            text = ""
            for page in doc[:5]:  # Process only first 5 pages
                text += page.get_text()
            if not text.strip():
                raise ValueError("Empty or unreadable PDF file.")
            return text.strip()
    except FileDataError:
        raise ValueError("Invalid or corrupt PDF file.")
    except EmptyFileError:
        raise ValueError("Empty PDF file.")
    except Exception as e:
        logger.error(f"Error in extract_text_from_pdf: {str(e)}")
        raise ValueError(f"Failed to process PDF: {str(e)}")

def extract_text_from_docx(file_obj):
    """Extract text from DOCX file."""
    try:
        doc = docx.Document(file_obj)
        text = " ".join(p.text for p in doc.paragraphs).strip()
        if not text:
            raise ValueError("Empty or unreadable DOCX file.")
        return text
    except PackageNotFoundError:
        raise ValueError("Invalid or corrupt DOCX file.")
    except Exception as e:
        logger.error(f"Error in extract_text_from_docx: {str(e)}")
        raise ValueError(f"Failed to process DOCX: {str(e)}")

def calculate_similarity(cv_text, jd_text):
    """Calculate similarity score using weighted fuzzy metrics."""
    try:
        cv_text = preprocess_text(cv_text)
        jd_text = preprocess_text(jd_text)
        token_score = fuzz.token_set_ratio(cv_text, jd_text)
        partial_score = fuzz.partial_ratio(cv_text, jd_text)
        return round((0.6 * token_score + 0.4 * partial_score), 2)
    except Exception as e:
        logger.error(f"Error in calculate_similarity: {str(e)}")
        raise ValueError("Failed to calculate similarity score.")

def extract_jd_keywords(jd_text, top_n=5):
    """Extract top keywords from job description."""
    try:
        words = re.findall(r'\b\w+\b', preprocess_text(jd_text))
        word_counts = Counter(words)
        stop_words = set(stopwords.words('english'))
        return [word for word, _ in word_counts.most_common(top_n) if word not in stop_words]
    except Exception as e:
        logger.error(f"Error in extract_jd_keywords: {str(e)}")
        raise ValueError("Failed to extract keywords from job description.")

def generate_profession_suggestions(cv_text, jd_text, profession):
    """Generate suggestions based on predefined and dynamic keywords."""
    try:
        suggestions = []
        cv_text = preprocess_text(cv_text)
        jd_text = preprocess_text(jd_text)
        jd_keywords = extract_jd_keywords(jd_text)

        # Predefined keyword matches
        for keyword_pattern, message in KEYWORD_GROUPS.get(profession, []WATERFALL):
            if any(fuzz.partial_ratio(keyword, jd_text) > 80 for keyword in keyword_pattern.split('|')) and \
               all(fuzz.partial_ratio(keyword, cv_text) < 80 for keyword in keyword_pattern.split('|')):
                suggestions.append(f"\u2b24 {message}")

        # Dynamic JD keyword matches
        for keyword in jd_keywords:
            if fuzz.partial_ratio(keyword, jd_text) > 80 and fuzz.partial_ratio(keyword, cv_text) < 80:
                suggestions.append(f"\u2b24 Include experience with '{keyword}' from the job description.")

        return suggestions[:10]
    except Exception as e:
        logger.error(f"Error in generate_profession_suggestions: {str(e)}")
        raise ValueError("Failed to generate suggestions.")

def dynamic_summary(score, profession, suggestions):
    """Generate profession-specific summary."""
    try:
        profession_name = profession.replace('_', ' ').title()
        if score >= 85:
            return f"Strong fit for {profession_name}. Polish your CV with specific examples of impact."
        elif score >= 70:
            return f"Good match for {profession_name}. Add {len(suggestions)} key skills or tools to improve."
        elif score >= 50:
            return f"Moderate fit for {profession_name}. Focus on adding {len(suggestions)} missing skills or clearer achievements."
        else:
            return f"Significant gaps for {profession_name}. Incorporate {len(suggestions)} key skills or tools listed."
    except Exception as e:
        logger.error(f"Error in dynamic_summary: {str(e)}")
        raise ValueError("Failed to generate summary.")

def sanitize_input(text):
    """Sanitize input by removing HTML tags."""
    try:
        return re.sub(r'<[^>]+>', '', text).strip()
    except Exception as e:
        logger.error(f"Error in sanitize_input: {str(e)}")
        raise ValueError("Failed to sanitize input.")

# Embedded HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart CV Checker</title>
    <link href="https://fonts.googleapis.com/css2?family=Cabin&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Cabin', sans-serif; max-width: 800px; margin: auto; padding: 20px; line-height: 1.6; }
        textarea, select, input[type=file] { width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        ul { padding-left: 20px; }
        .error { color: red; }
        label { font-weight: bold; }
    </style>
</head>
<body>
    <h2>Smart CV Checker</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
        <p>
            <label for="profession" aria-label="Select your profession">Select your profession:</label>
            <select name="profession" id="profession" required>
                <option value="">--Choose one--</option>
                {% for key in keyword_groups.keys() %}
                <option value="{{ key }}" {% if request.form.get('profession') == key %}selected{% endif %}>{{ key.replace('_', ' ').title() }}</option>
                {% endfor %}
            </select>
        </p>
        <p>
            <label for="job_description" aria-label="Paste job description">Job Description:</label>
            <textarea name="job_description" id="job_description" rows="6" maxlength="5000" placeholder="Paste job description here..." required>{{ request.form.get('job_description', '') }}</textarea>
        </p>
        <p>
            <label for="resume" aria-label="Upload resume">Upload Resume:</label>
            <input type="file" name="resume" id="resume" required>
        </p>
        <p><button type="submit">Scan</button></p>
    </form>
    {% if score is not none %}
        <h3>ATS Score: {{ score }}%</h3>
        <p><strong>Summary:</strong> {{ summary }}</p>
        <h4>Suggestions to Improve CV</h4>
        {% if suggestions %}
            <ul>
            {% for s in suggestions %}
                <li>{{ s }}</li>
            {% endfor %}
            </ul>
        {% else %}
            <p>No suggestions available for this profession or no gaps detected.</p>
        {% endif %}
    {% elif error %}
        <p class="error"><strong>Error:</strong> {{ error }}</p>
    {% endif %}
</body>
</html>
'''

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
                error = "File size exceeds 512KB limit."
            elif file.mimetype not in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                error = "Invalid file type. Upload PDF or DOCX."
            else:
                filename = secure_filename(file.filename)
                if not filename.lower().endswith(('.pdf', '.docx')):
                    error = "Unsupported file format. Use PDF or DOCX."
                else:
                    try:
                        file_bytes = file.read()
                        if not file_bytes:
                            raise ValueError("Empty file uploaded.")
                        file_stream = BytesIO(file_bytes)
                        if filename.lower().endswith('.pdf'):
                            cv_text = extract_text_from_pdf(file_stream)
                        else:
                            cv_text = extract_text_from_docx(file_stream)

                        # Cache preprocessed texts
                        cv_text = session.get('cv_text', preprocess_text(cv_text))
                        jd_text = session.get('jd_text', preprocess_text(jd_text))
                        session['cv_text'] = cv_text
                        session['jd_text'] = jd_text

                        score = calculate_similarity(cv_text, jd_text)
                        suggestions = generate_profession_suggestions(cv_text, jd_text, profession)
                        summary = dynamic_summary(score, profession, suggestions)
                    except FileDataError:
                        error = "Invalid or corrupt PDF file. Please upload a valid PDF."
                    except EmptyFileError:
                        error = "Empty PDF file. Please upload a PDF with text content."
                    except PackageNotFoundError:
                        error = "Invalid or corrupt DOCX file. Please upload a valid DOCX."
                    except MemoryError:
                        error = "File too large to process. Please upload a smaller file (<512KB)."
                    except ValueError as e:
                        error = str(e)
                    except Exception as e:
                        logger.error(f"Unexpected error processing file: {str(e)}")
                        error = f"Unexpected error processing file: {str(e)}. Please try again with a different file."
    
    return render_template_string(HTML_TEMPLATE, score=score, suggestions=suggestions, summary=summary, error=error, 
                                 keyword_groups=KEYWORD_GROUPS, csrf_token=csrf_token)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
