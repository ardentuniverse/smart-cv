import os
import re
import logging
import secrets
from flask import Flask, request, render_template_string, session
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

# Embedded keyword groups with soft_skills removed and new professions added
KEYWORD_GROUPS = {
    "tech": [
        ["python", "Include Python scripting or automation experience."],
        ["sql", "Mention SQL or database querying skills."],
        ["api", "Add RESTful API integration or development work."]
    ],
    "finance_banking": [
        ["erp", "List ERP systems used (e.g. SAP, Oracle, QuickBooks)."],
        ["audit", "Add experience preparing or undergoing audits."]
    ],
    "medical": [
        ["patient care", "Include hands-on or remote patient care experience."],
        ["clinical", "Mention clinical trials or procedures."]
    ],
    "education": [
        ["curriculum", "Mention curriculum design or learning material development."],
        ["assessment", "Add evaluation or grading experience."]
    ],
    "logistics": [
        ["inventory", "Add inventory control or stock management experience."],
        ["supply chain", "Mention end-to-end supply chain processes."]
    ],
    "legal": [
        ["contract", "Include contract drafting, review, or negotiation."],
        ["compliance", "Highlight regulatory or legal compliance tasks."]
    ],
    "hospitality": [
        ["guest service", "Highlight customer/guest satisfaction focus."],
        ["reservation", "Mention reservation systems or booking platforms used."]
    ],
    "oil_gas": [
        ["hse", "Mention Health, Safety & Environment responsibilities."],
        ["drilling", "Include drilling operations or well site management."]
    ],
    "sales_business_retail": [
        ["crm", "Mention CRM platforms like Salesforce, HubSpot."],
        ["lead generation", "Highlight prospecting or lead-gen activities."]
    ],
    "engineering": [
        ["cad", "Include CAD software experience (e.g., AutoCAD, SolidWorks)."],
        ["project management", "Mention project management or engineering design skills."]
    ],
    "marketing": [
        ["seo", "Include SEO or digital marketing campaign experience."],
        ["content creation", "Mention content creation or social media strategy skills."]
    ],
    "human_resources": [
        ["recruitment", "Highlight experience in talent acquisition or recruitment."],
        ["onboarding", "Mention employee onboarding or training program development."]
    ],
    "construction": [
        ["site management", "Include construction site management or supervision experience."],
        ["safety protocols", "Mention adherence to construction safety standards."]
    ],
    "data_science": [
        ["machine learning", "Include machine learning or statistical modeling experience."],
        ["data visualization", "Mention data visualization tools like Tableau or Power BI."]
    ]
}

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

    return render_template_string(HTML_TEMPLATE, score=score, suggestions=suggestions, summary=summary, error=error, 
                                 keyword_groups=KEYWORD_GROUPS, csrf_token=csrf_token)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
