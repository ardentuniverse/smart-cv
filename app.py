import os
import re
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
from rapidfuzz import fuzz
import fitz  # PyMuPDF
import docx
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB

# Profession-specific keyword suggestion maps
KEYWORD_GROUPS = {
    "tech": [
        ("python", "Include Python scripting or automation experience."),
        ("sql", "Mention SQL or database querying skills."),
        ("api", "Add RESTful API integration or development work."),
    ],
    "finance_banking": [
        ("erp", "List ERP systems used (e.g. SAP, Oracle, QuickBooks)."),
        ("audit", "Add experience preparing or undergoing audits."),
    ],
    "medical": [
        ("patient care", "Include hands-on or remote patient care experience."),
        ("clinical", "Mention clinical trials or procedures."),
    ],
    "education": [
        ("curriculum", "Mention curriculum design or learning material development."),
        ("assessment", "Add evaluation or grading experience."),
    ],
    "logistics": [
        ("inventory", "Add inventory control or stock management experience."),
        ("supply chain", "Mention end-to-end supply chain processes."),
    ],
    "legal": [
        ("contract", "Include contract drafting, review, or negotiation."),
        ("compliance", "Highlight regulatory or legal compliance tasks."),
    ],
    "hospitality": [
        ("guest service", "Highlight customer/guest satisfaction focus."),
        ("reservation", "Mention reservation systems or booking platforms used."),
    ],
    "oil_gas": [
        ("hse", "Mention Health, Safety & Environment responsibilities."),
        ("drilling", "Include drilling operations or well site management."),
    ],
    "sales_business_retail": [
        ("crm", "Mention CRM platforms like Salesforce, HubSpot."),
        ("lead generation", "Highlight prospecting or lead-gen activities."),
    ],
    "soft_skills": [
        ("communication", "Add proof of communication skills."),
        ("team", "Add examples of teamwork or collaboration."),
    ]
}

def extract_text_from_pdf(file_obj):
    text = ""
    with fitz.open(stream=file_obj, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def extract_text_from_docx(file_obj):
    doc = docx.Document(file_obj)
    return " ".join(p.text for p in doc.paragraphs).strip()

def calculate_similarity(cv_text, jd_text):
    return round(fuzz.token_set_ratio(cv_text, jd_text), 2)

def generate_profession_suggestions(cv_text, jd_text, profession):
    suggestions = []
    cv_text = cv_text.lower()
    jd_text = jd_text.lower()
    
    if profession not in KEYWORD_GROUPS:
        return []

    for keyword, message in KEYWORD_GROUPS[profession]:
        if keyword in jd_text and keyword not in cv_text:
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

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    suggestions = []
    summary = ""
    error = None

    if request.method == 'POST':
        profession = request.form.get("profession")
        file = request.files.get('resume')
        jd_text = request.form.get('job_description', '').strip()

        if not profession:
            error = "Please select a profession."
        elif not jd_text:
            error = "Please paste a job description."
        elif not file:
            error = "Please upload a CV file."
        else:
            filename = secure_filename(file.filename)
            if not filename.lower().endswith(('.pdf', '.docx')):
                error = "Unsupported file format. Use PDF or DOCX."
            else:
                try:
                    file_bytes = file.read()
                    if filename.endswith('.pdf'):
                        cv_text = extract_text_from_pdf(BytesIO(file_bytes))
                    else:
                        cv_text = extract_text_from_docx(BytesIO(file_bytes))

                    score = calculate_similarity(cv_text, jd_text)
                    suggestions = generate_profession_suggestions(cv_text, jd_text, profession)
                    summary = dynamic_summary(score)
                except Exception as e:
                    error = f"Error processing file: {str(e)}"

    return render_template_string('''
    <html>
    <head>
        <title>Smart CV Checker</title>
        <link href="https://fonts.googleapis.com/css2?family=Cabin&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Cabin', sans-serif; max-width: 800px; margin: auto; padding: 20px; line-height: 1.6; }
            textarea, select, input[type=file] { width: 100%; padding: 10px; margin: 10px 0; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
            ul { padding-left: 20px; }
        </style>
    </head>
    <body>
        <h2>Smart CV Checker</h2>
        <form method="post" enctype="multipart/form-data">
            <p>
              <label for="profession">Select your profession:</label>
              <select name="profession" id="profession" required>
                <option value="">--Choose one--</option>
                {% for key in keyword_groups.keys() %}
                <option value="{{ key }}" {% if request.form.get('profession') == key %}selected{% endif %}>{{ key.replace('_', ' ').title() }}</option>
                {% endfor %}
              </select>
            </p>
            <p><textarea name="job_description" rows="6" placeholder="Paste job description here..." required>{{ request.form.get('job_description', '') }}</textarea></p>
            <p><input type="file" name="resume" required></p>
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
                <p>No major gaps detected.</p>
            {% endif %}
        {% elif error %}
            <p style="color: red;"><strong>Error:</strong> {{ error }}</p>
        {% endif %}
    </body>
    </html>
    ''', score=score, suggestions=suggestions, summary=summary, error=error, keyword_groups=KEYWORD_GROUPS)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
