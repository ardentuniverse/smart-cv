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

# --------------------------------------
# ROLE_KEYWORDS (Moved + Cleaned)
# --------------------------------------
ROLE_KEYWORDS = {
    # Tech & IT
    "ads": ["Digital Marketing Specialist", "Paid Ads Manager", "Performance Marketer", "PPC Specialist"],
    "seo": ["SEO Analyst", "Technical SEO Specialist"],
    "ai": ["AI Engineer", "Machine Learning Engineer"],
    "machine learning": ["Machine Learning Engineer", "ML Researcher"],
    "architect": ["Solutions Architect", "Cloud Architect"],
    "aws": ["AWS Engineer", "Cloud Solutions Architect"],
    "azure": ["Azure Engineer", "Microsoft Cloud Specialist"],
    "backend": ["Backend Developer", "API Engineer"],
    "cloud": ["Cloud Engineer", "DevOps Cloud Specialist"],
    "cybersecurity": ["Cybersecurity Analyst", "Security Engineer"],
    "data": ["Data Analyst", "Data Scientist", "BI Developer"],
    "python": ["Python Developer", "Data Engineer", "Automation Specialist"],
    "sql": ["SQL Developer", "Database Analyst"],
    "visualization": ["Dashboard Analyst", "Data Visualization Specialist"],
    "developer": ["Software Developer", "Application Developer"],
    "devops": ["DevOps Engineer", "CI/CD Specialist"],
    "software engineering": ["Software Engineer", "Backend Engineer", "Systems Engineer"],
    "flutter": ["Flutter Developer", "Mobile App Developer"],
    "frontend": ["Frontend Developer", "React Developer", "Web Engineer"],
    "full stack": ["Full Stack Developer", "Web Application Developer"],
    "it": ["IT Support Specialist", "Systems Analyst"],
    "mobile": ["Mobile Developer", "iOS/Android Developer"],
    "network": ["Network Administrator", "IT Infrastructure Engineer"],
    "systems": ["Systems Engineer", "Linux Admin"],
    "product manager": ["Product Manager", "Product Owner"],
    "project manager": ["Project Manager", "Scrum Master"],
    "qa": ["QA Engineer", "Test Automation Engineer"],
    "quality assurance": ["QA Analyst", "SDET"],
    "software": ["Software Engineer", "Software Developer"],
    "solutions": ["Solutions Engineer", "Technical Consultant"],
    "ui ux": ["UI Designer", "UX Designer", "Product Designer"],
    "web": ["Web Developer", "Frontend Developer"],
    "wordpress": ["WordPress Developer", "CMS Specialist"],

    # Creative
    "graphics": ["Graphic Designer", "Visual Content Creator", "Multimedia Designer"],
    "video": ["Video Editor", "Motion Graphics Designer"],

    # HR & Operations
    "recruit": ["Recruiter", "HR Manager", "HR Assistant"],

    # Marketing & Content
    "content": ["Content Writer", "Content Strategist", "Copywriter"],
    "email": ["Email Marketer", "CRM Executive", "Lifecycle Marketing Specialist"],
    "social": ["Social Media Manager", "Community Manager", "Social Strategist"],
    "brand": ["Brand Manager", "Marketing Coordinator"],

    # Engineering (Non-IT)
    "mechanical engineering": ["Mechanical Engineer", "Structural Engineer"],
    "civil engineering": ["Civil Engineer"],
    "electrical engineering": ["Electrical Engineer"],

    # Finance
    "finance": ["Financial Analyst", "Accountant", "Investment Analyst", "Treasury Manager"],

    # Education
    "education": ["Teacher", "Lecturer", "Academic Coordinator", "Curriculum Developer"],

    # Logistics
    "logistics": ["Logistics Manager", "Supply Chain Analyst", "Warehouse Supervisor", "Inventory Planner"],

    # Aviation
    "aviation": ["Pilot", "Flight Attendant", "Air Traffic Controller", "Aviation Safety Officer"],

    # Banking
    "banking": ["Bank Teller", "Loan Officer", "Relationship Manager", "Branch Manager"],

    # Real Estate
    "real estate": ["Real Estate Agent", "Property Manager", "Leasing Consultant", "Real Estate Analyst"],

    # Admin & Secretarial
    "admin": ["Administrative Assistant", "Executive Assistant", "Office Manager", "Receptionist"],

    # Catering
    "catering": ["Catering Manager", "Event Caterer", "Banquet Supervisor", "Catering Assistant"],

    # Consultancy
    "consultancy": ["Management Consultant", "Business Consultant", "Strategy Consultant", "HR Consultant"],

    # Customer Care
    "customer care": ["Customer Service Representative", "Call Center Agent", "Client Relations Officer"],

    # Hospitality
    "hospitality": ["Hotel Manager", "Concierge", "Front Desk Officer", "Housekeeping Supervisor"],

    # Law
    "law": ["Lawyer", "Paralegal", "Legal Advisor", "Corporate Counsel"],

    # Manufacturing
    "manufacturing": ["Production Supervisor", "Plant Manager", "Manufacturing Engineer", "Assembly Line Worker"],

    # Healthcare
    "healthcare": ["Nurse", "Doctor", "Medical Assistant", "Clinical Research Associate"],

    # Oil & Gas
    "oil and gas": ["Petroleum Engineer", "Drilling Supervisor", "HSE Officer", "Geologist"],

    # Project
    "project": ["Project Coordinator", "Project Manager", "Scrum Master"],

    # Travel
    "travel": ["Travel Consultant", "Tour Operator", "Ticketing Agent", "Travel Coordinator"],

    # Sales
    "sales": ["Sales Executive", "Account Manager", "Territory Sales Rep", "Business Development Executive"],

    # Business Development
    "business development": ["Business Development Manager", "Partnership Manager", "Growth Strategist"],

    # Retail
    "retail": ["Store Manager", "Retail Sales Associate", "Merchandiser", "Cashier"]
}


# --------------------------------------
# Core Functions
# --------------------------------------
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

def suggest_job_titles(job_text):
    job_text = job_text.lower()
    matches = []
    for keyword, roles in ROLE_KEYWORDS.items():
        if fuzz.partial_ratio(keyword, job_text) > 80:
            matches.extend(roles)
    return matches[:4]

# ✅ Fixed to extract lowercase tool/tech/skills too
def extract_keywords_from_jd(text):
    return list(set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9\+\#\-/]{2,}\b', text.lower())))

def detect_tools_missing(cv_text, jd_text):
    jd_keywords = extract_keywords_from_jd(jd_text)
    suggestions = []
    for tool in jd_keywords:
        if tool.lower() not in cv_text:
            suggestions.append(f"⬤ Mention your experience with {tool}.")
    return suggestions[:5]

def detect_soft_skills(cv_text, jd_text):
    soft_skills = {
        "communication": "⬤ Mention your communication skills (written or verbal).",
        "team": "⬤ Add examples of teamwork or collaboration.",
        "attention": "⬤ Include proof of attention to detail or quality.",
        "adapt": "⬤ Mention adaptability or thriving in fast-paced environments."
    }
    suggestions = []
    for k, msg in soft_skills.items():
        if k in jd_text and k not in cv_text:
            suggestions.append(msg)
    return suggestions

def detect_weak_language(cv_text):
    weak_phrases = ["helped with", "was responsible for", "involved in", "worked on"]
    suggestions = []
    for phrase in weak_phrases:
        if phrase in cv_text:
            suggestions.append(f"⬤ Replace weak phrase '{phrase}' with strong action verbs like 'led', 'executed', or 'owned'.")
    return suggestions

def detect_missing_metrics(cv_text):
    if not re.search(r'\d+%|\$\d+|\d+\s+(users|leads|visits|conversions|clients|projects)', cv_text):
        return ["⬤ Include measurable outcomes — like percentages, revenue impact, or user metrics."]
    return []

def detect_buzzwords(cv_text):
    buzzwords = ["go-getter", "synergy", "hardworking", "self-starter", "detail-oriented", "team player",
                 "results-driven", "dynamic", "think outside the box", "motivated", "fast learner", "passionate",
                 "proactive", "guru", "rockstar", "ninja", "innovative", "dedicated", "strategic thinker",
                 "problem solver", "excellent communicator", "multi-tasker", "thought leader", "visionary",
                 "leverage", "empowered", "value-added", "cutting-edge", "circle back", "drill down",
                 "move the needle", "game changer", "world-class", "culture fit", "self-motivated", "growth mindset",
                 "holistic approach", "tech-savvy", "track record", "dynamic individual", "result-oriented"]
    found = [word for word in buzzwords if word in cv_text.lower()]
    if found:
        return [f"⬤ Avoid buzzwords like: {', '.join(found)}. Use concrete examples instead."]
    return []

def generate_recommendations(cv_text, jd_text):
    # Same logic as before — insert all your `keyword_pairs`, `semantic_keywords`, etc.
    # For brevity, this is omitted here — but your previous long version works perfectly with this.

    # Placeholder:
    return detect_soft_skills(cv_text, jd_text) + detect_weak_language(cv_text) + detect_missing_metrics(cv_text)

def dynamic_summary(score):
    if score >= 85:
        return "Strong fit. Just polish your CV for clarity and impact."
    elif score >= 70:
        return "Good match. You're close — refine with a few skill or tool mentions."
    elif score >= 50:
        return "Decent alignment, but some key skills or achievements are missing."
    else:
        return "Significant gaps found. Add missing tools, skills, or clearer impact."

def final_summary(cv_text, jd_text):
    score = calculate_similarity(cv_text, jd_text)
    roles = suggest_job_titles(jd_text)
    suggestions = generate_recommendations(cv_text, jd_text)
    summary = dynamic_summary(score)
    return score, roles, suggestions, summary

# --------------------------------------
# Routes
# --------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    roles = []
    suggestions = []
    summary = ""
    error = None

    if request.method == 'POST':
        file = request.files.get('resume')
        jd_text = request.form.get('job_description', '').strip()

        if not jd_text:
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
                    score, roles, suggestions, summary = final_summary(cv_text.lower(), jd_text.lower())
                except Exception as e:
                    error = f"Error processing file: {str(e)}"

    return render_template_string('''
    <html>
    <head>
        <title>Smart CV Checker</title>
        <link href="https://fonts.googleapis.com/css2?family=Cabin&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Cabin', sans-serif; max-width: 800px; margin: auto; padding: 20px; line-height: 1.6; }
            textarea, input[type=file] { width: 100%; padding: 10px; margin: 10px 0; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
            ul { padding-left: 20px; }
        </style>
    </head>
    <body>
        <h2>Smart CV Checker</h2>
        <form method="post" enctype="multipart/form-data">
            <p><textarea name="job_description" rows="6" placeholder="Paste job description here..." required>{{ request.form.get('job_description', '') }}</textarea></p>
            <p><input type="file" name="resume" required></p>
            <p><button type="submit">Scan</button></p>
        </form>
        {% if score is not none %}
            <h3>ATS Score: {{ score }}%</h3>
            <p><strong>Summary:</strong> {{ summary }}</p>

            {% if roles %}
                <h4>Potential Roles That Align With This Vacancy</h4>
                <ul>
                {% for title in roles %}
                    <li>{{ title }}</li>
                {% endfor %}
                </ul>
            {% endif %}

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
    ''', score=score, roles=roles, suggestions=suggestions, summary=summary, error=error)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
