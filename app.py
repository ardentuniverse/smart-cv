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
    ROLE_KEYWORDS = {
        "seo": ["SEO Specialist", "SEO Analyst"],
        "content": ["Content Writer", "Copywriter"],
        "recruit": ["Recruiter", "HR Assistant"],
        "data": ["Data Analyst", "Research Assistant"],
        "ads": ["PPC Specialist", "Paid Ads Manager"],
        "email": ["Email Marketer"],
        "python": ["Python Developer", "Data Engineer"],
        "ui ux": ["UI Designer", "UX Designer"],
        "customer": ["Customer Support", "Client Success Rep"],
        "javascript": ["Frontend Developer", "React Developer"],
        "project": ["Project Coordinator", "Scrum Master"],
        "wordpress": ["WordPress Developer"],
        "graphics": ["Graphic Designer"]
    }
    job_text = job_text.lower()
    matches = []
    for keyword, roles in ROLE_KEYWORDS.items():
        if fuzz.partial_ratio(keyword, job_text) > 80:
            matches.extend(roles)
    return matches[:4]

def extract_keywords_from_jd(text):
    return list(set(re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b', text)))

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
    buzzwords = ["go-getter", "synergy", "hardworking", "self-starter", "detail-oriented", "team player"]
    found = [word for word in buzzwords if word in cv_text.lower()]
    if found:
        return [f"⬤ Avoid buzzwords like: {', '.join(found)}. Use concrete examples instead."]
    return []

def generate_recommendations(cv_text, jd_text):
    recs = []
    cv_text = cv_text.lower()
    jd_text = jd_text.lower()

    semantic_keywords = {
        "google ads": ["performance marketing", "sem", "ppc"],
        "seo audit": ["seo review", "site audit"],
        "meta ads": ["facebook ads", "instagram campaigns"],
        "cms": ["wordpress", "webflow"],
        "ga4": ["google analytics 4"],
        "cross-functional": ["other departments", "collaborate with sales", "worked with product team"]
    }

    keyword_pairs = [
        ("google ads", 9, "⬤ Add Google Ads or PPC experience."),
        ("seo audit", 8, "⬤ Include SEO audits or tools like Ahrefs."),
        ("cms", 7, "⬤ Mention CMS platforms like WordPress."),
        ("ga4", 7, "⬤ Add GA4 or web analytics experience."),
        ("meta ads", 8, "⬤ Add Meta Ads (Facebook/Instagram) campaign work."),
        ("linkedin ads", 7, "⬤ Mention LinkedIn Ads or B2B paid social."),
        ("remote", 4, "⬤ Highlight remote or async work experience."),
        ("cross-functional", 6, "⬤ Include cross-team collaboration examples."),
        ("communication", 5, "⬤ Add proof of communication skills."),
        ("deadline", 4, "⬤ Describe how you meet deadlines under pressure.")
    ]

    for keyword, score, msg in keyword_pairs:
        if (keyword in jd_text or any(syn in jd_text for syn in semantic_keywords.get(keyword, []))) and keyword not in cv_text:
            recs.append((score, msg))

    recs.sort(reverse=True)
    messages = [msg for _, msg in recs[:6]]
    messages += detect_tools_missing(cv_text, jd_text)
    messages += detect_soft_skills(cv_text, jd_text)
    messages += detect_weak_language(cv_text)
    messages += detect_missing_metrics(cv_text)
    messages += detect_buzzwords(cv_text)

    return messages[:10]

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
                    score, roles, suggestions, summary = final_summary(cv_text, jd_text)
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
            <h3>Match Score: {{ score }}%</h3>
            <p><strong>Summary:</strong> {{ summary }}</p>

            {% if roles %}
                <h4>Potential Roles That Align With This Profile</h4>
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
