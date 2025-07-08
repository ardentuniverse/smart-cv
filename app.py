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
        "graphics": ["Graphic Designer"],
    }

    job_text = job_text.lower()
    matches = []
    for keyword, roles in ROLE_KEYWORDS.items():
        if fuzz.partial_ratio(keyword, job_text) > 80:
            matches.extend(roles)
    return matches[:4]

def detect_tools_missing(cv_text, jd_text):
    tools = {
        "google analytics": "Google Analytics",
        "ga4": "Google Analytics 4",
        "figma": "Figma",
        "sql": "SQL",
        "wordpress": "WordPress",
        "mailchimp": "Mailchimp",
        "trello": "Trello",
        "hubspot": "Hubspot",
        "notion": "Notion",
        "photoshop": "Photoshop"
    }
    suggestions = []
    for k, v in tools.items():
        if k in jd_text and k not in cv_text:
            suggestions.append(f"Add your experience with {v} if you’ve used it.")
    return suggestions

def detect_soft_skills(cv_text, jd_text):
    soft_skills = {
        "communication": "Mention your communication skills (written or verbal).",
        "team": "Add examples of teamwork or collaboration.",
        "attention": "Include details showing attention to detail or quality.",
        "adapt": "Mention adaptability or working in fast-paced environments."
    }
    suggestions = []
    for k, msg in soft_skills.items():
        if k in jd_text and k not in cv_text:
            suggestions.append(msg)
    return suggestions

def detect_weak_language(cv_text):
    strong_verbs = ["led", "managed", "designed", "built", "analyzed", "launched", "improved", "developed", "created"]
    if not any(verb in cv_text for verb in strong_verbs):
        return ["Use stronger action verbs like 'led', 'built', or 'improved' to describe your impact."]
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
        ("google ads", 9, "Add Google Ads or PPC experience."),
        ("seo audit", 8, "Include SEO audits or tools like Ahrefs."),
        ("cms", 7, "Mention CMS platforms like WordPress."),
        ("ga4", 7, "Add experience with GA4 or web analytics."),
        ("meta ads", 8, "Add Meta Ads (Facebook/Instagram) campaign work."),
        ("linkedin ads", 7, "Mention LinkedIn Ads or B2B paid social."),
        ("remote", 4, "State experience with remote or async work."),
        ("cross-functional", 6, "Highlight collaboration with other teams."),
        ("communication", 5, "Mention strong communication or storytelling."),
        ("deadline", 4, "Describe handling pressure or timelines."),
    ]

    # Direct + synonym checks
    for keyword, score, msg in keyword_pairs:
        if (keyword in jd_text or any(syn in jd_text for syn in semantic_keywords.get(keyword, []))) and keyword not in cv_text:
            recs.append((score, msg))

    recs.sort(reverse=True)
    messages = [msg for _, msg in recs[:6]]

    # Add tool, soft skill, and language recs
    messages += detect_tools_missing(cv_text, jd_text)
    messages += detect_soft_skills(cv_text, jd_text)
    messages += detect_weak_language(cv_text)

    return messages[:8]

def dynamic_summary(score):
    if score >= 85:
        return "Strong fit. Just polish your CV for clarity and impact."
    elif score >= 70:
        return "Good match. You're close — refine with a few skill or tool mentions."
    elif score >= 50:
        return "Decent alignment, but some key skills or achievements are missing."
    else:
        return "Significant gaps found. Rework your CV to better match this job."

def final_summary(cv_text, jd_text):
    score = calculate_similarity(cv_text, jd_text)
    roles = suggest_job_titles(jd_text)
    suggestions = generate_recommendations(cv_text, jd_text)
    summary = dynamic_summary(score)
    return score, roles, suggestions, summary

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    error = None
    roles = []
    suggestions = []
    summary = ""

    if request.method == 'POST':
        file = request.files.get('resume')
        job_text = request.form.get('job_description', '').strip()

        if not job_text:
            error = "Please paste a job description."
        elif not file:
            error = "Please upload a CV file."
        else:
            filename = secure_filename(file.filename)

            if not filename.lower().endswith(('.pdf', '.docx')):
                error = "Unsupported file format. Use PDF or DOCX."
            else:
                try:
                    file_stream = BytesIO(file.read())

                    if filename.lower().endswith('.pdf'):
                        cv_text = extract_text_from_pdf(file_stream)
                    else:
                        cv_text = extract_text_from_docx(file_stream)

                    score, roles, suggestions, summary = final_summary(cv_text, job_text)

                except Exception as e:
                    error = f"Error processing file: {str(e)}"

    return render_template_string("""
    <html>
    <head>
        <title>Smart CV Matcher</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta charset="UTF-8">
        <link href="https://fonts.googleapis.com/css2?family=Cabin&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Cabin', sans-serif; max-width: 800px; margin: auto; padding: 20px; line-height: 1.6; }
            textarea, input[type=file] { width: 100%; padding: 10px; margin: 10px 0; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
            ul { padding-left: 20px; }
        </style>
    </head>
    <body>
        <h2>Smart CV Matcher</h2>
        <form method="post" enctype="multipart/form-data">
            <p><textarea name="job_description" rows="6" placeholder="Paste job description here..." required>{{ request.form.get('job_description', '') }}</textarea></p>
            <p><input type="file" name="resume" required></p>
            <p><button type="submit">Upload & Match</button></p>
        </form>
        {% if score is not none %}
            <h3>Match Score: {{ score }}%</h3>
            <p><strong>Summary:</strong> {{ summary }}</p>

            {% if roles %}
                <h4>Potential Roles That Align With This Profile</h4>
                <ul>{% for title in roles %}<li>{{ title }}</li>{% endfor %}</ul>
            {% endif %}

            {% if suggestions %}
                <h4>Suggestions to Improve CV</h4>
                <ul>{% for s in suggestions %}<li>{{ s }}</li>{% endfor %}</ul>
            {% else %}
                <p>You're covering most of what the job requires — just fine-tune your language and achievements.</p>
            {% endif %}
        {% elif error %}
            <p style="color:red;"><strong>Error:</strong> {{ error }}</p>
        {% endif %}
    </body>
    </html>
    """, score=score, roles=roles, suggestions=suggestions, error=error, summary=summary)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
