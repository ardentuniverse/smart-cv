from flask import Flask, request, render_template_string 
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util
import spacy
import fitz  # PyMuPDF
import docx
from io import BytesIO
import os
import re
from functools import lru_cache

# === INIT ===
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# === MODEL LOADERS ===
@lru_cache()
def get_nlp():
    return spacy.load("en_core_web_sm")

@lru_cache()
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# === FILE UTILS ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_bytes):
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def extract_text_from_docx(file_bytes):
    file_stream = BytesIO(file_bytes)
    doc = docx.Document(file_stream)
    return " ".join([p.text for p in doc.paragraphs]).strip()

# === NLP CORE ===
def semantic_similarity(text1, text2):
    emb1 = get_embedder().encode(text1, convert_to_tensor=True)
    emb2 = get_embedder().encode(text2, convert_to_tensor=True)
    return round(util.cos_sim(emb1, emb2).item() * 100, 2)

def extract_skills(text):
    doc = get_nlp()(text)
    return set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2)

# === BASIC SUGGESTIONS ===
def smart_recommendations(cv_text, jd_text):
    jd_skills = extract_skills(jd_text)
    cv_skills = extract_skills(cv_text)
    missing = jd_skills - cv_skills
    return [f"Consider adding or emphasizing: **{skill}**" for skill in sorted(missing)][:10]

# === STRATEGIC, HUMAN-LIKE SUGGESTIONS ===
def strategic_recommendations(cv_text, jd_text, industry_guess="marketing"):
    suggestions = []
    jd_text = jd_text.lower()
    cv_text = cv_text.lower()

    def jd_mentions(phrases):
        return any(p in jd_text for p in phrases)

    def cv_lacks(phrases):
        return all(p not in cv_text for p in phrases)

    # B2B
    if jd_mentions(['b2b']) and cv_lacks(['b2b']):
        suggestions.append("Highlight B2B Experience: The job emphasizes B2B campaigns. Reframe any applicable projects or client work in a B2B context.")

    # Tech industry
    if jd_mentions(['tech', 'saas', 'startup']) and not any(w in cv_text for w in ['tech', 'software', 'saas']):
        suggestions.append("Emphasize Tech Industry Focus: Expand on any tech-related projects or tools you've worked with.")

    # Graphic design tools
    if jd_mentions(['canva', 'adobe', 'graphic design']) and cv_lacks(['canva', 'adobe']):
        suggestions.append("Add Graphic Design Skills: Mention Canva, Adobe, or similar tools if used — these are requested.")

    # LinkedIn Ads
    if jd_mentions(['linkedin campaign manager']) and cv_lacks(['linkedin ads', 'linkedin campaign']):
        suggestions.append("Mention LinkedIn Campaign Manager: Include this if you’ve managed LinkedIn Ads or similar B2B platforms.")

    # Collaboration
    if jd_mentions(['collaborate', 'cross-functional']) and not re.search(r'collaborat|cross[- ]functional|stakeholder', cv_text):
        suggestions.append("Emphasize Cross-Functional Collaboration: Add examples of working with tech, talent, or cross-department teams.")

    # Digital awareness
    if jd_mentions(['digital transformation', 'emerging tech', 'data-driven']) and not re.search(r'digital transformation|emerging tech|data[- ]driven', cv_text):
        suggestions.append("Show Digital Trend Awareness: Add a line about your interest or experience in data-driven or emerging tech.")

    # CMS tools
    if jd_mentions(['cms', 'wordpress', 'webflow']) and cv_lacks(['wordpress', 'webflow']):
        suggestions.append("Mention CMS Tools: Include WordPress, Webflow, or similar if you've used them — they're required.")

    # Campaign KPIs
    if jd_mentions(['kpi', 'roas', 'cpl', 'ctr']) and not re.search(r'roas|cpl|ctr|conversion rate', cv_text):
        suggestions.append("Quantify KPIs: Add campaign metrics like ROAS, CTR, or CPL to match the job's emphasis on performance.")

    # Certifications
    if jd_mentions(['certification', 'certified', 'google ads']) and not re.search(r'certification|google ads|linkedin ads', cv_text):
        suggestions.append("Add Certifications: Mention relevant certs like Google Ads, Analytics, or LinkedIn Ads to increase alignment.")

    return suggestions

# === ROUTES ===
@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    suggestions = []
    error = None

    if request.method == 'POST':
        job_text = request.form.get('job_description', '').strip()
        file = request.files.get('resume')

        if not job_text:
            error = "Please paste a job description."
        elif not file or not allowed_file(file.filename):
            error = "Upload a valid resume file (PDF or DOCX)."
        else:
            try:
                file_bytes = file.read()
                filename = secure_filename(file.filename)

                if filename.lower().endswith(".pdf"):
                    cv_text = extract_text_from_pdf(file_bytes)
                else:
                    cv_text = extract_text_from_docx(file_bytes)

                if not cv_text:
                    error = "We couldn't extract text from your CV. Try another file."
                else:
                    score = semantic_similarity(cv_text, job_text)
                    suggestions = smart_recommendations(cv_text, job_text)
                    suggestions += strategic_recommendations(cv_text, job_text)

            except Exception as e:
                error = f"Processing error: {str(e)}"

    return render_template_string("""
        <html>
        <head>
            <title>Smart CV Matcher</title>
            <link href="https://fonts.googleapis.com/css2?family=Cabin&display=swap" rel="stylesheet">
            <style>
                body { font-family: 'Cabin', sans-serif; max-width: 800px; margin: auto; padding: 20px; line-height: 1.6; }
                textarea, input[type=file] { width: 100%; padding: 10px; margin: 10px 0; }
                button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
                ul { padding-left: 20px; }
                li { margin-bottom: 10px; }
            </style>
        </head>
        <body>
            <h2>Smart CV</h2>
            <form method="post" enctype="multipart/form-data">
                <p><textarea name="job_description" rows="6" placeholder="Paste job description here..." required>{{ request.form.get('job_description', '') }}</textarea></p>
                <p><input type="file" name="resume" required></p>
                <p><button type="submit">Analyze</button></p>
            </form>

            {% if score is not none %}
                <h3>Semantic Match Score: {{ score }}%</h3>
                <h4>Recommended Changes to Improve CV Match:</h4>
                {% if suggestions %}
                    <ul>
                    {% for s in suggestions %}
                        <li>{{ s|safe }}</li>
                    {% endfor %}
                    </ul>
                {% else %}
                    <p>Your CV aligns well with this job. No critical gaps detected!</p>
                {% endif %}
            {% elif error %}
                <p style="color: red;"><strong>Error:</strong> {{ error }}</p>
            {% endif %}
        </body>
        </html>
    """, score=score, suggestions=suggestions, error=error)

# === MAIN ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
