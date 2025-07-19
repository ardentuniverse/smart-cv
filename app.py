import os
import re
import logging
import secrets
import mimetypes
from flask import Flask, request, render_template, session, abort
from werkzeug.utils import secure_filename
from rapidfuzz import fuzz
import fitz  # PyMuPDF
import docx
from io import BytesIO
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Only run downloads if needed
try:
    stopwords.words('english')
    word_tokenize("test")
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

app = Flask(__name__, template_folder="templates")
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB limit

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Precompile regex groups for performance
KEYWORD_GROUPS = {k: [(re.compile(pattern, re.I), msg) for pattern, msg in v] for k, v in {
    "tech": [
        ("python|scripting|pandas|django", "Include Python scripting or automation experience (e.g., Pandas, Django)."),
        ("sql|database|query|mysql|postgresql", "Mention SQL or database querying skills (e.g., MySQL, PostgreSQL)."),
        ("api|rest|graphql", "Add RESTful API or GraphQL integration experience.")
    ],
    # ... rest unchanged ...
}.items()}

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return ' '.join([w for w in tokens if w.isalnum() and w not in stop_words])

def extract_text_from_pdf(stream):
    try:
        doc = fitz.open(stream=stream, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        if not text.strip():
            raise ValueError("Empty or unreadable PDF.")
        return text
    except Exception:
        raise ValueError("Invalid or corrupt PDF file.")

def extract_text_from_docx(stream):
    try:
        doc = docx.Document(stream)
        text = " ".join(p.text for p in doc.paragraphs).strip()
        if not text:
            raise ValueError("Empty or unreadable DOCX.")
        return text
    except Exception:
        raise ValueError("Invalid or corrupt DOCX file.")

def calculate_similarity(raw_cv, raw_jd):
    cv = preprocess_text(raw_cv)
    jd = preprocess_text(raw_jd)
    token = fuzz.token_set_ratio(cv, jd)
    part = fuzz.partial_ratio(cv, jd)
    return round(0.6 * token + 0.4 * part, 2)

def extract_jd_keywords(jd_raw, top_n=5):
    jd = preprocess_text(jd_raw)
    words = re.findall(r'\b\w+\b', jd)
    counts = Counter(words)
    return [w for w,_ in counts.most_common(top_n) if w not in stopwords.words('english')]

def generate_profession_suggestions(raw_cv, raw_jd, profession):
    cv = preprocess_text(raw_cv)
    jd = preprocess_text(raw_jd)
    suggestions = []

    for regex, msg in KEYWORD_GROUPS.get(profession, []):
        if regex.search(jd) and not regex.search(cv):
            suggestions.append(f"⚬ {msg}")

    for kw in extract_jd_keywords(raw_jd):
        if fuzz.partial_ratio(kw, jd) > 80 and fuzz.partial_ratio(kw, cv) < 80:
            suggestions.append(f"⚬ Include experience with '{kw}' from the job description.")

    return suggestions[:10]

def dynamic_summary(score, profession, suggestions):
    title = profession.replace('_', ' ').title()
    n = len(suggestions)
    if score >= 85:
        return f"Strong fit for {title}. Polish your CV with specific results."
    elif score >= 70:
        return f"Good match for {title}. Add {n} key skill(s) to improve."
    elif score >= 50:
        return f"Moderate fit for {title}. Focus on adding {n} missing skill(s)."
    else:
        return f"Significant gaps for {title}. Incorporate {n} key skill(s)."

def sanitize_input(text):
    return re.sub(r'<[^>]+>', '', text).strip()

@app.errorhandler(413)
def file_too_large(e):
    return render_template("index.html", error="File exceeds 1MB limit."), 413

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    suggestions = []
    summary = ""
    error = None
    session.setdefault('csrf_token', secrets.token_hex(16))

    if request.method == 'POST':
        token = request.form.get('csrf_token')
        if not token or token != session['csrf_token']:
            abort(400, "Invalid CSRF token.")
        profession = request.form.get("profession")
        jd_text = sanitize_input(request.form.get('job_description', ''))
        f = request.files.get('resume')

        if not profession:
            error = "Select a profession."
        elif not jd_text:
            error = "Paste a job description."
        elif not f:
            error = "Upload your resume."
        else:
            data = f.read()
            if len(data) > app.config['MAX_CONTENT_LENGTH']:
                error = "File exceeds 1MB limit."
            else:
                filename = secure_filename(f.filename or "")
                ext = os.path.splitext(filename)[1].lower()
                mime, _ = mimetypes.guess_type(filename)
                if ext not in ('.pdf', '.docx') or mime not in ('application/pdf',
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'):
                    error = "Upload a valid PDF or DOCX."
                else:
                    try:
                        stream = BytesIO(data)
                        cv_raw = extract_text_from_pdf(stream) if ext == '.pdf' else extract_text_from_docx(stream)
                        score = calculate_similarity(cv_raw, jd_text)
                        suggestions = generate_profession_suggestions(cv_raw, jd_text, profession)
                        summary = dynamic_summary(score, profession, suggestions)
                    except ValueError as ve:
                        error = str(ve)
                    except Exception as ex:
                        logger.error(f"File processing error: {ex}")
                        error = "Unexpected error, try another file."

    return render_template("index.html",
                           score=score, suggestions=suggestions,
                           summary=summary, error=error,
                           keyword_groups=KEYWORD_GROUPS,
                           csrf_token=session['csrf_token'])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
