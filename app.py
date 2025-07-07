from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
import os
import fitz  # PyMuPDF
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB limit

# Extract text from PDF
def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# Extract text from DOCX
def extract_text_from_docx(path):
    doc = docx.Document(path)
    return " ".join([p.text for p in doc.paragraphs]).strip()

# Calculate TF-IDF similarity score
def calculate_similarity(cv_text, job_text):
    if not cv_text or not job_text:
        return 0.0
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_text, cv_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

# Get missing keywords from job description not found in CV
def get_missing_keywords(cv_text, job_text):
    stopwords = {
        "the", "and", "to", "in", "of", "for", "on", "with", "a", "an",
        "is", "at", "by", "this", "that", "are", "as", "be", "from", "or"
    }
    job_words = set(word.lower().strip(",.():;") for word in job_text.split())
    cv_words = set(word.lower().strip(",.():;") for word in cv_text.split())
    keywords = job_words - cv_words
    keywords = [word for word in keywords if word not in stopwords and len(word) > 2]
    return sorted(keywords)

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    error = None
    missing_keywords = []

    if request.method == 'POST':
        file = request.files.get('resume')
        job_text = request.form.get('job_description', '').strip()

        if not job_text:
            error = "Please paste a job description."
        elif not file:
            error = "Please upload a CV file."
        else:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            try:
                if filename.lower().endswith('.pdf'):
                    cv_text = extract_text_from_pdf(path)
                elif filename.lower().endswith('.docx'):
                    cv_text = extract_text_from_docx(path)
                else:
                    error = "Unsupported file format. Use PDF or DOCX."
                    cv_text = ""

                if not error:
                    score = calculate_similarity(cv_text, job_text)
                    missing_keywords = get_missing_keywords(cv_text, job_text)

            except Exception as e:
                error = f"Error processing file: {str(e)}"

    return render_template_string("""
        <style>
          body {
            font-family: Arial, sans-serif;
            max-width: 700px;
            margin: 2rem auto;
            padding: 1.5rem;
            background: #fafafa;
            color: #333;
            border-radius: 8px;
            box-shadow: 0 0 8px rgba(0,0,0,0.05);
          }
          textarea, input[type="file"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 1rem;
            font-size: 1rem;
          }
          button {
            padding: 10px 20px;
            background-color: #0b5ed7;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
          }
          button:hover {
            background-color: #094db1;
          }
          h2, h3, h4 {
            color: #0b5ed7;
          }
          ul {
            padding-left: 1.2rem;
          }
          li {
            margin-bottom: 0.3rem;
          }
        </style>

        <h2>Smart CV Matcher (Lite)</h2>
        <form method="post" enctype="multipart/form-data">
            <label><strong>Paste Job Description</strong></label>
            <textarea name="job_description" rows="6" placeholder="E.g. We're hiring a backend developer skilled in Django, REST APIs, and PostgreSQL..." required>{{ request.form.get('job_description', '') }}</textarea>
            
            <label><strong>Upload Your CV (PDF or DOCX)</strong></label>
            <input type="file" name="resume" required>
            
            <button type="submit">Upload & Match</button>
        </form>

        {% if score is not none %}
          <h3>Match Score: {{ score }}%</h3>
        {% endif %}

        {% if missing_keywords %}
          <h4>Suggested Keywords to Add:</h4>
          <ul>
          {% for word in missing_keywords[:15] %}
              <li>{{ word }}</li>
          {% endfor %}
          </ul>
        {% elif error %}
          <p style="color: red;"><strong>Error:</strong> {{ error }}</p>
        {% endif %}
    """, score=score, error=error, missing_keywords=missing_keywords)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
