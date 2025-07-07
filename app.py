from flask import Flask, request, render_template_string, abort
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

# Job description for matching
JOB_DESCRIPTION = """We are looking for a React developer experienced with GraphQL and TypeScript."""

def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return " ".join([p.text for p in doc.paragraphs]).strip()

def calculate_similarity(cv_text, job_text):
    if not cv_text:
        return 0.0
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_text, cv_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    error = None

    if request.method == 'POST':
        file = request.files.get('resume')
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            try:
                if filename.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(path)
                elif filename.lower().endswith('.docx'):
                    text = extract_text_from_docx(path)
                else:
                    error = "Unsupported file format. Please upload a PDF or DOCX file."
                    text = ""

                if not error:
                    score = calculate_similarity(text, JOB_DESCRIPTION)

            except Exception as e:
                error = f"Error processing file: {str(e)}"

    return render_template_string("""
        <h2>Smart CV Matcher (Lite)</h2>
        <form method="post" enctype="multipart/form-data">
            <p><input type="file" name="resume" required></p>
            <p><button type="submit">Upload & Match</button></p>
        </form>
        {% if score is not none %}
            <h3>Match Score: {{ score }}%</h3>
        {% elif error %}
            <p style="color: red;"><strong>Error:</strong> {{ error }}</p>
        {% endif %}
    """, score=score, error=error)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
