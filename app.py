from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
import os
import fitz  # PyMuPDF
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# This is the job description to match against
JOB_DESCRIPTION = """We are looking for a React developer experienced with GraphQL and TypeScript."""

def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return " ".join([p.text for p in doc.paragraphs])

def calculate_similarity(cv_text, job_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_text, cv_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    if request.method == 'POST':
        file = request.files['resume']
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(path)
            elif filename.endswith('.docx'):
                text = extract_text_from_docx(path)
            else:
                text = ""

            score = calculate_similarity(text, JOB_DESCRIPTION)

    return render_template_string("""
        <h2>Smart CV Matcher (Lite)</h2>
        <form method="post" enctype="multipart/form-data">
            <p><input type="file" name="resume" required></p>
            <p><button type="submit">Upload & Match</button></p>
        </form>
        {% if score is not none %}
            <h3>Match Score: {{ score }}%</h3>
        {% endif %}
    """, score=score)

if __name__ == '__main__':
    app.run(debug=True)
