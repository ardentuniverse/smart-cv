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
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB limit

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
    if not cv_text or not job_text:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([job_text, cv_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

def suggest_job_titles(job_text):
    ROLE_KEYWORDS = {
        "seo": ["SEO Specialist", "SEO Analyst", "Technical SEO Executive"],
        "content": ["Content Writer", "Content Strategist", "Copywriter"],
        "recruitment": ["Recruiter", "Talent Sourcer", "HR Assistant"],
        "social media": ["Social Media Manager", "Community Manager", "Social Strategist"],
        "data": ["Data Analyst", "Data Scientist", "Research Assistant"],
        "ads": ["PPC Specialist", "Performance Marketer", "Paid Ads Manager"],
        "email": ["Email Marketer", "CRM Executive", "Lifecycle Marketing Specialist"],
        "copywriting": ["Copywriter", "Marketing Copywriter"],
        "wordpress": ["WordPress Developer", "Web Content Manager"],
        "graphics": ["Graphic Designer", "Visual Content Creator"],
        "python": ["Python Developer", "Data Engineer"],
        "research": ["UX Researcher", "Market Research Assistant"],
        "ui ux": ["UI Designer", "UX Designer", "Product Designer"],
        "project management": ["Project Coordinator", "Product Manager", "Scrum Master"],
        "customer": ["Customer Support Specialist", "Customer Success Manager"],
        "javascript": ["Frontend Developer", "React Developer", "JavaScript Engineer"]
    }
    matches = []
    for keyword, roles in ROLE_KEYWORDS.items():
        if keyword in job_text.lower():
            matches.extend(roles)
    return matches[:4]

def generate_recommendations(cv_text, job_text):
    recs = []
    cv_text = cv_text.lower()
    job_text = job_text.lower()

    keyword_pairs = [
        ("meta ads", "Add experience with Meta Ads or Facebook/Instagram paid campaigns."),
        ("linkedin ads", "Mention experience with LinkedIn Ads or paid social campaigns."),
        ("google ads", "Include hands-on experience with Google Ads or PPC platforms."),
        ("seo audit", "Describe SEO audit processes or tools like Screaming Frog, Ahrefs, or SEMrush."),
        ("cms", "Mention experience using a CMS like WordPress, Webflow, or similar."),
        ("google analytics", "Highlight your skills in Google Analytics or data dashboards."),
        ("remote", "Indicate comfort with remote work setups or self-directed projects."),
        ("fast-paced", "Include experience thriving in fast-paced environments or startups."),
        ("cross-functional", "Highlight projects involving collaboration with other teams (e.g., Sales, Product)."),
        ("content strategy", "Showcase your approach to planning and executing content strategies."),
        ("communication", "Mention your written or verbal communication strengths."),
        ("first class", "Include strong academic background or relevant certifications."),
        ("research", "Demonstrate research projects or market/UX analysis experience."),
        ("reporting", "Include examples of performance reporting, campaign summaries, or dashboards."),
        ("deadline", "Describe how you handle pressure and meet tight deadlines."),
    ]

    for keyword, message in keyword_pairs:
        if keyword in job_text and keyword not in cv_text:
            recs.append(message)

    return recs

def final_summary(cv_text, job_text):
    score = calculate_similarity(cv_text, job_text)
    roles = suggest_job_titles(job_text)
    suggestions = generate_recommendations(cv_text, job_text)
    return score, roles, suggestions

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    error = None
    roles = []
    suggestions = []

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
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)

                try:
                    if filename.lower().endswith('.pdf'):
                        cv_text = extract_text_from_pdf(path)
                    else:
                        cv_text = extract_text_from_docx(path)

                    if not error:
                        score, roles, suggestions = final_summary(cv_text, job_text)

                except Exception as e:
                    error = f"Error processing file: {str(e)}"

                finally:
                    if os.path.exists(path):
                        os.remove(path)

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
            </style>
        </head>
        <body>
            <h2>Smart CV</h2>
            <form method="post" enctype="multipart/form-data">
                <p><textarea name="job_description" rows="6" placeholder="Paste job description here..." required>{{ request.form.get('job_description', '') }}</textarea></p>
                <p><input type="file" name="resume" required></p>
                <p><button type="submit">Upload & Match</button></p>
            </form>
            {% if score is not none %}
                <h3>CV–JD Match Score: {{ score }}%</h3>

                {% if roles %}
                    <h4>Potential Roles That Align With This Profile</h4>
                    <ul>
                    {% for title in roles %}
                        <li>{{ title }}</li>
                    {% endfor %}
                    </ul>
                {% endif %}

                <h4>Smart Suggestions to Improve CV</h4>
                {% if suggestions %}
                    <ul>
                    {% for s in suggestions %}
                        <li>{{ s }}</li>
                    {% endfor %}
                    </ul>
                {% else %}
                    <p>No major gaps detected. Your CV covers most requirements, but fine-tuning could boost your score.</p>
                {% endif %}

                <h4>Final Thoughts</h4>
                {% if suggestions %}
                    <p>To improve alignment with this role, consider:</p>
                    <ul>
                        {% for s in suggestions %}
                            <li>{{ s }}</li>
                        {% endfor %}
                    </ul>
                    <p>This can raise your match score from {{ score }}% to above 90%.</p>
                {% else %}
                    <p>You're already covering many key areas from the job description.
                    For even better results, make sure your CV includes measurable impact, tools, and team contributions.</p>
                {% endif %}
            {% elif error %}
                <p style="color: red;"><strong>Error:</strong> {{ error }}</p>
            {% endif %}
        </body>
        </html>
    """, score=score, error=error, roles=roles, suggestions=suggestions)
    
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
