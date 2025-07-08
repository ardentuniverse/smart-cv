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
        "flutter": ["Flutter Developer", "Mobile App Developer"],
        "frontend": ["Frontend Developer", "React Developer", "Web Engineer"],
        "full stack": ["Full Stack Developer", "Web Application Developer"],
        "it": ["IT Support Specialist", "Systems Analyst"],
        "mobile": ["Mobile Developer", "iOS/Android Developer"],
        "network": ["Network Administrator", "IT Infrastructure Engineer"],
        "systems": ["Systems Engineer", "Linux Admin"],
        "product manager": ["Product Manager", "Product Owner"],
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

        ## Engineering (Non-IT)
        "engineering": ["Mechanical Engineer", "Civil Engineer", "Electrical Engineer", "Structural Engineer"],

        ## Finance
        "finance": ["Financial Analyst", "Accountant", "Investment Analyst", "Treasury Manager"],

        ## Education
        "education": ["Teacher", "Lecturer", "Academic Coordinator", "Curriculum Developer"],

        ## Logistics & Supply Chain
        "logistics": ["Logistics Manager", "Supply Chain Analyst", "Warehouse Supervisor", "Inventory Planner"],

        ## Aviation
        "aviation": ["Pilot", "Flight Attendant", "Air Traffic Controller", "Aviation Safety Officer"],

        ## Banking
        "banking": ["Bank Teller", "Loan Officer", "Relationship Manager", "Branch Manager"],

        ## Real Estate
        "real estate": ["Real Estate Agent", "Property Manager", "Leasing Consultant", "Real Estate Analyst"],

        ## Admin & Secretarial
        "admin": ["Administrative Assistant", "Executive Assistant", "Office Manager", "Receptionist"],

        ## Catering
        "catering": ["Catering Manager", "Event Caterer", "Banquet Supervisor", "Catering Assistant"],

        ## Consultancy
        "consultancy": ["Management Consultant", "Business Consultant", "Strategy Consultant", "HR Consultant"],

        ## Customer Care
        "customer care": ["Customer Service Representative", "Call Center Agent", "Client Relations Officer"],

        ## Hospitality & Hotel
        "hospitality": ["Hotel Manager", "Concierge", "Front Desk Officer", "Housekeeping Supervisor"],

        ## Law
        "law": ["Lawyer", "Paralegal", "Legal Advisor", "Corporate Counsel"],

        ## Manufacturing
        "manufacturing": ["Production Supervisor", "Plant Manager", "Manufacturing Engineer", "Assembly Line Worker"],

        ## Healthcare & Medical
        "healthcare": ["Nurse", "Doctor", "Medical Assistant", "Clinical Research Associate"],

        ## Oil & Gas
        "oil and gas": ["Petroleum Engineer", "Drilling Supervisor", "HSE Officer", "Geologist"],

        ## Project Management
        "project": ["Project Coordinator", "Project Manager", "Scrum Master"],

        ## Travels & Aviation (Merged with aviation for some roles)
        "travel": ["Travel Consultant", "Tour Operator", "Ticketing Agent", "Travel Coordinator"],

        ## Sales
        "sales": ["Sales Executive", "Account Manager", "Territory Sales Rep", "Business Development Executive"],

        ## Business Development
        "business development": ["Business Development Manager", "Partnership Manager", "Growth Strategist"],

        ## Retail
        "retail": ["Store Manager", "Retail Sales Associate", "Merchandiser", "Cashier"]
    }

    job_text = job_text.lower()
    matches = []
    
    for keyword, roles in ROLE_KEYWORDS.items():
        if keyword in job_text:  # Fast exact substring check first
            matches.extend(roles)
        else:
            # If keyword not directly in text, do fuzzy match
            if fuzz.partial_ratio(keyword, job_text) > 80:
                matches.extend(roles)

    return matches[:4]  # Limit to top 4 matches

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
    buzzwords = ["go-getter", "synergy", "hardworking", "self-starter", "detail-oriented", "team player", "results-driven", "dynamic", "think outside the box", "motivated", "fast learner", "passionate", "proactive", "guru", "rockstar", "ninja", "innovative", "dedicated", "strategic thinker", "problem solver", "excellent communicator", "multi-tasker", "thought leader", "visionary", "leverage", "empowered", "value-added", "cutting-edge", "circle back", "drill down", "move the needle", "game changer", "world-class", "culture fit", "self-motivated", "growth mindset", "holistic approach", "tech-savvy", "track record", "dynamic individual", "result-oriented"]
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
        "cross-functional": ["other departments", "collaborate with sales", "worked with product team"],
    }    
    
    keyword_pairs = [     
        ("google ads", 9, "⬤ Add Google Ads or PPC experience."),
        ("seo audit", 8, "⬤ Include SEO audits or tools like Ahrefs."),
        ("cms", 7, "⬤ Mention CMS platforms like WordPress."),
        ("ga4", 4, "⬤ Add GA4 or web analytics experience."),
        ("meta ads", 8, "⬤ Add Meta Ads (Facebook/Instagram) campaign work."),
        ("linkedin ads", 7, "⬤ Mention LinkedIn Ads or B2B paid social."),
    ]

    keyword_pairs_soft_skills = [ 
        ("remote", 4, "⬤ Highlight remote work experience."),
        ("cross-functional", 6, "⬤ Include cross-team collaboration examples."),
        ("deadline", 4, "⬤ Describe how you meet deadlines under pressure."),
        ("communication", 5, "⬤ Add proof of communication skills."),
        ("problem solving", 5, "⬤ Demonstrate critical thinking or complex problem solving."),
        ("stakeholder", 6, "⬤ Mention working with internal or external stakeholders."),
        ("leadership", 6, "⬤ Add leadership, mentoring, or team management experience."),
        ("presentation", 5, "⬤ Include presentation or public speaking achievements."),
        ("multitask", 4, "⬤ Show your ability to manage multiple responsibilities."),
    ]
    
    keyword_pairs_tech = [         
        ("python", 8, "⬤ Include Python scripting or automation experience."),
        ("sql", 7, "⬤ Mention SQL or database querying skills."),
        ("aws", 8, "⬤ Add experience with AWS cloud services."),
        ("api", 7, "⬤ Add RESTful API integration or development work."),
        ("ci/cd", 6, "⬤ Mention CI/CD pipeline experience (e.g. Jenkins, GitHub Actions)."),
        ("git", 5, "⬤ Add Git version control or collaborative dev work."),
        ("react", 7, "⬤ Include React or component-based frontend experience."),
        ("agile", 6, "⬤ Reference Agile methodologies or Scrum processes."),
        ("docker", 6, "⬤ Add Docker or containerization experience."),
        ("analytics", 7, "⬤ Mention web, business, or product analytics work."),
    ]

    keyword_pairs_finance_banking = [     
        ("financial modeling", 8, "⬤ Add financial models or forecasting tools (e.g. Excel, Power BI)."),
        ("budgeting", 7, "⬤ Mention budget planning or cost control experience."),
        ("accounts payable", 6, "⬤ Include AP/AR processing or invoicing."),
        ("erp", 7, "⬤ List ERP systems used (e.g. SAP, Oracle, QuickBooks)."),
        ("compliance", 6, "⬤ Highlight financial/legal compliance responsibilities."),
        ("audit", 6, "⬤ Add experience preparing or undergoing audits."),
    ]
    
    keyword_pairs_education = [    
        ("curriculum", 7, "⬤ Mention curriculum design or learning material development."),
        ("lesson planning", 6, "⬤ Include structured lesson or training plans."),
        ("student engagement", 6, "⬤ Highlight engagement or interactive teaching techniques."),
        ("assessment", 5, "⬤ Add evaluation or grading experience."),
    ]
    
    keyword_pairs_logistics = [
        ("inventory", 7, "⬤ Add inventory control or stock management experience."),
        ("supply chain", 8, "⬤ Mention end-to-end supply chain processes."),
        ("logistics", 8, "⬤ Include logistics coordination or freight management."),
        ("warehouse", 6, "⬤ Describe warehouse or distribution experience."),
        ("procurement", 7, "⬤ Add vendor or purchase order management."),
    ]    
    
    keyword_pairs_medical = [    
        ("patient care", 8, "⬤ Include hands-on or remote patient care experience."),
        ("clinical", 7, "⬤ Mention clinical trials or procedures."),
        ("emr", 6, "⬤ Add EMR/EHR systems used (e.g. Epic, Cerner)."),
        ("triage", 6, "⬤ Highlight triage or emergency response experience."),
        ("pharma", 6, "⬤ Include pharmaceutical or drug safety background."),
    ]    

    keyword_pairs_legal = [
        ("compliance", 7, "⬤ Highlight regulatory or legal compliance tasks."),
        ("contract", 7, "⬤ Include contract drafting, review, or negotiation."),
        ("legal research", 6, "⬤ Mention legal research or case analysis experience."),
        ("paralegal", 6, "⬤ Add paralegal or legal assistant tasks."),
    ]    
    
    keyword_pairs_hospitality = [    
        ("guest service", 8, "⬤ Highlight customer/guest satisfaction focus."),
        ("reservation", 6, "⬤ Mention reservation systems or booking platforms used."),
        ("event planning", 7, "⬤ Add event or banquet coordination experience."),
        ("food safety", 6, "⬤ Include HACCP or other safety certification."),
    ]    
    
    keyword_pairs_oil_gas = [    
        ("drilling", 7, "⬤ Include drilling operations or well site management."),
        ("hse", 8, "⬤ Mention Health, Safety & Environment responsibilities."),
        ("pipeline", 6, "⬤ Add pipeline inspection or maintenance work."),
        ("rig", 7, "⬤ Describe offshore/onshore rig experience."),
        ("geology", 6, "⬤ Include geological survey or subsurface mapping."),    
    ]
    
    keyword_pairs_sales_business_retail = [  
        ("kpi", 7, "⬤ Add KPIs or targets you’ve exceeded."),
        ("crm", 7, "⬤ Mention CRM platforms like Salesforce, HubSpot."),
        ("lead generation", 8, "⬤ Highlight prospecting or lead-gen activities."),
        ("negotiation", 6, "⬤ Include negotiation or closing deals."),
        ("upsell", 5, "⬤ Add upselling or cross-selling metrics."),
        ("retail", 6, "⬤ Mention store operations or customer interaction."),
    ]

    all_keyword_pairs = [
        keyword_pairs,
        keyword_pairs_soft_skills,
        keyword_pairs_tech,
        keyword_pairs_finance_banking,
        keyword_pairs_education,
        keyword_pairs_logistics,
        keyword_pairs_medical,
        keyword_pairs_legal,
        keyword_pairs_hospitality,
        keyword_pairs_oil_gas,
        keyword_pairs_sales_business_retail,
    ]    

    for keyword_list in all_keyword_pairs:
        for keyword, score, msg in keyword_list:
            if (keyword in jd_text or any(syn in jd_text for syn in semantic_keywords.get(keyword, []))):
                if not (keyword in cv_text or any(syn in cv_text for syn in semantic_keywords.get(keyword, []))):
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
