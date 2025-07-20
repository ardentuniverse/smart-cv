from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util
import spacy
import fitz  # PyMuPDF
import docx
import os
import re
from io import BytesIO
from functools import lru_cache

# Load model once globally
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return model.encode(text, convert_to_tensor=True)

def calculate_score(cv_text, jd_text):
    cv_embedding = embed_text(cv_text)
    jd_embedding = embed_text(jd_text)
    similarity = util.cos_sim(cv_embedding, jd_embedding).item()
    return round(similarity * 100, 2)

# Semantic smart match helper
def is_semantic_match(jd_text, cv_text, keyword):
    jd_score = util.cos_sim(embed_text(jd_text), embed_text(keyword)).item()
    cv_score = util.cos_sim(embed_text(cv_text), embed_text(keyword)).item()
    return jd_score > 0.3 and cv_score < 0.25

# === INIT ===
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

# === UTILS ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file):
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif ext == 'docx':
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

@lru_cache(maxsize=128)
def embed_text(text):
    return model.encode(text, convert_to_tensor=True)

def missing_keyword(word, jd, cv):
    return word in jd and word not in cv

def missing_any_keyword(keywords, jd, cv):
    return any(k in jd for k in keywords) and not any(k in cv for k in keywords)

# === SMART SUGGESTIONS ===
import re

def generate_suggestions(cv_text, jd_text, field):
    # Lowercase and strip punctuation for safer matching
    cv_lower = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', cv_text.lower()))
    jd_lower = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', jd_text.lower()))
    suggestions = []

    FIELD_RULES = {
        "Administration / Secretarial": [
            {
                'title': 'Calendar Management',
                'keywords': ['calendar'],
                'feedback': 'Mention experience with scheduling or managing executive calendars.',
                'example': 'Managed complex calendars, scheduled meetings, and coordinated appointments for senior executives.'
            },
            {
                'title': 'Microsoft Office Tools',
                'keywords': ['word', 'excel', 'powerpoint'],
                'feedback': 'Mention your proficiency with MS Word, Excel, or PowerPoint.',
                'example': 'Created reports and managed budgets using Microsoft Excel and PowerPoint.'
            },
            {
                'title': 'Communication Skills',
                'keywords': ['communication'],
                'feedback': 'Demonstrate written and verbal communication skills relevant to office correspondence.',
                'example': 'Drafted executive correspondence and handled official communication for departmental activities.'
            },
            {
                'title': 'Travel Arrangements',
                'keywords': ['travel'],
                'feedback': 'If applicable, mention booking or coordinating travel for teams or executives.',
                'example': 'Coordinated local and international travel arrangements for the regional director.'
            }
        ],
        "Agriculture / Agro-Allied": [
            {
                'title': 'Agricultural Equipment Experience',
                'keywords': ['tractor'],
                'feedback': 'Mention any experience using tractors, planters, sprayers, or irrigation systems.',
                'example': 'Operated and maintained tractors for crop planting and spraying.'
            },
            {
                'title': 'Livestock or Aquaculture Experience',
                'keywords': ['livestock', 'poultry', 'fish', 'aquaculture'],
                'feedback': 'Highlight any hands-on work with poultry, fish farming, or livestock.',
                'example': 'Managed daily operations in a 500-bird poultry farm including feeding, vaccination, and sales.'
            },
            {
                'title': 'Yield Improvement Achievements',
                'keywords': ['yield'],
                'feedback': 'Showcase any efforts that led to increased farm productivity.',
                'example': 'Improved cassava yields by 25% through adoption of improved varieties and soil testing.'
            },
            {
                'title': 'AgriTech Tools',
                'keywords': ['agritech', 'gis', 'drone'],
                'feedback': 'Mention if you have worked with drones, GIS software, or digital farm mapping tools.',
                'example': 'Used drone imagery and GIS tools to monitor field health and optimize fertilizer use.'
            },
            {
                'title': 'Agricultural Extension Work',
                'keywords': ['extension'],
                'feedback': 'Mention any experience in community training or farmer outreach programs.',
                'example': 'Conducted farmer training sessions under a USAID extension program for maize farmers.'
            },
            {
                'title': 'Agricultural Value Chain Exposure',
                'keywords': ['value chain'],
                'feedback': 'Highlight experience across any segment: production, processing, marketing, or export.',
                'example': 'Oversaw cassava processing and linked producers to off-takers through cooperative channels.'
            }
        ],
        "Aviation / Airline": [
            {
                'title': 'Aviation Safety Compliance',
                'keywords': ['safety'],
                'feedback': 'Mention any safety protocol training or adherence to aviation regulations.',
                'example': 'Ensured compliance with FAA/ICAO safety standards during all ground operations.'
            },
            {
                'title': 'Cabin Crew Experience',
                'keywords': ['flight attendant', 'cabin crew'],
                'feedback': 'If you’ve worked as cabin crew or in-flight services, highlight it clearly.',
                'example': 'Served as lead cabin crew on over 500 domestic and international flights, ensuring passenger comfort and safety.'
            },
            {
                'title': 'Ticketing and Reservations',
                'keywords': ['ticketing'],
                'feedback': 'Mention your experience with airline reservation systems (e.g., Amadeus, Sabre).',
                'example': 'Handled reservations and ticketing using Amadeus GDS for over 150 daily bookings.'
            },
            {
                'title': 'Ground Operations Support',
                'keywords': ['ground operations', 'ground support'],
                'feedback': 'Highlight tasks related to baggage, boarding, or ramp services.',
                'example': 'Coordinated baggage handling, boarding processes, and marshaling at Lagos International Airport.'
            },
            {
                'title': 'Air Traffic Knowledge',
                'keywords': ['air traffic'],
                'feedback': 'Even if you’re not a controller, referencing basic familiarity with ATC procedures can help.',
                'example': 'Worked closely with air traffic controllers to streamline pilot communications during runway congestion.'
            }
        ],
        "Banking": [
            {
                'title': 'Credit Analysis / Risk Assessment',
                'keywords': ['credit analysis', 'loan', 'risk profile', 'credit risk'],
                'feedback': 'Highlight your experience evaluating loan applications, risk profiles, or conducting creditworthiness assessments.',
                'example': 'Conducted credit analysis for SME loan applications, including cash flow projections and risk scoring.'
            },
            {
                'title': 'Regulatory Compliance',
                'keywords': ['compliance', 'regulatory', 'cbn'],
                'feedback': 'Mention your knowledge of CBN regulations, anti-money laundering (AML), or KYC policies.',
                'example': 'Ensured full compliance with CBN regulatory guidelines, including AML/KYC documentation for all clients.'
            },
            {
                'title': 'Customer Service in Banking',
                'keywords': ['customer service'],
                'feedback': 'Emphasize any client-facing roles or experience resolving customer issues in a banking context.',
                'example': 'Handled daily customer interactions, resolving account issues and improving satisfaction scores by 15%.'
            },
            {
                'title': 'Investment or Portfolio Management',
                'keywords': ['investment', 'portfolio', 'asset'],
                'feedback': 'Include your experience managing client portfolios, financial advisory, or wealth planning.',
                'example': 'Managed a portfolio of high-net-worth clients, delivering investment returns above benchmark by 6%.'
            },
            {
                'title': 'Financial Statement Analysis',
                'keywords': ['financial statement', 'balance sheet', 'p&l'],
                'feedback': 'Mention skills in analyzing balance sheets, income statements, or financial ratios.',
                'example': 'Reviewed company balance sheets and cash flow reports as part of corporate loan assessments.'
            },
            {
                'title': 'Sales Target Achievement',
                'keywords': ['sales target'],
                'feedback': 'If relevant, add metrics showing your contribution to product sales, deposits, or cross-selling in banking.',
                'example': 'Achieved 120% of monthly deposit mobilization targets and cross-sold 40+ insurance packages.'
            }
        ],
        "Catering / Confectionery": [
            {
                'title': 'Menu Planning',
                'keywords': ['menu planning', 'menu'],
                'feedback': 'Mention experience in creating or planning menus for events or daily operations.',
                'example': 'Planned customized menus for weddings, corporate events, and daily meal services.'
            },
            {
                'title': 'Food Safety and Hygiene',
                'keywords': ['food safety', 'hygiene', 'sanitation'],
                'feedback': 'Include any experience or certification related to food hygiene or safety standards.',
                'example': 'Maintained strict hygiene protocols in line with NAFDAC and HACCP standards.'
            },
            {
                'title': 'Baking Experience',
                'keywords': ['baking'],
                'feedback': 'Highlight specific baking skills or types of products (e.g., cakes, pastries, bread).',
                'example': 'Baked and decorated over 300 cakes and pastries monthly for a high-volume bakery.'
            },
            {
                'title': 'Inventory Management',
                'keywords': ['inventory'],
                'feedback': 'Mention tracking, ordering, or stock management of kitchen or baking supplies.',
                'example': 'Managed stock levels and ensured timely ordering of baking ingredients and materials.'
            },
            {
                'title': 'Catering Event Experience',
                'keywords': ['catering service', 'catering'],
                'feedback': 'Include any experience managing or working in event-based catering services.',
                'example': 'Delivered end-to-end catering services for corporate functions and private events of up to 500 guests.'
            },
            {
                'title': 'Team Coordination',
                'keywords': ['team', 'staff', 'crew'],
                'feedback': 'Demonstrate leadership or coordination in kitchen or event teams.',
                'example': 'Supervised a team of 6 kitchen staff, ensuring smooth service delivery under tight timelines.'
            }
        ],
        "Consultancy": [
            {
                'title': 'Client Advisory',
                'keywords': ['client'],
                'feedback': 'Demonstrate your role in advising clients or delivering tailored recommendations.',
                'example': 'Provided strategic advisory services to clients across banking and retail sectors.'
            },
            {
                'title': 'Problem Solving',
                'keywords': ['problem'],
                'feedback': 'Highlight your ability to analyze complex issues and deliver solutions.',
                'example': 'Led root cause analysis to address client’s operational inefficiencies, improving process turnaround by 25%.'
            },
            {
                'title': 'Business Process Improvement',
                'keywords': ['business improvement', 'transformation', 'process improvement'],
                'feedback': 'Include examples of streamlining business operations or driving transformation.',
                'example': 'Redesigned procurement workflow, reducing vendor turnaround time by 40%.'
            },
            {
                'title': 'Consulting Frameworks',
                'keywords': ['framework'],
                'feedback': 'Mention using structured approaches like SWOT, PESTLE, or McKinsey 7S for analysis.',
                'example': 'Used SWOT and Porter’s Five Forces to assess client’s market positioning.'
            },
            {
                'title': 'Recommendations & Reporting',
                'keywords': ['recommendation', 'proposal', 'report'],
                'feedback': 'Emphasize experience creating reports, insights, or strategic proposals.',
                'example': 'Delivered board-level reports with actionable recommendations on growth strategy.'
            },
            {
                'title': 'Presentation Delivery',
                'keywords': ['presentation', 'deck', 'slides'],
                'feedback': 'Show that you’ve created and delivered professional slide decks or executive presentations.',
                'example': 'Designed and delivered presentations to C-suite stakeholders during project closeouts.'
            },
            {
                'title': 'Change Management',
                'keywords': ['change', 'change management'],
                'feedback': 'Mention experience supporting or managing organizational change.',
                'example': 'Supported change management strategy for post-merger integration, impacting 3 departments.'
            }
        ],
        "Customer Care": [
            {
                'title': 'CRM Tools',
                'keywords': ['crm'],
                'feedback': 'Mention experience with customer relationship management (CRM) tools like Salesforce, Zoho, or HubSpot.',
                'example': 'Managed over 300 customer interactions weekly using Zoho CRM to track inquiries and complaints.'
            },
            {
                'title': 'Inbound Call Handling',
                'keywords': ['inbound'],
                'feedback': 'Showcase experience handling incoming calls or customer requests.',
                'example': 'Answered 50+ inbound calls daily, resolving billing issues and service inquiries.'
            },
            {
                'title': 'Support Ticket Systems',
                'keywords': ['ticketing'],
                'feedback': 'Mention using systems like Zendesk, Freshdesk, or Jira to manage support requests.',
                'example': 'Used Zendesk to track, assign, and close over 200 customer support tickets monthly.'
            },
            {
                'title': 'Customer Retention & Satisfaction',
                'keywords': ['retention', 'resolve', 'satisfaction'],
                'feedback': 'Highlight your contributions to retaining customers and improving satisfaction scores.',
                'example': 'Improved customer retention by 15% through consistent follow-ups and issue resolution.'
            },
            {
                'title': 'Complaint Resolution',
                'keywords': ['complaint'],
                'feedback': 'Add experience handling escalations or resolving customer issues professionally.',
                'example': 'Resolved customer complaints within 24 hours, achieving a 90% satisfaction rate.'
            },
            {
                'title': 'Multichannel Support Experience',
                'keywords': ['multichannel', 'email', 'phone', 'chat', 'social media'],
                'feedback': 'Mention handling customer inquiries via email, phone, live chat, or social media.',
                'example': 'Provided real-time support through live chat, email, and social media channels.'
            }
        ],
        "Data / Business Analysis / AI": [
            {
                'title': 'SQL & Database Queries',
                'keywords': ['sql', 'mysql', 'postgresql'],
                'feedback': 'Mention your ability to write SQL queries or interact with relational databases.',
                'example': 'Extracted insights from large datasets using MySQL and optimized SQL queries for faster processing.'
            },
            {
                'title': 'Power BI Proficiency',
                'keywords': ['power bi'],
                'feedback': 'Show familiarity with Microsoft Power BI for reporting and dashboards.',
                'example': 'Built interactive dashboards in Power BI to track KPIs and business performance metrics.'
            },
            {
                'title': 'Python for Data Analysis',
                'keywords': ['python'],
                'feedback': 'Mention use of Python libraries like pandas, NumPy, or matplotlib for analysis.',
                'example': 'Performed exploratory data analysis with Python using pandas and matplotlib.'
            },
            {
                'title': 'Machine Learning / Predictive Modeling',
                'keywords': ['machine learning', 'ml', 'predictive modeling'],
                'feedback': 'Highlight experience building models or applying machine learning techniques.',
                'example': 'Developed a predictive churn model using scikit-learn, improving retention strategy.'
            },
            {
                'title': 'Excel & Data Manipulation',
                'keywords': ['excel', 'pivot', 'vlookup'],
                'feedback': 'Showcase your proficiency in Excel for cleaning or analyzing data.',
                'example': 'Used pivot tables and VLOOKUP to clean and analyze monthly sales data.'
            },
            {
                'title': 'Insight Generation',
                'keywords': ['insight'],
                'feedback': 'Mention how your analysis led to decisions or business outcomes.',
                'example': 'Generated actionable insights from user behavior data that led to a 20% increase in conversions.'
            },
            {
                'title': 'Data Visualization',
                'keywords': ['visualization', 'dashboard', 'chart'],
                'feedback': 'Include experience creating visuals or dashboards to communicate data findings.',
                'example': 'Built dashboards with Tableau to help stakeholders monitor performance in real time.'
            }
        ],
        "Education / Teaching": [
            {
                'title': 'Curriculum Planning',
                'keywords': ['lesson plan', 'curriculum', 'scheme of work'],
                'feedback': 'Include experience with lesson planning, curriculum design, or schemes of work.',
                'example': 'Designed weekly lesson plans aligned with national curriculum and tailored to different learning styles.'
            },
            {
                'title': 'Student Assessment',
                'keywords': ['assessment', 'test', 'evaluation'],
                'feedback': 'Mention experience with tests, evaluations, or grading.',
                'example': 'Developed and graded tests to monitor student progress and improve learning outcomes.'
            },
            {
                'title': 'Use of ICT in Teaching',
                'keywords': ['ict', 'digital tools', 'google classroom', 'edtech', 'microsoft teams', 'zoom'],
                'feedback': 'Mention use of digital tools or platforms for instruction.',
                'example': 'Used Google Classroom and Zoom to deliver hybrid learning to over 100 secondary students.'
            },
            {
                'title': 'Inclusive or Special Needs Teaching',
                'keywords': ['inclusive education', 'special needs', 'ieps', 'inclusive', 'iep'],
                'feedback': 'Highlight experience with special education or inclusive classrooms.',
                'example': 'Adapted learning strategies to support students with learning disabilities and created Individualized Education Plans (IEPs).'
            },
            {
                'title': 'Classroom Management',
                'keywords': ['classroom management'],
                'feedback': 'Mention how you maintain discipline, order, and an engaging classroom.',
                'example': 'Implemented effective classroom management strategies, reducing disruptions and improving learning engagement.'
            },
            {
                'title': 'Teacher Training or Mentorship',
                'keywords': ['training'],
                'feedback': 'If applicable, mention experience training other teachers or mentoring student teachers.',
                'example': 'Mentored five trainee teachers during their teaching practice and led weekly training workshops on pedagogy.'
            }
        ],
        "Engineering / Technical": [
            {
                'title': 'CAD Software Proficiency',
                'keywords': ['cad', 'autocad', 'solidworks'],
                'feedback': 'Highlight experience with computer-aided design tools such as AutoCAD or SolidWorks.',
                'example': 'Designed mechanical parts and systems using AutoCAD and SolidWorks, optimizing for durability and cost-efficiency.'
            },
            {
                'title': 'Piping and Layout Design',
                'keywords': ['piping'],
                'feedback': 'If applicable, include your role in piping system design or analysis.',
                'example': 'Conducted piping layout and stress analysis for industrial facilities using Caesar II.'
            },
            {
                'title': 'Preventive Maintenance',
                'keywords': ['preventive maintenance'],
                'feedback': 'Mention experience with preventive or corrective maintenance programs.',
                'example': 'Implemented a preventive maintenance schedule that reduced machinery breakdowns by 35%.'
            },
            {
                'title': 'Regulatory & Safety Compliance',
                'keywords': ['compliance', 'regulation', 'hse', 'health and safety'],
                'feedback': 'Include knowledge of health, safety, or environmental (HSE) standards.',
                'example': 'Ensured compliance with HSE regulations during facility upgrades and maintenance.'
            },
            {
                'title': 'System Troubleshooting',
                'keywords': ['troubleshooting'],
                'feedback': 'Demonstrate problem-solving or diagnostic experience with technical systems.',
                'example': 'Diagnosed electrical faults and performed root cause analysis to restore system functionality.'
            },
            {
                'title': 'Installation and Commissioning',
                'keywords': ['equipment installation', 'commissioning', 'installation'],
                'feedback': 'Mention projects involving the setup or commissioning of equipment or systems.',
                'example': 'Led the installation and commissioning of a new industrial HVAC system within budget and timeline.'
            }
        ],
        "Finance / Accounting / Audit": [
            {
                'title': 'Financial Reporting',
                'keywords': ['financial reporting'],
                'feedback': 'Include experience preparing financial statements or monthly/quarterly reports.',
                'example': 'Prepared monthly financial reports and reconciliations in compliance with IFRS standards.'
            },
            {
                'title': 'Accounting Standards',
                'keywords': ['ifrs', 'gaap'],
                'feedback': 'Mention your knowledge or use of IFRS or GAAP standards.',
                'example': 'Ensured compliance with IFRS reporting requirements during quarterly audits.'
            },
            {
                'title': 'Budget Management',
                'keywords': ['budget'],
                'feedback': 'Mention experience preparing, monitoring, or managing budgets.',
                'example': 'Developed and monitored departmental budgets to ensure cost-efficiency and alignment with targets.'
            },
            {
                'title': 'Audit Experience',
                'keywords': ['audit'],
                'feedback': 'Include internal or external audit experience, if relevant.',
                'example': 'Participated in internal audits to identify compliance gaps and recommend corrective actions.'
            },
            {
                'title': 'Tax Compliance',
                'keywords': ['tax', 'vat'],
                'feedback': 'Mention experience with tax computations, filing, or compliance.',
                'example': 'Handled monthly VAT filings and annual tax computations for the organization.'
            },
            {
                'title': 'Reconciliation Skills',
                'keywords': ['reconciliation', 'bank reconciliation'],
                'feedback': 'Highlight experience with account or bank reconciliations.',
                'example': 'Performed weekly bank reconciliations to ensure accuracy between financial records and bank statements.'
            },
            {
                'title': 'Cost Control Initiatives',
                'keywords': ['cost control'],
                'feedback': 'Mention any involvement in reducing costs or improving expense efficiency.',
                'example': 'Implemented cost control measures that reduced departmental spending by 12% over two quarters.'
            }
        ],
        "Hospitality / Hotel / Restaurant": [
            {
                'title': 'Guest Services',
                'keywords': ['guest service'],
                'feedback': 'Highlight your experience attending to guests, resolving complaints, or managing guest relations.',
                'example': 'Handled guest check-ins, resolved complaints, and ensured 5-star service delivery.'
            },
            {
                'title': 'Food Safety Standards',
                'keywords': ['food safety', 'haccp'],
                'feedback': 'Mention knowledge of food safety procedures or certifications like HACCP.',
                'example': 'Maintained HACCP compliance and ensured kitchen hygiene standards were strictly followed.'
            },
            {
                'title': 'Event Management',
                'keywords': ['event'],
                'feedback': 'Mention experience with planning, setting up, or coordinating events.',
                'example': 'Coordinated banquet events and ensured smooth execution of weddings and conferences.'
            },
            {
                'title': 'Front Desk Operations',
                'keywords': ['front desk'],
                'feedback': 'Include tasks such as handling reservations, check-ins, or switchboard operations.',
                'example': 'Managed front desk operations including check-ins, check-outs, and room reservations.'
            },
            {
                'title': 'Inventory or Stock Management',
                'keywords': ['inventory'],
                'feedback': 'Mention roles involving inventory tracking, stocktaking, or supplier coordination.',
                'example': 'Maintained accurate kitchen inventory and coordinated with suppliers for daily stock replenishment.'
            },
            {
                'title': 'Bar Service Experience',
                'keywords': ['bartender', 'bar', 'cocktail'],
                'feedback': 'Include roles related to bartending, drink mixing, or bar stock handling.',
                'example': 'Worked as a bartender, creating custom cocktails and managing bar supplies.'
            },
            {
                'title': 'Housekeeping Duties',
                'keywords': ['housekeeping', 'room cleaning', 'room'],
                'feedback': 'Mention experience in room preparation, cleaning, or laundry.',
                'example': 'Performed housekeeping tasks including room cleaning, linen replacement, and restocking amenities.'
            }
        ],
        "Human Resources / HR": [
            {
                'title': 'Recruitment Experience',
                'keywords': ['recruit', 'sourcing', 'headhunting'],
                'feedback': 'Mention involvement in recruitment, talent sourcing, or selection.',
                'example': 'Led end-to-end recruitment processes including job posting, CV screening, and candidate interviews.'
            },
            {
                'title': 'Employee Onboarding',
                'keywords': ['onboarding', 'orientation'],
                'feedback': 'Highlight experience with onboarding, induction, or orientation programs.',
                'example': 'Developed and executed onboarding programs for new hires, reducing early attrition by 30%.'
            },
            {
                'title': 'HR Policies & Compliance',
                'keywords': ['hr policy', 'policy'],
                'feedback': 'Show that you’ve worked on developing or enforcing HR policies.',
                'example': 'Drafted HR policies on employee conduct and performance appraisal, ensuring compliance with labour laws.'
            },
            {
                'title': 'Payroll Management',
                'keywords': ['payroll'],
                'feedback': 'Mention if you’ve handled payroll processing, salary computation, or related tools.',
                'example': 'Managed payroll for 200+ employees using Sage HR and ensured timely salary disbursements.'
            },
            {
                'title': 'Performance Management',
                'keywords': ['performance'],
                'feedback': 'Show any experience in performance reviews or appraisal systems.',
                'example': 'Coordinated annual performance reviews and implemented KPIs across departments.'
            },
            {
                'title': 'Employee Relations',
                'keywords': ['employee relations'],
                'feedback': 'Highlight conflict resolution, staff engagement, or grievance handling skills.',
                'example': 'Handled staff grievances and conducted employee engagement surveys that improved satisfaction scores.'
            },
            {
                'title': 'Training & Development',
                'keywords': ['training', 'development'],
                'feedback': 'Mention organizing or facilitating employee training or upskilling programs.',
                'example': 'Facilitated monthly training sessions on workplace ethics and customer service excellence.'
            }
        ],
        "ICT / Computer": [
            {
                'title': 'IT Support Experience',
                'keywords': ['it support', 'helpdesk', 'technical support'],
                'cv_keywords': ['helpdesk', 'it support', 'technical support', 'ticketing system'],
                'feedback': 'Highlight experience in helpdesk or end-user technical support.',
                'example': 'Provided tier-1 IT support resolving hardware/software issues and managing ticketing systems like Zendesk.'
            },
            {
                'title': 'Networking Skills',
                'keywords': ['network', 'router', 'firewall'],
                'cv_keywords': ['network', 'router', 'switch', 'firewall', 'lan', 'wan'],
                'feedback': 'Demonstrate experience with routers, switches, LAN/WAN, or firewalls.',
                'example': 'Configured Cisco routers and managed LAN/WAN infrastructure for a 50-user network.'
            },
            {
                'title': 'System Administration',
                'keywords': ['system admin', 'server'],
                'cv_keywords': ['system administrator', 'sysadmin', 'active directory', 'windows server', 'linux'],
                'feedback': 'Mention your experience managing servers or user accounts.',
                'example': 'Managed Windows Server 2019 environments and handled user access through Active Directory.'
            },
            {
                'title': 'Cybersecurity Awareness',
                'keywords': ['cybersecurity', 'security'],
                'cv_keywords': ['cybersecurity', 'security audit', 'vulnerability', 'threat'],
                'feedback': 'Include experience with security protocols or threat detection tools.',
                'example': 'Performed routine system vulnerability checks and enforced endpoint security policies.'
            },
            {
                'title': 'Hardware Maintenance',
                'keywords': ['hardware'],
                'cv_keywords': ['hardware', 'repair', 'installation', 'maintenance'],
                'feedback': 'List your experience repairing or maintaining computers or printers.',
                'example': 'Repaired desktop computers, replaced faulty hardware components, and handled printer maintenance.'
            },
            {
                'title': 'Software Support',
                'keywords': ['software'],
                'cv_keywords': ['software installation', 'software support', 'patching', 'license'],
                'feedback': 'Show knowledge of installing, configuring, or troubleshooting applications.',
                'example': 'Installed licensed software for 100+ users and provided troubleshooting for application errors.'
            },
            {
                'title': 'Cloud Technologies',
                'keywords': ['cloud'],
                'cv_keywords': ['aws', 'azure', 'gcp', 'cloud'],
                'feedback': 'Include cloud platforms you’ve worked with such as AWS or Azure.',
                'example': 'Deployed virtual machines and storage services using Microsoft Azure.'
            },
            {
                'title': 'Scripting and Automation',
                'keywords': ['script', 'automation'],
                'cv_keywords': ['bash', 'powershell', 'automation', 'scripting'],
                'feedback': 'Mention any use of scripting languages to automate IT tasks.',
                'example': 'Automated server backup tasks using PowerShell scripts.'
            }
        ],
        "Programming & Development": [
            {
                'title': 'Python Experience',
                'keywords': ['python'],
                'cv_keywords': ['python'],
                'feedback': 'Python is a core requirement for this role, but your CV doesn’t mention it. Include relevant experience if applicable.',
                'example': 'Developed RESTful APIs and backend services using Python and Django.'
            },
            {
                'title': 'JavaScript Skills',
                'keywords': ['javascript'],
                'cv_keywords': ['javascript'],
                'feedback': 'The job calls for JavaScript expertise. Add your experience with JavaScript, frameworks, or frontend logic.',
                'example': 'Built dynamic user interfaces using JavaScript and modern frameworks like React.'
            },
            {
                'title': 'React Framework',
                'keywords': ['react'],
                'cv_keywords': ['react'],
                'feedback': 'React is mentioned in the job description, but missing from your CV. Include it if you have experience.',
                'example': 'Implemented complex frontend components using React and Redux.'
            },
            {
                'title': 'API Integration / Development',
                'keywords': ['api'],
                'cv_keywords': ['api'],
                'feedback': 'API knowledge is required. Mention experience with building or consuming APIs.',
                'example': 'Integrated third-party APIs and built custom RESTful services.'
            },
            {
                'title': 'Version Control (Git)',
                'keywords': ['git'],
                'cv_keywords': ['git'],
                'feedback': 'Git or version control isn’t mentioned on your CV. Include this if you’ve worked with it.',
                'example': 'Used Git for version control and collaborative development via GitHub.'
            },
            {
                'title': 'Agile Methodologies',
                'keywords': ['agile'],
                'cv_keywords': ['agile'],
                'feedback': 'Agile is a key work style in development. Highlight experience with sprints or Scrum practices.',
                'example': 'Collaborated in Agile teams using Scrum methodology and participated in sprint planning and reviews.'
            },
            {
                'title': 'TypeScript Knowledge',
                'keywords': ['typescript'],
                'cv_keywords': ['typescript'],
                'feedback': 'TypeScript is listed but not reflected on your CV. Mention it if relevant.',
                'example': 'Built scalable frontend components using TypeScript with Angular and React.'
            },
            {
                'title': 'Node.js Backend Skills',
                'keywords': ['node.js'],
                'cv_keywords': ['node', 'node.js'],
                'feedback': 'Node.js appears in the JD but not in your CV. Add relevant backend experience if any.',
                'example': 'Developed REST APIs and real-time services using Node.js and Express.'
            },
            {
                'title': 'Database Technologies',
                'keywords': ['database'],
                'cv_keywords': ['sql', 'mysql', 'postgresql', 'mongodb'],
                'feedback': 'Database skills are needed for this role. Include any SQL or NoSQL experience.',
                'example': 'Designed and optimized MySQL queries; also worked with MongoDB for document-based data.'
            },
            {
                'title': 'Deployment Experience',
                'keywords': ['deployment'],
                'cv_keywords': ['deployment'],
                'feedback': 'Deployment responsibilities are mentioned. Add details if you’ve deployed applications.',
                'example': 'Deployed web applications to AWS EC2 and managed CI/CD pipelines.'
            }
        ],
        "UI/UX & Design": [
            {
                'title': 'Figma Proficiency',
                'keywords': ['figma'],
                'cv_keywords': ['figma'],
                'feedback': 'Figma is a core tool in UI/UX jobs. Add it if you’ve used it for design, prototyping, or collaboration.',
                'example': 'Designed interactive product prototypes and high-fidelity UI screens using Figma.'
            },
            {
                'title': 'User Research Skills',
                'keywords': ['user research'],
                'cv_keywords': ['user research'],
                'feedback': 'This role involves user research, but your CV doesn’t reflect that. Include methods like interviews, surveys, or usability tests.',
                'example': 'Conducted user interviews and usability testing to inform design decisions.'
            },
            {
                'title': 'UX Writing',
                'keywords': ['ux writing'],
                'cv_keywords': ['ux writing'],
                'feedback': 'UX writing is expected but missing on your CV. Include if you’ve worked on microcopy or content strategy.',
                'example': 'Wrote intuitive microcopy for app onboarding and error messages, improving user guidance.'
            },
            {
                'title': 'Wireframing Experience',
                'keywords': ['wireframe'],
                'cv_keywords': ['wireframe'],
                'feedback': 'Wireframing is listed in the job, but your CV doesn’t mention it. Add tools or examples if relevant.',
                'example': 'Created wireframes using Balsamiq and Figma to map early-stage product flows.'
            },
            {
                'title': 'Design Systems',
                'keywords': ['design system'],
                'cv_keywords': ['design system'],
                'feedback': 'Design system experience is requested. Include it if you’ve worked with or built one.',
                'example': 'Maintained and extended the company’s Figma-based design system for scalable product UI.'
            },
            {
                'title': 'Accessibility Standards',
                'keywords': ['accessibility'],
                'cv_keywords': ['accessibility'],
                'feedback': 'Accessibility is mentioned, but not reflected on your CV. Mention WCAG compliance or accessibility testing.',
                'example': 'Designed components compliant with WCAG guidelines to ensure usability for all users.'
            },
            {
                'title': 'Adobe Tools',
                'keywords': ['adobe'],
                'cv_keywords': ['photoshop', 'illustrator', 'adobe xd'],
                'feedback': 'Adobe tools are listed, but your CV doesn’t reflect experience with Photoshop, Illustrator, or XD.',
                'example': 'Used Adobe XD for wireframing and Illustrator for visual design assets.'
            },
            {
                'title': 'Interaction Design',
                'keywords': ['interaction design'],
                'cv_keywords': ['interaction design'],
                'feedback': 'Interaction design is required, but not shown on your CV. Add if you’ve worked on animations, transitions, or flows.',
                'example': 'Designed intuitive user flows and transitions for a mobile e-commerce application.'
            },
            {
                'title': 'Prototyping Skills',
                'keywords': ['prototyping'],
                'cv_keywords': ['prototype'],
                'feedback': 'Prototyping is mentioned in the JD. Add if you’ve created interactive prototypes for testing or stakeholder feedback.',
                'example': 'Built interactive prototypes in Figma to validate features with stakeholders before development.'
            },
            {
                'title': 'UI Design',
                'keywords': ['ui design'],
                'cv_keywords': ['ui design'],
                'feedback': 'UI design is listed in the role but missing on your CV. Be sure to include layout or visual work.',
                'example': 'Designed responsive UI layouts for a fintech dashboard across web and mobile.'
            }
        ],
        "DevOps": [
            {
                'title': 'CI/CD Pipeline Experience',
                'keywords': ['ci/cd'],
                'cv_keywords': ['ci/cd', 'jenkins', 'github actions', 'gitlab ci'],
                'feedback': 'CI/CD is a core requirement, but your CV doesn’t reflect experience in this area. Add relevant tools or pipelines you’ve worked with.',
                'example': 'Implemented CI/CD pipelines using GitHub Actions to automate testing and deployments.'
            },
            {
                'title': 'Docker',
                'keywords': ['docker'],
                'cv_keywords': ['docker'],
                'feedback': 'Docker is mentioned, but not found on your CV. Include it if you’ve containerized applications or managed images.',
                'example': 'Containerized microservices using Docker to standardize environments across dev and prod.'
            },
            {
                'title': 'Kubernetes Experience',
                'keywords': ['kubernetes'],
                'cv_keywords': ['kubernetes'],
                'feedback': 'Kubernetes is a key part of the job, but not reflected on your CV. Mention if you’ve used it for orchestration or scaling.',
                'example': 'Deployed and scaled containerized apps on Kubernetes using Helm and kubectl.'
            },
            {
                'title': 'Cloud Platforms',
                'keywords': ['cloud'],
                'cv_keywords': ['aws', 'azure', 'gcp'],
                'feedback': 'The role requires cloud experience, but your CV doesn’t show any. Mention AWS, Azure, or GCP if applicable.',
                'example': 'Managed cloud infrastructure on AWS using EC2, S3, and CloudFormation.'
            },
            {
                'title': 'Infrastructure as Code (IaC)',
                'keywords': ['infrastructure as code'],
                'cv_keywords': ['terraform', 'cloudformation', 'pulumi'],
                'feedback': 'IaC is mentioned but missing from your CV. Add if you’ve used Terraform, CloudFormation, etc.',
                'example': 'Used Terraform to provision and manage infrastructure as code across multiple environments.'
            },
            {
                'title': 'Monitoring Tools',
                'keywords': ['monitoring'],
                'cv_keywords': ['prometheus', 'grafana', 'datadog', 'cloudwatch'],
                'feedback': 'Monitoring is part of the job but not reflected on your CV. Include tools like Prometheus, Grafana, or CloudWatch.',
                'example': 'Set up Prometheus and Grafana dashboards to monitor app performance and system health.'
            },
            {
                'title': 'Linux Skills',
                'keywords': ['linux'],
                'cv_keywords': ['linux'],
                'feedback': 'Linux administration is a typical requirement in DevOps, but not seen on your CV.',
                'example': 'Managed server configuration and automation on Ubuntu and CentOS environments.'
            },
            {
                'title': 'Configuration Management',
                'keywords': ['ansible'],
                'cv_keywords': ['ansible'],
                'feedback': 'Ansible is listed but not reflected on your CV. Include it if you’ve used it for server setup or provisioning.',
                'example': 'Automated server configuration and deployments using Ansible playbooks.'
            },
            {
                'title': 'Scripting & Automation',
                'keywords': ['scripting'],
                'cv_keywords': ['bash', 'shell', 'python'],
                'feedback': 'Scripting is part of the JD, but your CV doesn’t show relevant skills. Include languages like Bash or Python.',
                'example': 'Wrote Bash scripts to automate log rotation, backups, and service restarts.'
            },
            {
                'title': 'SRE Knowledge',
                'keywords': ['site reliability'],
                'cv_keywords': ['sre', 'site reliability'],
                'feedback': 'Site Reliability Engineering is part of the role, but your CV doesn’t reflect SRE responsibilities or mindset.',
                'example': 'Applied SRE practices to improve service uptime and manage SLIs, SLOs, and SLAs.'
            }
        ],
        "Testing / QA": [
            {
                'title': 'Test Case Development',
                'keywords': ['test case'],
                'cv_keywords': ['test case'],
                'feedback': 'Mention your experience designing or writing test cases.',
                'example': 'Designed detailed manual and automated test cases for new feature rollouts.'
            },
            {
                'title': 'Bug Tracking',
                'keywords': ['bug'],
                'cv_keywords': ['bug'],
                'feedback': 'Include experience identifying, logging, and tracking software bugs.',
                'example': 'Reported and tracked bugs using Jira and collaborated with developers to resolve critical issues.'
            },
            {
                'title': 'Automated Testing',
                'keywords': ['automated testing', 'selenium', 'test automation'],
                'cv_keywords': ['automation', 'automated test', 'selenium'],
                'feedback': 'Highlight experience with automation frameworks if mentioned in the job description.',
                'example': 'Implemented automated test scripts using Selenium WebDriver for regression testing.'
            },
            {
                'title': 'Quality Assurance Process',
                'keywords': ['qa process'],
                'cv_keywords': ['qa'],
                'feedback': 'Explain your contribution to ensuring product quality through QA processes.',
                'example': 'Participated in end-to-end QA processes ensuring compliance with software quality standards.'
            },
            {
                'title': 'QA Tools',
                'keywords': ['tools'],
                'cv_keywords': ['postman', 'jira', 'testrail', 'selenium', 'cypress'],
                'feedback': 'Mention QA tools you have used that align with the job description.',
                'example': 'Used Postman for API testing and TestRail for test case management.'
            }
        ],
        "Product Management": [
            {
                'title': 'Product Roadmap',
                'keywords': ['roadmap'],
                'cv_keywords': ['roadmap'],
                'feedback': 'Mention your experience creating or managing a product roadmap.',
                'example': 'Developed quarterly product roadmaps aligned with customer needs and company goals.'
            },
            {
                'title': 'Stakeholder Engagement',
                'keywords': ['stakeholders'],
                'cv_keywords': ['stakeholders'],
                'feedback': 'Show how you worked with stakeholders to shape product direction.',
                'example': 'Collaborated with engineering, sales, and marketing stakeholders to prioritize product features.'
            },
            {
                'title': 'Market Research',
                'keywords': ['market research'],
                'cv_keywords': ['market'],
                'feedback': 'Include any user research or market validation efforts you’ve led.',
                'example': 'Conducted competitor and market research to guide MVP feature prioritization.'
            },
            {
                'title': 'Product KPIs',
                'keywords': ['kpi'],
                'cv_keywords': ['kpi', 'metrics', 'product success'],
                'feedback': 'Discuss how you measured product success using KPIs or data.',
                'example': 'Defined and tracked KPIs such as user retention and conversion rate post-launch.'
            },
            {
                'title': 'Cross-functional Collaboration',
                'keywords': ['cross-functional'],
                'cv_keywords': ['cross-functional'],
                'feedback': 'Mention any experience working across teams (design, dev, marketing).',
                'example': 'Led cross-functional teams to deliver product features on time and within scope.'
            },
            {
                'title': 'Agile Environment',
                'keywords': ['agile'],
                'cv_keywords': ['agile'],
                'feedback': 'If you’ve worked in agile teams, include that experience.',
                'example': 'Managed backlog and sprint planning as part of an agile product development team.'
            },
            {
                'title': 'User Stories',
                'keywords': ['user stories'],
                'cv_keywords': ['user stories'],
                'feedback': 'Include your experience writing or refining user stories.',
                'example': 'Created detailed user stories and acceptance criteria to align development with business needs.'
            },
            {
                'title': 'Product Manager Role',
                'keywords': ['product manager', 'product owner'],
                'cv_keywords': ['product manager'],
                'feedback': 'If the job is explicitly for a Product Manager, make sure your title or experience reflects this clearly.',
                'example': 'Worked as a Product Manager leading end-to-end product development from ideation to launch.'
            }
        ],
        "Project Management": [
            {
                'title': 'Project Lifecycle Understanding',
                'keywords': ['project lifecycle'],
                'cv_keywords': ['project lifecycle'],
                'feedback': 'Highlight your experience managing full project lifecycles, from initiation to delivery.',
                'example': 'Oversaw end-to-end delivery of software projects from planning to deployment across agile teams.'
            },
            {
                'title': 'Scrum Methodology',
                'keywords': ['scrum'],
                'cv_keywords': ['scrum'],
                'feedback': 'Include your familiarity or certification with Scrum methodologies if relevant.',
                'example': 'Facilitated daily standups and sprint reviews as a certified Scrum Master for cross-functional teams.'
            },
            {
                'title': 'Stakeholder Communication',
                'keywords': ['stakeholder'],
                'cv_keywords': ['stakeholder'],
                'feedback': 'Demonstrate your ability to engage or report to stakeholders across the project lifecycle.',
                'example': 'Liaised with internal and external stakeholders to align project scope and deliverables.'
            },
            {
                'title': 'Budget or Resource Management',
                'keywords': ['budget'],
                'cv_keywords': ['budget'],
                'feedback': 'If applicable, mention budget oversight or efficient resource management on tech projects.',
                'example': 'Managed project budgets of up to $250,000 and reallocated resources to meet tight timelines.'
            },
            {
                'title': 'Project Management Tools (Jira)',
                'keywords': ['jira'],
                'cv_keywords': ['jira'],
                'feedback': 'Mention Jira or other PM tools if used for task tracking and sprint management.',
                'example': 'Utilized Jira for backlog grooming, sprint planning, and monitoring team velocity.'
            },
            {
                'title': 'Timeline Management',
                'keywords': ['timeline'],
                'cv_keywords': ['timeline'],
                'feedback': 'Show your ability to deliver projects on time or manage shifting deadlines effectively.',
                'example': 'Delivered complex web projects 2 weeks ahead of schedule through proactive sprint planning.'
            }
        ],
        "Insurance": [
            {
                'title': 'Underwriting Knowledge',
                'keywords': ['underwriting'],
                'cv_keywords': ['underwriting'],
                'feedback': 'Include your experience or familiarity with risk assessment or underwriting processes.',
                'example': 'Performed risk evaluation and underwriting for SME business clients using data-driven models.'
            },
            {
                'title': 'Claims Processing',
                'keywords': ['claims'],
                'cv_keywords': ['claims'],
                'feedback': 'Mention experience with claims review, assessment, or resolution.',
                'example': 'Processed over 200 auto insurance claims, ensuring quick resolution and minimal client churn.'
            },
            {
                'title': 'Policy Administration',
                'keywords': ['policy administration'],
                'cv_keywords': ['policy administration'],
                'feedback': 'Highlight tasks involving policy issuance, renewals, endorsements, or cancellations.',
                'example': 'Managed life insurance policy administration including endorsements and renewals for 500+ clients.'
            },
            {
                'title': 'Actuarial or Statistical Analysis',
                'keywords': ['actuarial'],
                'cv_keywords': ['actuarial'],
                'feedback': 'If applicable, mention actuarial tasks like risk modeling or premium calculation.',
                'example': 'Supported actuarial team in pricing strategies using mortality and claims data trends.'
            },
            {
                'title': 'Insurance-Specific Software',
                'keywords': ['insurance software'],
                'cv_keywords': ['insurance software', 'guidewire', 'epic'],
                'feedback': 'Include tools like Guidewire, Applied Epic, or proprietary claims/policy software.',
                'example': 'Utilized Guidewire PolicyCenter for quote generation and policy management.'
            }
        ],
        "Law / Legal": [
            {
                'title': 'Legal Research',
                'keywords': ['legal research'],
                'cv_keywords': ['legal research'],
                'feedback': 'Highlight your experience with researching statutes, case law, or legal precedents.',
                'example': 'Conducted legal research to support litigation on commercial dispute cases, ensuring accurate case citations.'
            },
            {
                'title': 'Legal Drafting Skills',
                'keywords': ['drafting'],
                'cv_keywords': ['drafting'],
                'feedback': 'Include your ability to draft contracts, pleadings, affidavits, or legal opinions.',
                'example': 'Drafted commercial contracts and NDAs, reducing client risk exposure in cross-border transactions.'
            },
            {
                'title': 'Litigation Experience',
                'keywords': ['litigation'],
                'cv_keywords': ['litigation'],
                'feedback': 'Mention your exposure to civil/criminal litigation, court procedures, or trial preparation.',
                'example': 'Supported litigation team in case preparation, filings, and court appearances at magistrate and high courts.'
            },
            {
                'title': 'Regulatory Compliance',
                'keywords': ['compliance'],
                'cv_keywords': ['compliance'],
                'feedback': 'Show familiarity with compliance frameworks, regulatory audits, or policy reviews.',
                'example': 'Monitored legal compliance with AML and data protection laws across company departments.'
            },
            {
                'title': 'Contract Review & Negotiation',
                'keywords': ['contract review'],
                'cv_keywords': ['contract review'],
                'feedback': 'Add details of contract review, risk flagging, and negotiation support if applicable.',
                'example': 'Reviewed supplier contracts to flag risk clauses and negotiated terms favourable to company interests.'
            },
            {
                'title': 'Corporate Law Knowledge',
                'keywords': ['corporate law'],
                'cv_keywords': ['corporate law'],
                'feedback': 'Include experience advising on business formation, governance, or shareholder issues.',
                'example': 'Advised startups on company incorporation, board structure, and regulatory filings.'
            },
            {
                'title': 'Due Diligence Support',
                'keywords': ['due diligence'],
                'cv_keywords': ['due diligence'],
                'feedback': 'Mention M&A or compliance due diligence, especially for transactional or commercial law roles.',
                'example': 'Conducted legal due diligence for acquisition targets, reviewing corporate filings and liabilities.'
            },
            {
                'title': 'Work in Legal Practice or Chambers',
                'keywords': ['law firm'],
                'cv_keywords': ['law firm'],
                'feedback': 'Mention prior experience in a law firm, legal clinic, or court internship.',
                'example': 'Interned at XYZ Chambers, assisting with legal drafting, file prep, and court submissions.'
            }
        ],
        "Logistics": [
            {
                'title': 'Supply Chain Knowledge',
                'keywords': ['supply chain'],
                'cv_keywords': ['supply chain'],
                'feedback': 'Mention your experience coordinating or optimizing supply chain activities, if applicable.',
                'example': 'Coordinated end-to-end supply chain operations from procurement to last-mile delivery.'
            },
            {
                'title': 'Inventory Management',
                'keywords': ['inventory'],
                'cv_keywords': ['inventory'],
                'feedback': 'Include experience with inventory control, stock audits, or warehouse systems.',
                'example': 'Implemented an automated inventory tracking system, reducing stock variance by 20%.'
            },
            {
                'title': 'Fleet Management',
                'keywords': ['fleet'],
                'cv_keywords': ['fleet'],
                'feedback': 'Highlight any experience managing transportation fleet, maintenance, or routing.',
                'example': 'Oversaw a delivery fleet of 30 vehicles, optimizing route schedules to reduce fuel costs.'
            },
            {
                'title': 'Logistics Software Proficiency',
                'keywords': ['logistics software'],
                'cv_keywords': ['logistics software'],
                'feedback': 'Mention relevant logistics or ERP software (e.g., SAP, Oracle, Odoo, TMS) you’ve used.',
                'example': 'Used SAP SCM to track inventory movement and generate logistics performance reports.'
            },
            {
                'title': 'Warehouse Operations',
                'keywords': ['warehouse'],
                'cv_keywords': ['warehouse'],
                'feedback': 'Include warehouse-related duties like loading, storage, picking/packing, or layout planning.',
                'example': 'Managed warehouse layout optimization, improving picking speed and storage efficiency.'
            },
            {
                'title': 'Delivery Coordination',
                'keywords': ['delivery'],
                'cv_keywords': ['delivery'],
                'feedback': 'Showcase your experience planning, tracking, or improving delivery operations.',
                'example': 'Coordinated daily deliveries across 12 states, achieving 96% on-time rate.'
            },
            {
                'title': 'Import Logistics Experience',
                'keywords': ['import'],
                'cv_keywords': ['import'],
                'feedback': 'Highlight your knowledge of customs, documentation, or freight handling.',
                'example': 'Handled import documentation and clearing processes for high-value consignments.'
            },
            {
                'title': 'Export Coordination',
                'keywords': ['export'],
                'cv_keywords': ['export'],
                'feedback': 'Include export compliance, shipment tracking, or vendor coordination experience.',
                'example': 'Coordinated export shipments, ensuring all compliance documentation met customs requirements.'
            },
            {
                'title': 'Route Optimization',
                'keywords': ['route optimization'],
                'cv_keywords': ['route optimization'],
                'feedback': 'Mention your ability to plan efficient routes for cost-saving and delivery speed.',
                'example': 'Used GPS and route planning tools to optimize delivery schedules and reduce turnaround time.'
            },
            {
                'title': 'Logistics Coordination',
                'keywords': ['logistics coordination'],
                'cv_keywords': ['logistics coordination'],
                'feedback': 'Describe your coordination efforts across departments, vendors, or field teams.',
                'example': 'Liaised with suppliers, drivers, and warehouse teams to ensure timely order fulfillment.'
            }
        ],
        "Manufacturing": [
            {
                'title': 'Production Planning',
                'keywords': ['production planning'],
                'cv_keywords': ['production planning'],
                'feedback': 'Highlight your role in scheduling production, managing timelines, or meeting output targets.',
                'example': 'Planned and executed weekly production schedules to meet 98% of customer orders on time.'
            },
            {
                'title': 'Quality Control',
                'keywords': ['quality control'],
                'cv_keywords': ['quality control'],
                'feedback': 'Include responsibilities around inspecting, testing, or enforcing product standards.',
                'example': 'Performed in-process quality checks to ensure compliance with ISO 9001 standards.'
            },
            {
                'title': 'Machine Operation Skills',
                'keywords': ['machine operation'],
                'cv_keywords': ['machine operation'],
                'feedback': 'Mention machines or equipment you’ve operated, maintained, or calibrated.',
                'example': 'Operated CNC machines to produce precision parts for automotive components.'
            },
            {
                'title': 'Lean Manufacturing Knowledge',
                'keywords': ['lean manufacturing'],
                'cv_keywords': ['lean'],
                'feedback': 'Include familiarity with lean methods such as 5S, Kaizen, or Six Sigma.',
                'example': 'Led Kaizen events that reduced production waste by 18%.'
            },
            {
                'title': 'Health and Safety Compliance',
                'keywords': ['health and safety'],
                'cv_keywords': ['safety'],
                'feedback': 'Show your experience ensuring safe working environments or adhering to HSE standards.',
                'example': 'Trained factory workers on safety procedures, resulting in a 50% drop in incidents.'
            },
            {
                'title': 'Assembly Line Experience',
                'keywords': ['assembly line'],
                'cv_keywords': ['assembly'],
                'feedback': 'State your involvement in assembling, inspecting, or improving line processes.',
                'example': 'Worked on an automated assembly line, ensuring efficient part alignment and minimal defects.'
            },
            {
                'title': 'Preventive Maintenance',
                'keywords': ['preventive maintenance'],
                'cv_keywords': ['maintenance'],
                'feedback': 'Include routine checks, equipment servicing, or downtime reduction efforts.',
                'example': 'Implemented preventive maintenance schedules that reduced machine breakdowns by 30%.'
            },
            {
                'title': 'ERP or Manufacturing Software Proficiency',
                'keywords': ['manufacturing software'],
                'cv_keywords': ['erp', 'sap'],
                'feedback': 'Mention platforms like SAP, Oracle Manufacturing, or MES if used.',
                'example': 'Used SAP MRP to track production materials and schedule jobs efficiently.'
            },
            {
                'title': 'Technical Drawing Interpretation',
                'keywords': ['technical drawings'],
                'cv_keywords': ['technical drawing', 'blueprint'],
                'feedback': 'Highlight your ability to read blueprints, schematics, or CAD diagrams.',
                'example': 'Interpreted mechanical blueprints to fabricate components to exact specifications.'
            },
            {
                'title': 'Packaging & Finishing',
                'keywords': ['packaging'],
                'cv_keywords': ['packaging'],
                'feedback': 'Include tasks involving final product packaging, labeling, or dispatch.',
                'example': 'Led a packaging line team that increased throughput by 22%.'
            }
        ],
        "Media / Advertising / Branding": [
            {
                'title': 'Brand Strategy',
                'keywords': ['brand strategy'],
                'cv_keywords': ['brand strategy'],
                'feedback': 'Highlight any involvement in crafting, executing, or overseeing brand strategy.',
                'example': 'Led the development of a refreshed brand strategy that increased audience engagement by 35%.'
            },
            {
                'title': 'Content Creation',
                'keywords': ['content creation'],
                'cv_keywords': ['content creation'],
                'feedback': 'Include your skills or portfolio in creating compelling written, visual, or multimedia content.',
                'example': 'Produced weekly video content for YouTube and Instagram, growing followers by 20k in 6 months.'
            },
            {
                'title': 'Advertising Campaigns',
                'keywords': ['campaign'],
                'cv_keywords': ['campaign'],
                'feedback': 'Emphasize your experience planning, executing, or measuring advertising campaigns.',
                'example': 'Managed digital ad campaigns across Meta and Google Ads, achieving a 4.8x return on ad spend.'
            },
            {
                'title': 'Social Media Marketing',
                'keywords': ['social media'],
                'cv_keywords': ['social media'],
                'feedback': 'Mention your proficiency in managing or growing social platforms, especially with tools or strategy.',
                'example': 'Built and managed a brand’s social media presence across 5 platforms using Hootsuite and native tools.'
            },
            {
                'title': 'Copywriting',
                'keywords': ['copywriting'],
                'cv_keywords': ['copywriting'],
                'feedback': 'Show your experience writing compelling ad copy, headlines, or promotional material.',
                'example': 'Crafted persuasive ad copy and landing pages that increased email signups by 42%.'
            },
            {
                'title': 'Media Buying / Planning',
                'keywords': ['media buying'],
                'cv_keywords': ['media buying'],
                'feedback': 'Include experience with negotiating, purchasing, or planning media across channels.',
                'example': 'Planned and executed TV and radio media buys worth ₦50M+, optimizing for maximum reach and cost-efficiency.'
            },
            {
                'title': 'Creative Direction',
                'keywords': ['creative direction'],
                'cv_keywords': ['creative direction'],
                'feedback': 'Highlight leadership in conceptualizing or overseeing visual campaigns or storytelling.',
                'example': 'Directed a team of designers and videographers to deliver a national rebranding campaign.'
            },
            {
                'title': 'Ad Copywriting',
                'keywords': ['ad copy'],
                'cv_keywords': ['ad copy'],
                'feedback': 'Specify your contributions to writing copy that resonates with target audiences and meets goals.',
                'example': 'Wrote conversion-driven ad copy for ecommerce clients with CTRs exceeding industry benchmarks.'
            }
        ],
        "Medical / Healthcare": [
            {
                'title': 'Clinical Experience',
                'keywords': ['clinical'],
                'cv_keywords': ['clinical'],
                'feedback': 'Emphasize your direct patient care or clinical rotation experience relevant to the role.',
                'example': 'Completed 12 months of clinical rotations in internal medicine, pediatrics, and emergency care.'
            },
            {
                'title': 'Diagnostic Skills',
                'keywords': ['diagnosis'],
                'cv_keywords': ['diagnosis'],
                'feedback': 'Mention your ability to assess symptoms, interpret tests, or contribute to medical diagnosis.',
                'example': 'Skilled in diagnosing common respiratory and gastrointestinal conditions through clinical assessments.'
            },
            {
                'title': 'Patient Care Competence',
                'keywords': ['patient care'],
                'cv_keywords': ['patient care'],
                'feedback': 'Include experience offering compassionate and effective care to diverse patient populations.',
                'example': 'Provided holistic patient care in outpatient and inpatient settings, ensuring high recovery rates.'
            },
            {
                'title': 'Treatment Administration',
                'keywords': ['treatment'],
                'cv_keywords': ['treatment'],
                'feedback': 'Highlight your ability to recommend, prescribe, or support treatment procedures.',
                'example': 'Administered IV medications and assisted in minor surgical procedures under physician supervision.'
            },
            {
                'title': 'Electronic Medical Records (EMR)',
                'keywords': ['emr', 'electronic medical records'],
                'cv_keywords': ['emr', 'electronic medical records'],
                'feedback': 'Mention your proficiency with EMR systems or digital patient data entry.',
                'example': 'Maintained accurate patient data using EMR platforms like OpenMRS and Medisoft.'
            },
            {
                'title': 'Licensure or Certification',
                'keywords': ['medical license'],
                'cv_keywords': ['medical license'],
                'feedback': 'List any medical license, registration, or certifications required for practice.',
                'example': 'Licensed Medical Doctor (MDCN) with valid registration and annual practice license.'
            },
            {
                'title': 'Infection Control Practices',
                'keywords': ['infection control'],
                'cv_keywords': ['infection control'],
                'feedback': 'Demonstrate your adherence to safety protocols and hygiene in clinical settings.',
                'example': 'Implemented WHO-standard infection control measures, reducing ward infection rates by 20%.'
            },
            {
                'title': 'Patient Counseling',
                'keywords': ['counseling'],
                'cv_keywords': ['counseling'],
                'feedback': 'Include examples of how you educate or counsel patients on treatments, medication, or lifestyle.',
                'example': 'Conducted pre- and post-operative counseling for patients undergoing elective surgeries.'
            }
        ],
        "NGO / Non-Profit": [
            {
                'title': 'Grant Writing Experience',
                'keywords': ['grant writing'],
                'cv_keywords': ['grant writing'],
                'feedback': 'Highlight your experience writing or contributing to successful grant proposals.',
                'example': 'Co-authored grant proposals that secured over $150,000 in donor funding from USAID and DFID.'
            },
            {
                'title': 'Donor Relations',
                'keywords': ['donor'],
                'cv_keywords': ['donor'],
                'feedback': 'Show your experience managing donor expectations or reporting progress to funders.',
                'example': 'Managed quarterly donor reports and maintained communication with institutional funders.'
            },
            {
                'title': 'Community Engagement',
                'keywords': ['community engagement'],
                'cv_keywords': ['community engagement'],
                'feedback': 'Demonstrate your ability to work with local communities, stakeholders, or target beneficiaries.',
                'example': 'Led grassroots outreach programs that impacted over 3,000 rural women in Northern Nigeria.'
            },
            {
                'title': 'Monitoring & Evaluation (M&E)',
                'keywords': ['monitoring and evaluation', 'm&e'],
                'cv_keywords': ['monitoring and evaluation', 'm&e'],
                'feedback': 'Include your M&E experience, especially designing frameworks or analyzing impact.',
                'example': 'Developed M&E tools and analyzed program KPIs to assess effectiveness of nutrition intervention.'
            },
            {
                'title': 'Proposal Development',
                'keywords': ['proposal writing'],
                'cv_keywords': ['proposal writing'],
                'feedback': 'Showcase your role in developing or contributing to project proposals.',
                'example': 'Drafted project concept notes and full proposals for UNDP-funded youth empowerment initiatives.'
            },
            {
                'title': 'Advocacy or Policy Engagement',
                'keywords': ['advocacy'],
                'cv_keywords': ['advocacy'],
                'feedback': 'Mention your involvement in advocacy campaigns or policy lobbying.',
                'example': 'Coordinated advocacy campaigns on sexual health rights, reaching over 5,000 adolescents.'
            },
            {
                'title': 'Stakeholder or Partnership Building',
                'keywords': ['partnerships'],
                'cv_keywords': ['partnerships'],
                'feedback': 'Include your work with local/international partners or coalitions.',
                'example': 'Forged multi-sector partnerships with government and NGOs for water sanitation projects.'
            },
            {
                'title': 'Program or Donor Reporting',
                'keywords': ['reporting'],
                'cv_keywords': ['reporting'],
                'feedback': 'Highlight your experience preparing technical or narrative reports.',
                'example': 'Compiled monthly project progress reports aligned with donor M&E requirements.'
            }
        ],
        "Oil and Gas / Energy": [
            {
                'title': 'Health, Safety & Environment (HSE)',
                'keywords': ['hse', 'health safety'],
                'cv_keywords': ['hse', 'health safety'],
                'feedback': 'Mention your understanding or certification in HSE procedures and compliance.',
                'example': 'Implemented HSE protocols that reduced onsite incidents by 30% over a 12-month period.'
            },
            {
                'title': 'Upstream Operations Experience',
                'keywords': ['upstream'],
                'cv_keywords': ['upstream'],
                'feedback': 'Include relevant upstream exploration or drilling activities if applicable.',
                'example': 'Worked on upstream drilling operations across multiple onshore and offshore assets.'
            },
            {
                'title': 'Downstream Operations',
                'keywords': ['downstream'],
                'cv_keywords': ['downstream'],
                'feedback': 'Highlight any refining, marketing, or distribution experience in downstream segments.',
                'example': 'Supervised product distribution and depot operations for refined petroleum products.'
            },
            {
                'title': 'Rig Operations',
                'keywords': ['rig'],
                'cv_keywords': ['rig'],
                'feedback': 'Mention experience working on or with drilling rigs—onshore or offshore.',
                'example': 'Assisted in rig commissioning and monitored drilling parameters during exploratory well operations.'
            },
            {
                'title': 'Pipeline Engineering / Monitoring',
                'keywords': ['pipeline'],
                'cv_keywords': ['pipeline'],
                'feedback': 'Highlight roles related to pipeline inspection, construction, or maintenance.',
                'example': 'Coordinated pipeline integrity testing and ensured compliance with environmental standards.'
            },
            {
                'title': 'Regulatory Compliance',
                'keywords': ['compliance'],
                'cv_keywords': ['compliance'],
                'feedback': 'Show your ability to adhere to industry regulations and environmental standards.',
                'example': 'Ensured NUPRC and DPR regulatory compliance in daily drilling operations.'
            },
            {
                'title': 'Reservoir Management',
                'keywords': ['reservoir'],
                'cv_keywords': ['reservoir'],
                'feedback': 'If relevant, indicate experience with reservoir modeling, monitoring, or production optimization.',
                'example': 'Worked with geologists and reservoir engineers to optimize well performance in mature fields.'
            },
            {
                'title': 'Renewable Energy Exposure',
                'keywords': ['renewable'],
                'cv_keywords': ['renewable'],
                'feedback': 'If applicable, highlight exposure to solar, wind, or hybrid energy projects.',
                'example': 'Led feasibility studies for off-grid solar installations in rural electrification projects.'
            }
        ],
        "Procurement / Store-keeping / Supply Chain": [
            {
                'title': 'Vendor Management',
                'keywords': ['vendor management'],
                'cv_keywords': ['vendor management'],
                'feedback': 'Highlight your experience in sourcing, negotiating, or evaluating vendors.',
                'example': 'Managed relationships with over 20 international and local suppliers, ensuring compliance with procurement policies.'
            },
            {
                'title': 'Inventory Management',
                'keywords': ['inventory'],
                'cv_keywords': ['inventory', 'stock'],
                'feedback': 'Include experience with stock control, reorder levels, or warehouse systems.',
                'example': 'Implemented inventory tracking system that reduced stockouts by 30% and minimized holding costs.'
            },
            {
                'title': 'Supply Chain Operations',
                'keywords': ['supply chain'],
                'cv_keywords': ['supply chain'],
                'feedback': 'Demonstrate knowledge of end-to-end supply chain, including logistics and procurement.',
                'example': 'Coordinated cross-border supply chain operations, reducing lead times by 20%.'
            },
            {
                'title': 'ERP Software Proficiency',
                'keywords': ['erp'],
                'cv_keywords': ['erp'],
                'feedback': 'Mention any ERP systems used for procurement or inventory control (e.g., SAP, Oracle).',
                'example': 'Used SAP MM module for purchase requisitions, order tracking, and vendor invoice management.'
            },
            {
                'title': 'Cost Saving Initiatives',
                'keywords': ['cost saving'],
                'cv_keywords': ['cost saving'],
                'feedback': 'Show impact on procurement costs or efficiency improvements.',
                'example': 'Negotiated bulk purchasing deals, saving the company ₦15M in annual procurement costs.'
            },
            {
                'title': 'RFQ/RFP Process Handling',
                'keywords': ['rfq', 'request for quotation'],
                'cv_keywords': ['rfq', 'request for quotation'],
                'feedback': 'Include experience preparing or managing Requests for Quotation or Proposals.',
                'example': 'Drafted and evaluated RFQs for construction supplies, ensuring compliance with procurement standards.'
            },
            {
                'title': 'Warehouse Management',
                'keywords': ['warehouse'],
                'cv_keywords': ['warehouse'],
                'feedback': 'Highlight warehouse operations, layout optimization, or safety protocols.',
                'example': 'Led warehouse reorganization project that improved space utilization and reduced item retrieval time by 40%.'
            },
            {
                'title': 'Logistics Coordination',
                'keywords': ['logistics'],
                'cv_keywords': ['logistics'],
                'feedback': 'Mention coordination of shipping, delivery, or distribution logistics.',
                'example': 'Supervised inbound and outbound logistics across 6 Nigerian states, ensuring timely product delivery.'
            }
        ],
        "Real Estate": [
            {
                'title': 'Property Management Experience',
                'keywords': ['property management'],
                'cv_keywords': ['property management'],
                'feedback': 'Mention your involvement in managing residential or commercial properties.',
                'example': 'Managed a portfolio of 30+ rental properties, ensuring tenant satisfaction and timely maintenance.'
            },
            {
                'title': 'Real Estate Valuation',
                'keywords': ['valuation'],
                'cv_keywords': ['valuation'],
                'feedback': 'Highlight your skills or certifications related to valuing properties.',
                'example': 'Conducted real estate appraisals and valuations for both residential and commercial properties.'
            },
            {
                'title': 'Leasing & Tenancy',
                'keywords': ['leasing'],
                'cv_keywords': ['leasing'],
                'feedback': 'Include your role in tenant acquisition, lease negotiation, or agreement management.',
                'example': 'Handled lease negotiations and tenant onboarding for a mixed-use commercial building.'
            },
            {
                'title': 'Site Inspection Experience',
                'keywords': ['site inspection'],
                'cv_keywords': ['site inspection'],
                'feedback': 'Mention any experience in conducting property/site inspections and reporting.',
                'example': 'Performed routine site inspections to ensure compliance with building standards and maintenance contracts.'
            },
            {
                'title': 'Title Documentation & Verification',
                'keywords': ['title documentation'],
                'cv_keywords': ['title documentation'],
                'feedback': 'Highlight your involvement in verifying or handling title documents for real estate transactions.',
                'example': 'Reviewed title documents and coordinated with legal teams to validate property ownership and transfer.'
            },
            {
                'title': 'Realtor Certification or Experience',
                'keywords': ['realtor'],
                'cv_keywords': ['realtor'],
                'feedback': 'State any realtor license or experience in property brokerage or sales.',
                'example': 'Certified Realtor with experience closing residential deals worth over ₦500M.'
            }
        ],
        "Safety and Environment / HSE": [
            {
                'title': 'HSE Compliance',
                'keywords': ['hse compliance'],
                'cv_keywords': ['hse compliance'],
                'feedback': 'Highlight your experience ensuring Health, Safety, and Environmental (HSE) compliance in workplace operations.',
                'example': 'Ensured strict compliance with HSE policies, reducing incident rates by 40% across site operations.'
            },
            {
                'title': 'Risk Assessment Expertise',
                'keywords': ['risk assessment'],
                'cv_keywords': ['risk assessment'],
                'feedback': 'Demonstrate your ability to identify and mitigate safety risks through structured assessments.',
                'example': 'Conducted regular risk assessments and implemented mitigation plans in line with ISO 45001 standards.'
            },
            {
                'title': 'Incident Investigation',
                'keywords': ['incident investigation'],
                'cv_keywords': ['incident investigation'],
                'feedback': 'Include your experience in investigating and reporting workplace incidents or near misses.',
                'example': 'Led root cause analysis and reporting for on-site incidents, improving future safety protocols.'
            },
            {
                'title': 'Safety Training Implementation',
                'keywords': ['safety training'],
                'cv_keywords': ['safety training'],
                'feedback': 'Mention your role in organizing or delivering safety training programs.',
                'example': 'Facilitated monthly HSE awareness sessions for over 100 staff, boosting compliance rates.'
            },
            {
                'title': 'Personal Protective Equipment (PPE)',
                'keywords': ['ppe'],
                'cv_keywords': ['ppe'],
                'feedback': 'Reflect awareness or enforcement of PPE usage in safety-sensitive environments.',
                'example': 'Implemented strict PPE compliance guidelines and monitored daily adherence across worksites.'
            },
            {
                'title': 'Environmental Management',
                'keywords': ['environmental management'],
                'cv_keywords': ['environmental management'],
                'feedback': 'Highlight your involvement in minimizing environmental impact or managing waste/sustainability practices.',
                'example': 'Developed site-specific environmental management plans, leading to ISO 14001 certification.'
            },
            {
                'title': 'ISO 14001 Compliance',
                'keywords': ['iso 14001'],
                'cv_keywords': ['iso 14001'],
                'feedback': 'Mention any knowledge or application of ISO 14001 Environmental Management Systems.',
                'example': 'Assisted in preparing documentation and procedures for successful ISO 14001 audit clearance.'
            },
            {
                'title': 'HSE Auditing',
                'keywords': ['hse audits'],
                'cv_keywords': ['hse audits'],
                'feedback': 'Showcase your role in internal or third-party HSE audit processes.',
                'example': 'Conducted quarterly HSE audits across multiple departments, ensuring continuous safety compliance.'
            }
        ],
        "Sales / Marketing / Retail / Business Development": [
            {
                'title': 'Sales Target Achievement',
                'keywords': ['sales target'],
                'cv_keywords': ['sales target'],
                'feedback': 'Demonstrate your ability to meet or exceed sales targets or quotas.',
                'example': 'Achieved 120% of quarterly sales target through strategic client acquisition and upselling.'
            },
            {
                'title': 'Business Development',
                'keywords': ['business development'],
                'cv_keywords': ['business development'],
                'feedback': 'Highlight experience in identifying growth opportunities or expanding market reach.',
                'example': 'Spearheaded business development initiatives that increased client base by 35% in 12 months.'
            },
            {
                'title': 'Client Acquisition',
                'keywords': ['client acquisition'],
                'cv_keywords': ['client acquisition'],
                'feedback': 'Show your success in acquiring new customers or entering new markets.',
                'example': 'Secured 50+ new clients through lead generation, cold outreach, and networking strategies.'
            },
            {
                'title': 'CRM Tools (e.g., Salesforce, HubSpot)',
                'keywords': ['crm'],
                'cv_keywords': ['crm'],
                'feedback': 'Mention your experience using CRM systems to manage sales pipelines or client engagement.',
                'example': 'Used HubSpot to track sales activities and nurture leads, reducing deal closure time by 20%.'
            },
            {
                'title': 'Retail Sales Management',
                'keywords': ['retail sales'],
                'cv_keywords': ['retail sales'],
                'feedback': 'If relevant, include experience with point-of-sale systems, floor supervision, or retail customer service.',
                'example': 'Managed daily retail operations and improved upsell rates by training staff on customer engagement.'
            },
            {
                'title': 'Sales Funnel Optimization',
                'keywords': ['sales funnel'],
                'cv_keywords': ['sales funnel'],
                'feedback': 'Include knowledge or use of strategies to move prospects through the sales funnel.',
                'example': 'Designed lead nurturing campaigns that improved conversion rates across all funnel stages.'
            },
            {
                'title': 'Negotiation & Deal Closing',
                'keywords': ['negotiation'],
                'cv_keywords': ['negotiation'],
                'feedback': 'Emphasize your skill in negotiating terms and closing high-value deals.',
                'example': 'Negotiated and closed a $500,000 annual supply deal with a multinational client.'
            },
            {
                'title': 'Market Research & Analysis',
                'keywords': ['market research'],
                'cv_keywords': ['market research'],
                'feedback': 'Highlight how you’ve used market insights to guide product positioning or outreach.',
                'example': 'Conducted competitive market analysis to reposition brand messaging, resulting in a 25% traffic uplift.'
            },
            {
                'title': 'Digital Marketing Strategy',
                'keywords': ['digital marketing'],
                'cv_keywords': ['digital marketing'],
                'feedback': 'Mention experience running or contributing to online campaigns via SEO, ads, email, or social media.',
                'example': 'Launched integrated digital campaigns across Google Ads and Meta, generating 3X ROAS.'
            },
            {
                'title': 'Brand Awareness Building',
                'keywords': ['brand awareness'],
                'cv_keywords': ['brand awareness'],
                'feedback': 'Show how you’ve contributed to increasing visibility or perception of a brand or product.',
                'example': 'Executed PR and influencer marketing campaigns that increased brand visibility by 60%.'
            }
        ],
        "Science": [
            {
                'title': 'Laboratory Experience',
                'keywords': ['laboratory'],
                'cv_keywords': ['laboratory'],
                'feedback': 'Include any hands-on experience in laboratory settings, procedures, or safety protocols.',
                'example': 'Conducted chemical experiments in compliance with ISO lab standards and maintained detailed logs.'
            },
            {
                'title': 'Scientific Data Analysis',
                'keywords': ['data analysis'],
                'cv_keywords': ['data analysis'],
                'feedback': 'Mention your skills in analyzing experimental or field data using scientific methods or tools.',
                'example': 'Used SPSS and R to analyze biodiversity data, leading to publishable findings on species distribution.'
            },
            {
                'title': 'Scientific Research',
                'keywords': ['scientific research'],
                'cv_keywords': ['scientific research'],
                'feedback': 'Demonstrate involvement in formal research studies, publications, or investigations.',
                'example': 'Led a two-year study on antimicrobial resistance with results published in a peer-reviewed journal.'
            },
            {
                'title': 'Scientific Report Writing',
                'keywords': ['report writing'],
                'cv_keywords': ['report writing'],
                'feedback': 'Show ability to draft research papers, lab reports, or technical documents for scientific audiences.',
                'example': 'Authored detailed reports on water quality assessments submitted to regulatory bodies.'
            },
            {
                'title': 'Microscopy & Imaging',
                'keywords': ['microscopy'],
                'cv_keywords': ['microscopy'],
                'feedback': 'Highlight experience with microscopes (optical, electron, etc.) or image analysis tools.',
                'example': 'Used SEM and fluorescence microscopy to study cell structures and capture high-resolution images.'
            },
            {
                'title': 'Scientific Fieldwork',
                'keywords': ['fieldwork'],
                'cv_keywords': ['fieldwork'],
                'feedback': 'Include experiences collecting samples or conducting experiments in real-world environments.',
                'example': 'Carried out geological surveys and sediment sampling across multiple coastal sites.'
            },
            {
                'title': 'Spectrometry Techniques',
                'keywords': ['spectrometry'],
                'cv_keywords': ['spectrometry'],
                'feedback': 'Mention your use of spectroscopy (e.g., UV-Vis, IR, Mass Spec) in research or analysis.',
                'example': 'Performed GC-MS analysis to determine pollutant levels in soil and water samples.'
            },
            {
                'title': 'Grant Writing & Funding',
                'keywords': ['research grant'],
                'cv_keywords': ['grant'],
                'feedback': 'Highlight experience with securing or contributing to scientific grants and research funding.',
                'example': 'Secured a $50,000 research grant from TETFund to study disease resistance in crops.'
            },
            {
                'title': 'Scientific Publications',
                'keywords': ['publication'],
                'cv_keywords': ['publication'],
                'feedback': 'List any peer-reviewed articles, journals, or research papers you’ve contributed to.',
                'example': 'Co-authored 3 papers in international journals on climate variability in Sub-Saharan Africa.'
            },
            {
                'title': 'Scientific Tools & Software',
                'keywords': ['scientific software'],
                'cv_keywords': ['matlab', 'spss', 'r ', 'stata'],
                'feedback': 'Mention software relevant to your scientific domain such as MATLAB, R, SPSS, or Stata.',
                'example': 'Analyzed molecular dynamics using MATLAB and visualized results using custom scripts.'
            }
        ],
        "Security / Intelligence": [
            {
                'title': 'Threat Analysis',
                'keywords': ['threat analysis'],
                'cv_keywords': ['threat analysis'],
                'feedback': 'Showcase your ability to assess security threats and develop countermeasures.',
                'example': 'Performed threat analysis on sensitive company operations, reducing potential breaches by 60%.'
            },
            {
                'title': 'Surveillance Operations',
                'keywords': ['surveillance'],
                'cv_keywords': ['surveillance'],
                'feedback': 'Include any experience monitoring environments through CCTV or physical patrols.',
                'example': 'Monitored restricted access zones using CCTV systems and performed daily security audits.'
            },
            {
                'title': 'Access Control',
                'keywords': ['access control'],
                'cv_keywords': ['access control'],
                'feedback': 'Mention your handling of personnel or system access — physical or digital.',
                'example': 'Managed access control using biometric systems for over 200 employees.'
            },
            {
                'title': 'Incident Response',
                'keywords': ['incident response'],
                'cv_keywords': ['incident response'],
                'feedback': 'Highlight how you respond to or report security incidents, breaches, or suspicious activity.',
                'example': 'Led rapid response to security breach resulting in zero data loss and complete system recovery within 2 hours.'
            },
            {
                'title': 'Intelligence Gathering',
                'keywords': ['intel gathering', 'intelligence gathering'],
                'cv_keywords': ['intelligence'],
                'feedback': 'Detail any activities involving information gathering, analysis, or reporting on threats.',
                'example': 'Compiled and analyzed regional threat intelligence for use in proactive security planning.'
            },
            {
                'title': 'Counter-Terrorism Strategies',
                'keywords': ['counter-terrorism'],
                'cv_keywords': ['counter-terrorism'],
                'feedback': 'Include experience working with or developing measures against terrorism or insurgent threats.',
                'example': 'Worked with local agencies to implement counter-terrorism training protocols for staff.'
            },
            {
                'title': 'Emergency Response Readiness',
                'keywords': ['emergency response'],
                'cv_keywords': ['emergency response'],
                'feedback': 'Mention drills, trainings, or real incidents where you managed emergency protocols.',
                'example': 'Coordinated fire evacuation drills and emergency response during a facility lockdown.'
            },
            {
                'title': 'Security Audits & Assessments',
                'keywords': ['security audit'],
                'cv_keywords': ['security audit'],
                'feedback': 'Indicate any role in evaluating and strengthening security posture.',
                'example': 'Conducted quarterly security audits and implemented changes that improved compliance scores by 40%.'
            },
            {
                'title': 'Law Enforcement Background',
                'keywords': ['law enforcement'],
                'cv_keywords': ['police', 'security officer', 'law enforcement'],
                'feedback': 'Mention any law enforcement training, partnerships, or experience.',
                'example': 'Worked closely with local police in apprehending trespassers and maintaining perimeter control.'
            },
            {
                'title': 'Confidentiality & Information Handling',
                'keywords': ['confidentiality'],
                'cv_keywords': ['confidentiality'],
                'feedback': 'Reinforce your commitment to handling sensitive or classified data securely.',
                'example': 'Handled sensitive intelligence files under strict access and confidentiality protocols.'
            }
        ],
        "Travels & Tours": [
            {
                'title': 'Itinerary Planning',
                'keywords': ['itinerary planning'],
                'cv_keywords': ['itinerary'],
                'feedback': 'Mention your experience creating travel plans or schedules for clients or groups.',
                'example': 'Designed detailed travel itineraries covering flights, transfers, lodging, and excursions across 5 countries.'
            },
            {
                'title': 'Visa Assistance & Documentation',
                'keywords': ['visa processing'],
                'cv_keywords': ['visa'],
                'feedback': 'Highlight any experience with visa application, embassy liaison, or travel documentation.',
                'example': 'Processed over 300 visa applications for clients, ensuring 98% success rate.'
            },
            {
                'title': 'Tour Guide Experience',
                'keywords': ['tour guiding'],
                'cv_keywords': ['tour guide'],
                'feedback': 'Include any roles where you guided groups, explained attractions, or coordinated tours.',
                'example': 'Led daily walking tours for up to 25 tourists across historical sites with 4.9-star average feedback.'
            },
            {
                'title': 'Flight Booking & Ticketing',
                'keywords': ['flight booking'],
                'cv_keywords': ['flight booking', 'ticketing'],
                'feedback': 'Mention your proficiency with booking tools or airline systems.',
                'example': 'Booked domestic and international flights using Amadeus GDS system for over 200 clients monthly.'
            },
            {
                'title': 'Travel Agency Operations',
                'keywords': ['travel agency'],
                'cv_keywords': ['travel agency'],
                'feedback': 'Indicate experience managing or working within a travel agency setup.',
                'example': 'Coordinated agency logistics, partnered with airlines, and handled travel bookings end-to-end.'
            },
            {
                'title': 'Client Experience & Satisfaction',
                'keywords': ['customer satisfaction'],
                'cv_keywords': ['customer satisfaction'],
                'feedback': 'Include achievements or metrics showing how you enhanced client travel experiences.',
                'example': 'Achieved 95% repeat customer rate by providing personalized travel solutions and real-time support.'
            },
            {
                'title': 'Hotel Reservations',
                'keywords': ['hotel booking'],
                'cv_keywords': ['hotel'],
                'feedback': 'Highlight experience booking accommodation, negotiating rates, or handling cancellations.',
                'example': 'Managed hotel bookings across 3 continents, negotiating group discounts and resolving reservation issues.'
            },
            {
                'title': 'Tour Package Design',
                'keywords': ['tour packages'],
                'cv_keywords': ['tour packages'],
                'feedback': 'Mention any creative or logistical input you had in creating travel packages.',
                'example': 'Developed honeymoon and adventure tour packages that boosted sales by 30% in one year.'
            },
            {
                'title': 'International Travel Coordination',
                'keywords': ['international travel'],
                'cv_keywords': ['international'],
                'feedback': 'Show familiarity with global travel routes, regulations, and client needs.',
                'example': 'Coordinated international group travel for conferences, ensuring smooth cross-border logistics.'
            },
            {
                'title': 'Travel Insurance Advisory',
                'keywords': ['travel insurance'],
                'cv_keywords': ['insurance'],
                'feedback': 'Mention helping clients understand and choose travel insurance plans.',
                'example': 'Guided clients through travel insurance options, reducing claims processing delays.'
            }
        ]
    }    

    rules = FIELD_RULES.get(field, [])
    for rule in rules:
        keyword_hit = any(k in jd_lower for k in rule['keywords']) and not any(k in cv_lower for k in rule['keywords'])
        semantic_hit = any(is_semantic_match(jd_text, cv_text, k) for k in rule['keywords'])
    
        if keyword_hit or semantic_hit:
            suggestions.append({
                'title': rule['title'],
                'feedback': rule['feedback'],
                'example': rule.get('example', '')
            })

    return suggestions        

# === ROUTES ===
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['cv']
        jd_text = request.form['jd']
        field = request.form['field']

        if not file or not allowed_file(file.filename):
            return 'Invalid file format. Upload a .pdf or .docx file.'

        # Extract CV text
        cv_text = extract_text(file)

        # Score CV vs JD using improved scoring
        score = calculate_score(cv_text, jd_text)

        # Get smart suggestions
        suggestions = generate_suggestions(cv_text, jd_text, field)

        # If nothing found, fallback to "general improvement" tips
        if not suggestions:
            suggestions.append({
                'title': 'No major gaps found — but here’s a tip',
                'feedback': 'Consider adding more measurable results, tools used, or leadership achievements to strengthen this CV.',
                'example': 'Improved project turnaround time by 25%.'
            })

        # Display results
        html_result = f"""
            <h2>Semantic Match Score: {score}%</h2>
            <h3>Smart Suggestions</h3>
            <ul>
        """
        for s in suggestions:
            html_result += f"<li><strong>{s['title']}</strong>: {s['feedback']}<br><em>e.g., {s['example']}</em></li>"
        html_result += "</ul>"
        return html_result

    # Default GET view
    return render_template_string('''
        <h1>Smart CV Matcher</h1>
        <form method="POST" enctype="multipart/form-data">
            <label>Upload CV (.pdf or .docx):</label><br>
            <input type="file" name="cv" required><br><br>

            <label>Paste Job Description:</label><br>
            <textarea name="jd" rows="10" cols="60" required></textarea><br><br>

            <label>Select Field:</label><br>
            <select name="field" required>
                <option value="Administration / Secretarial">Administration / Secretarial</option>
                <option value="Agriculture / Agro-Allied">Agriculture / Agro-Allied</option>
                <option value="Aviation / Airline">Aviation / Airline</option>
                <option value="Banking">Banking</option>
                <option value="Catering / Confectionery">Catering / Confectionery</option>
                <option value="Consultancy">Consultancy</option>
                <option value="Customer Care">Customer Care</option>
                <option value="Data / Business Analysis / AI">Data / Business Analysis / AI</option>
                <option value="Education / Teaching">Education / Teaching</option>
                <option value="Engineering / Technical">Engineering / Technical</option>
                <option value="Finance / Accounting / Audit">Finance / Accounting / Audit</option>
                <option value="Hospitality / Hotel / Restaurant">Hospitality / Hotel / Restaurant</option>
                <option value="Human Resources / HR">Human Resources / HR</option>
                <option value="ICT / Computer">ICT / Computer</option>
                <option value="Programming & Development">Programming & Development</option>
                <option value="UI/UX & Design">UI/UX & Design</option>
                <option value="DevOps">DevOps</option>
                <option value="Testing / QA">Testing / QA</option>
                <option value="Product Management">Product Management</option>
                <option value="Project Management">Project Management</option>
                <option value="Insurance">Insurance</option>
                <option value="Law / Legal">Law / Legal</option>
                <option value="Logistics">Logistics</option>
                <option value="Manufacturing">Manufacturing</option>
                <option value="Media / Advertising / Branding">Media / Advertising / Branding</option>
                <option value="Medical / Healthcare">Medical / Healthcare</option>
                <option value="NGO / Non-Profit">NGO / Non-Profit</option>
                <option value="Oil and Gas / Energy">Oil and Gas / Energy</option>
                <option value="Procurement / Store-keeping / Supply Chain">Procurement / Store-keeping / Supply Chain</option>
                <option value="Real Estate">Real Estate</option>
                <option value="Safety and Environment / HSE">Safety and Environment / HSE</option>
                <option value="Sales / Marketing / Retail / Business Development">Sales / Marketing / Retail / Business Development</option>
                <option value="Science">Science</option>
                <option value="Security / Intelligence">Security / Intelligence</option>
                <option value="Travels & Tours">Travels & Tours</option>
            </select><br><br>

            <input type="submit" value="Check CV">
        </form>
    ''')

 
# === MAIN ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
