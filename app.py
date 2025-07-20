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
def generate_suggestions(cv_text, jd_text, field):
    cv_lower = cv_text.lower()
    jd_lower = jd_text.lower()
    suggestions = []

    if field == "Administration / Secretarial":
        if missing_keyword('calendar', jd_lower, cv_lower):
            suggestions.append({
                'title': 'Calendar Management',
                'feedback': 'Mention experience with scheduling or managing executive calendars.',
                'example': 'Managed complex calendars, scheduled meetings, and coordinated appointments for senior executives.'
            })

        if missing_any_keyword(['word', 'excel', 'powerpoint'], jd_lower, cv_lower):
            suggestions.append({
                'title': 'Microsoft Office Tools',
                'feedback': 'Mention your proficiency with MS Word, Excel, or PowerPoint.',
                'example': 'Created reports and managed budgets using Microsoft Excel and PowerPoint.'
            })

        if missing_keyword('communication', jd_lower, cv_lower):
            suggestions.append({
                'title': 'Communication Skills',
                'feedback': 'Demonstrate written and verbal communication skills relevant to office correspondence.',
                'example': 'Drafted executive correspondence and handled official communication for departmental activities.'
            })

        if missing_keyword('travel', jd_lower, cv_lower):
            suggestions.append({
                'title': 'Travel Arrangements',
                'feedback': 'If applicable, mention booking or coordinating travel for teams or executives.',
                'example': 'Coordinated local and international travel arrangements for the regional director.'
            })

    if field == "Agriculture / Agro-Allied":
        if missing_keyword('tractor', jd_lower, cv_lower):
            suggestions.append({
                'title': 'Agricultural Equipment Experience',
                'feedback': 'Mention any experience using tractors, planters, sprayers, or irrigation systems.',
                'example': 'Operated and maintained tractors for crop planting and spraying.'
            })

        if missing_any_keyword(['livestock', 'poultry', 'fish', 'aquaculture'], jd_lower, cv_lower):
            suggestions.append({
                'title': 'Livestock or Aquaculture Experience',
                'feedback': 'Highlight any hands-on work with poultry, fish farming, or livestock.',
                'example': 'Managed daily operations in a 500-bird poultry farm including feeding, vaccination, and sales.'
            })

        if missing_keyword('yield', jd_lower, cv_lower):
            suggestions.append({
                'title': 'Yield Improvement Achievements',
                'feedback': 'Showcase any efforts that led to increased farm productivity.',
                'example': 'Improved cassava yields by 25% through adoption of improved varieties and soil testing.'
            })

        if missing_any_keyword(['agritech', 'gis', 'drone'], jd_lower, cv_lower):
            suggestions.append({
                'title': 'AgriTech Tools',
                'feedback': 'Mention if you have worked with drones, GIS software, or digital farm mapping tools.',
                'example': 'Used drone imagery and GIS tools to monitor field health and optimize fertilizer use.'
            })

        if missing_keyword('extension', jd_lower, cv_lower):
            suggestions.append({
                'title': 'Agricultural Extension Work',
                'feedback': 'Mention any experience in community training or farmer outreach programs.',
                'example': 'Conducted farmer training sessions under a USAID extension program for maize farmers.'
            })

        if missing_keyword('value chain', jd_lower, cv_lower):
            suggestions.append({
                'title': 'Agricultural Value Chain Exposure',
                'feedback': 'Highlight experience across any segment: production, processing, marketing, or export.',
                'example': 'Oversaw cassava processing and linked producers to off-takers through cooperative channels.'
            })

    if field == "Aviation / Airline":
        if 'safety' in jd_lower and 'safety' not in cv_lower:
            suggestions.append({
                'title': 'Aviation Safety Compliance',
                'feedback': 'Mention any safety protocol training or adherence to aviation regulations.',
                'example': 'Ensured compliance with FAA/ICAO safety standards during all ground operations.'
            })

        if 'flight attendant' in jd_lower and not any(x in cv_lower for x in ['cabin crew', 'flight attendant']):
            suggestions.append({
                'title': 'Cabin Crew Experience',
                'feedback': 'If you’ve worked as cabin crew or in-flight services, highlight it clearly.',
                'example': 'Served as lead cabin crew on over 500 domestic and international flights, ensuring passenger comfort and safety.'
            })

        if 'ticketing' in jd_lower and 'ticketing' not in cv_lower:
            suggestions.append({
                'title': 'Ticketing and Reservations',
                'feedback': 'Mention your experience with airline reservation systems (e.g., Amadeus, Sabre).',
                'example': 'Handled reservations and ticketing using Amadeus GDS for over 150 daily bookings.'
            })

        if 'ground operations' in jd_lower and 'ground' not in cv_lower:
            suggestions.append({
                'title': 'Ground Operations Support',
                'feedback': 'Highlight tasks related to baggage, boarding, or ramp services.',
                'example': 'Coordinated baggage handling, boarding processes, and marshaling at Lagos International Airport.'
            })

        if 'air traffic' in jd_lower and 'air traffic' not in cv_lower:
            suggestions.append({
                'title': 'Air Traffic Knowledge',
                'feedback': 'Even if you’re not a controller, referencing basic familiarity with ATC procedures can help.',
                'example': 'Worked closely with air traffic controllers to streamline pilot communications during runway congestion.'
            })

    if field == "Banking":
        if any(x in jd_lower for x in ['credit analysis', 'loan', 'risk profile']) and not any(x in cv_lower for x in ['credit analysis', 'credit risk', 'loan']):
            suggestions.append({
                'title': 'Credit Analysis / Risk Assessment',
                'feedback': 'Highlight your experience evaluating loan applications, risk profiles, or conducting creditworthiness assessments.',
                'example': 'Conducted credit analysis for SME loan applications, including cash flow projections and risk scoring.'
            })
    
        if any(x in jd_lower for x in ['compliance', 'regulatory', 'cbn']) and not any(x in cv_lower for x in ['compliance', 'regulatory', 'cbn']):
            suggestions.append({
                'title': 'Regulatory Compliance',
                'feedback': 'Mention your knowledge of CBN regulations, anti-money laundering (AML), or KYC policies.',
                'example': 'Ensured full compliance with CBN regulatory guidelines, including AML/KYC documentation for all clients.'
            })
    
        if 'customer service' in jd_lower and 'customer service' not in cv_lower:
            suggestions.append({
                'title': 'Customer Service in Banking',
                'feedback': 'Emphasize any client-facing roles or experience resolving customer issues in a banking context.',
                'example': 'Handled daily customer interactions, resolving account issues and improving satisfaction scores by 15%.'
            })
    
        if any(x in jd_lower for x in ['investment', 'portfolio', 'asset']) and not any(x in cv_lower for x in ['investment', 'portfolio', 'asset']):
            suggestions.append({
                'title': 'Investment or Portfolio Management',
                'feedback': 'Include your experience managing client portfolios, financial advisory, or wealth planning.',
                'example': 'Managed a portfolio of high-net-worth clients, delivering investment returns above benchmark by 6%.'
            })
    
        if any(x in jd_lower for x in ['financial statement', 'balance sheet', 'p&l']) and not any(x in cv_lower for x in ['financial statement', 'balance sheet', 'p&l']):
            suggestions.append({
                'title': 'Financial Statement Analysis',
                'feedback': 'Mention skills in analyzing balance sheets, income statements, or financial ratios.',
                'example': 'Reviewed company balance sheets and cash flow reports as part of corporate loan assessments.'
            })
    
        if 'sales target' in jd_lower and 'sales target' not in cv_lower:
            suggestions.append({
                'title': 'Sales Target Achievement',
                'feedback': 'If relevant, add metrics showing your contribution to product sales, deposits, or cross-selling in banking.',
                'example': 'Achieved 120% of monthly deposit mobilization targets and cross-sold 40+ insurance packages.'
            })

    if field == "Catering / Confectionery":
        if 'menu planning' in jd_lower and 'menu' not in cv_lower:
            suggestions.append({
                'title': 'Menu Planning',
                'feedback': 'Mention experience in creating or planning menus for events or daily operations.',
                'example': 'Planned customized menus for weddings, corporate events, and daily meal services.'
            })

        if 'food safety' in jd_lower and not any(x in cv_lower for x in ['hygiene', 'sanitation', 'food safety']):
            suggestions.append({
                'title': 'Food Safety and Hygiene',
                'feedback': 'Include any experience or certification related to food hygiene or safety standards.',
                'example': 'Maintained strict hygiene protocols in line with NAFDAC and HACCP standards.'
            })

        if 'baking' in jd_lower and 'baking' not in cv_lower:
            suggestions.append({
                'title': 'Baking Experience',
                'feedback': 'Highlight specific baking skills or types of products (e.g., cakes, pastries, bread).',
                'example': 'Baked and decorated over 300 cakes and pastries monthly for a high-volume bakery.'
            })

        if 'inventory' in jd_lower and 'inventory' not in cv_lower:
            suggestions.append({
                'title': 'Inventory Management',
                'feedback': 'Mention tracking, ordering, or stock management of kitchen or baking supplies.',
                'example': 'Managed stock levels and ensured timely ordering of baking ingredients and materials.'
            })

        if 'catering service' in jd_lower and 'catering' not in cv_lower:
            suggestions.append({
                'title': 'Catering Event Experience',
                'feedback': 'Include any experience managing or working in event-based catering services.',
                'example': 'Delivered end-to-end catering services for corporate functions and private events of up to 500 guests.'
            })

        if 'team' in jd_lower and not any(x in cv_lower for x in ['team', 'staff', 'crew']):
            suggestions.append({
                'title': 'Team Coordination',
                'feedback': 'Demonstrate leadership or coordination in kitchen or event teams.',
                'example': 'Supervised a team of 6 kitchen staff, ensuring smooth service delivery under tight timelines.'
            })

    if field == "Consultancy":
        if 'client' in jd_lower and 'client' not in cv_lower:
            suggestions.append({
                'title': 'Client Advisory',
                'feedback': 'Demonstrate your role in advising clients or delivering tailored recommendations.',
                'example': 'Provided strategic advisory services to clients across banking and retail sectors.'
            })

        if 'problem' in jd_lower and 'problem-solving' not in cv_lower:
            suggestions.append({
                'title': 'Problem Solving',
                'feedback': 'Highlight your ability to analyze complex issues and deliver solutions.',
                'example': 'Led root cause analysis to address client’s operational inefficiencies, improving process turnaround by 25%.'
            })

        if 'business improvement' in jd_lower and not any(x in cv_lower for x in ['transformation', 'process improvement', 'business improvement']):
            suggestions.append({
                'title': 'Business Process Improvement',
                'feedback': 'Include examples of streamlining business operations or driving transformation.',
                'example': 'Redesigned procurement workflow, reducing vendor turnaround time by 40%.'
            })

        if 'framework' in jd_lower and 'framework' not in cv_lower:
            suggestions.append({
                'title': 'Consulting Frameworks',
                'feedback': 'Mention using structured approaches like SWOT, PESTLE, or McKinsey 7S for analysis.',
                'example': 'Used SWOT and Porter’s Five Forces to assess client’s market positioning.'
            })

        if 'recommendation' in jd_lower and not any(x in cv_lower for x in ['recommendation', 'proposal', 'report']):
            suggestions.append({
                'title': 'Recommendations & Reporting',
                'feedback': 'Emphasize experience creating reports, insights, or strategic proposals.',
                'example': 'Delivered board-level reports with actionable recommendations on growth strategy.'
            })

        if 'presentation' in jd_lower and not any(x in cv_lower for x in ['presentation', 'deck', 'slides']):
            suggestions.append({
                'title': 'Presentation Delivery',
                'feedback': 'Show that you’ve created and delivered professional slide decks or executive presentations.',
                'example': 'Designed and delivered presentations to C-suite stakeholders during project closeouts.'
            })

        if 'change' in jd_lower and 'change management' not in cv_lower:
            suggestions.append({
                'title': 'Change Management',
                'feedback': 'Mention experience supporting or managing organizational change.',
                'example': 'Supported change management strategy for post-merger integration, impacting 3 departments.'
            })

    if field == "Customer Care":
        if 'crm' in jd_lower and 'crm' not in cv_lower:
            suggestions.append({
                'title': 'CRM Tools',
                'feedback': 'Mention experience with customer relationship management (CRM) tools like Salesforce, Zoho, or HubSpot.',
                'example': 'Managed over 300 customer interactions weekly using Zoho CRM to track inquiries and complaints.'
            })

        if 'inbound' in jd_lower and 'inbound' not in cv_lower:
            suggestions.append({
                'title': 'Inbound Call Handling',
                'feedback': 'Showcase experience handling incoming calls or customer requests.',
                'example': 'Answered 50+ inbound calls daily, resolving billing issues and service inquiries.'
            })

        if 'ticketing' in jd_lower and 'ticketing' not in cv_lower:
            suggestions.append({
                'title': 'Support Ticket Systems',
                'feedback': 'Mention using systems like Zendesk, Freshdesk, or Jira to manage support requests.',
                'example': 'Used Zendesk to track, assign, and close over 200 customer support tickets monthly.'
            })

        if any(x in jd_lower for x in ['retention', 'resolve', 'satisfaction']) and not any(x in cv_lower for x in ['retention', 'resolve', 'satisfaction']):
            suggestions.append({
                'title': 'Customer Retention & Satisfaction',
                'feedback': 'Highlight your contributions to retaining customers and improving satisfaction scores.',
                'example': 'Improved customer retention by 15% through consistent follow-ups and issue resolution.'
            })

        if 'complaint' in jd_lower and 'complaint' not in cv_lower:
            suggestions.append({
                'title': 'Complaint Resolution',
                'feedback': 'Add experience handling escalations or resolving customer issues professionally.',
                'example': 'Resolved customer complaints within 24 hours, achieving a 90% satisfaction rate.'
            })

        if 'multichannel' in jd_lower and not any(x in cv_lower for x in ['email', 'phone', 'chat', 'social media']):
            suggestions.append({
                'title': 'Multichannel Support Experience',
                'feedback': 'Mention handling customer inquiries via email, phone, live chat, or social media.',
                'example': 'Provided real-time support through live chat, email, and social media channels.'
            })

    if field == "Data / Business Analysis / AI":
        if any(x in jd_lower for x in ['sql', 'mysql', 'postgresql']) and 'sql' not in cv_lower:
            suggestions.append({
                'title': 'SQL & Database Queries',
                'feedback': 'Mention your ability to write SQL queries or interact with relational databases.',
                'example': 'Extracted insights from large datasets using MySQL and optimized SQL queries for faster processing.'
            })

        if 'power bi' in jd_lower and 'power bi' not in cv_lower:
            suggestions.append({
                'title': 'Power BI Proficiency',
                'feedback': 'Show familiarity with Microsoft Power BI for reporting and dashboards.',
                'example': 'Built interactive dashboards in Power BI to track KPIs and business performance metrics.'
            })

        if 'python' in jd_lower and 'python' not in cv_lower:
            suggestions.append({
                'title': 'Python for Data Analysis',
                'feedback': 'Mention use of Python libraries like pandas, NumPy, or matplotlib for analysis.',
                'example': 'Performed exploratory data analysis with Python using pandas and matplotlib.'
            })

        if any(x in jd_lower for x in ['machine learning', 'ml', 'predictive modeling']) and not any(x in cv_lower for x in ['machine learning', 'ml', 'predictive']):
            suggestions.append({
                'title': 'Machine Learning / Predictive Modeling',
                'feedback': 'Highlight experience building models or applying machine learning techniques.',
                'example': 'Developed a predictive churn model using scikit-learn, improving retention strategy.'
            })

        if any(x in jd_lower for x in ['excel', 'pivot', 'vlookup']) and not any(x in cv_lower for x in ['excel', 'pivot', 'vlookup']):
            suggestions.append({
                'title': 'Excel & Data Manipulation',
                'feedback': 'Showcase your proficiency in Excel for cleaning or analyzing data.',
                'example': 'Used pivot tables and VLOOKUP to clean and analyze monthly sales data.'
            })

        if 'insight' in jd_lower and 'insight' not in cv_lower:
            suggestions.append({
                'title': 'Insight Generation',
                'feedback': 'Mention how your analysis led to decisions or business outcomes.',
                'example': 'Generated actionable insights from user behavior data that led to a 20% increase in conversions.'
            })

        if 'visualization' in jd_lower and not any(x in cv_lower for x in ['visualization', 'dashboard', 'chart']):
            suggestions.append({
                'title': 'Data Visualization',
                'feedback': 'Include experience creating visuals or dashboards to communicate data findings.',
                'example': 'Built dashboards with Tableau to help stakeholders monitor performance in real time.'
            })

    if field == "Education / Teaching":
        if any(x in jd_lower for x in ['lesson plan', 'curriculum', 'scheme of work']) and not any(x in cv_lower for x in ['lesson plan', 'curriculum', 'scheme of work']):
            suggestions.append({
                'title': 'Curriculum Planning',
                'feedback': 'Include experience with lesson planning, curriculum design, or schemes of work.',
                'example': 'Designed weekly lesson plans aligned with national curriculum and tailored to different learning styles.'
            })

        if 'assessment' in jd_lower and not any(x in cv_lower for x in ['assessment', 'test', 'evaluation']):
            suggestions.append({
                'title': 'Student Assessment',
                'feedback': 'Mention experience with tests, evaluations, or grading.',
                'example': 'Developed and graded tests to monitor student progress and improve learning outcomes.'
            })

        if any(x in jd_lower for x in ['ict', 'digital tools', 'google classroom', 'edtech']) and not any(x in cv_lower for x in ['ict', 'google classroom', 'edtech', 'microsoft teams', 'zoom']):
            suggestions.append({
                'title': 'Use of ICT in Teaching',
                'feedback': 'Mention use of digital tools or platforms for instruction.',
                'example': 'Used Google Classroom and Zoom to deliver hybrid learning to over 100 secondary students.'
            })

        if any(x in jd_lower for x in ['inclusive education', 'special needs', 'ieps']) and not any(x in cv_lower for x in ['inclusive', 'special needs', 'iep']):
            suggestions.append({
                'title': 'Inclusive or Special Needs Teaching',
                'feedback': 'Highlight experience with special education or inclusive classrooms.',
                'example': 'Adapted learning strategies to support students with learning disabilities and created Individualized Education Plans (IEPs).'
            })

        if 'classroom management' in jd_lower and 'classroom management' not in cv_lower:
            suggestions.append({
                'title': 'Classroom Management',
                'feedback': 'Mention how you maintain discipline, order, and an engaging classroom.',
                'example': 'Implemented effective classroom management strategies, reducing disruptions and improving learning engagement.'
            })

        if 'training' in jd_lower and 'training' not in cv_lower:
            suggestions.append({
                'title': 'Teacher Training or Mentorship',
                'feedback': 'If applicable, mention experience training other teachers or mentoring student teachers.',
                'example': 'Mentored five trainee teachers during their teaching practice and led weekly training workshops on pedagogy.'
            })

    if field == "Engineering / Technical":
        if any(x in jd_lower for x in ['cad', 'autocad', 'solidworks']) and not any(x in cv_lower for x in ['cad', 'autocad', 'solidworks']):
            suggestions.append({
                'title': 'CAD Software Proficiency',
                'feedback': 'Highlight experience with computer-aided design tools such as AutoCAD or SolidWorks.',
                'example': 'Designed mechanical parts and systems using AutoCAD and SolidWorks, optimizing for durability and cost-efficiency.'
            })

        if 'piping' in jd_lower and 'piping' not in cv_lower:
            suggestions.append({
                'title': 'Piping and Layout Design',
                'feedback': 'If applicable, include your role in piping system design or analysis.',
                'example': 'Conducted piping layout and stress analysis for industrial facilities using Caesar II.'
            })

        if 'preventive maintenance' in jd_lower and 'preventive maintenance' not in cv_lower:
            suggestions.append({
                'title': 'Preventive Maintenance',
                'feedback': 'Mention experience with preventive or corrective maintenance programs.',
                'example': 'Implemented a preventive maintenance schedule that reduced machinery breakdowns by 35%.'
            })

        if any(x in jd_lower for x in ['compliance', 'regulation', 'hse', 'health and safety']) and not any(x in cv_lower for x in ['compliance', 'regulation', 'hse', 'health and safety']):
            suggestions.append({
                'title': 'Regulatory & Safety Compliance',
                'feedback': 'Include knowledge of health, safety, or environmental (HSE) standards.',
                'example': 'Ensured compliance with HSE regulations during facility upgrades and maintenance.'
            })

        if 'troubleshooting' in jd_lower and 'troubleshooting' not in cv_lower:
            suggestions.append({
                'title': 'System Troubleshooting',
                'feedback': 'Demonstrate problem-solving or diagnostic experience with technical systems.',
                'example': 'Diagnosed electrical faults and performed root cause analysis to restore system functionality.'
            })

        if any(x in jd_lower for x in ['equipment installation', 'commissioning']) and not any(x in cv_lower for x in ['installation', 'commissioning']):
            suggestions.append({
                'title': 'Installation and Commissioning',
                'feedback': 'Mention projects involving the setup or commissioning of equipment or systems.',
                'example': 'Led the installation and commissioning of a new industrial HVAC system within budget and timeline.'
            })

    if field == "Finance / Accounting / Audit":
        if 'financial reporting' in jd_lower and 'financial reporting' not in cv_lower:
            suggestions.append({
                'title': 'Financial Reporting',
                'feedback': 'Include experience preparing financial statements or monthly/quarterly reports.',
                'example': 'Prepared monthly financial reports and reconciliations in compliance with IFRS standards.'
            })

        if any(x in jd_lower for x in ['ifrs', 'gaap']) and not any(x in cv_lower for x in ['ifrs', 'gaap']):
            suggestions.append({
                'title': 'Accounting Standards',
                'feedback': 'Mention your knowledge or use of IFRS or GAAP standards.',
                'example': 'Ensured compliance with IFRS reporting requirements during quarterly audits.'
            })

        if 'budget' in jd_lower and 'budget' not in cv_lower:
            suggestions.append({
                'title': 'Budget Management',
                'feedback': 'Mention experience preparing, monitoring, or managing budgets.',
                'example': 'Developed and monitored departmental budgets to ensure cost-efficiency and alignment with targets.'
            })

        if 'audit' in jd_lower and 'audit' not in cv_lower:
            suggestions.append({
                'title': 'Audit Experience',
                'feedback': 'Include internal or external audit experience, if relevant.',
                'example': 'Participated in internal audits to identify compliance gaps and recommend corrective actions.'
            })

        if any(x in jd_lower for x in ['tax', 'vat']) and not any(x in cv_lower for x in ['tax', 'vat']):
            suggestions.append({
                'title': 'Tax Compliance',
                'feedback': 'Mention experience with tax computations, filing, or compliance.',
                'example': 'Handled monthly VAT filings and annual tax computations for the organization.'
            })

        if any(x in jd_lower for x in ['reconciliation', 'bank reconciliation']) and 'reconciliation' not in cv_lower:
            suggestions.append({
                'title': 'Reconciliation Skills',
                'feedback': 'Highlight experience with account or bank reconciliations.',
                'example': 'Performed weekly bank reconciliations to ensure accuracy between financial records and bank statements.'
            })

        if 'cost control' in jd_lower and 'cost control' not in cv_lower:
            suggestions.append({
                'title': 'Cost Control Initiatives',
                'feedback': 'Mention any involvement in reducing costs or improving expense efficiency.',
                'example': 'Implemented cost control measures that reduced departmental spending by 12% over two quarters.'
            })

    if field == "Hospitality / Hotel / Restaurant":
        if 'guest service' in jd_lower and 'guest service' not in cv_lower:
            suggestions.append({
                'title': 'Guest Services',
                'feedback': 'Highlight your experience attending to guests, resolving complaints, or managing guest relations.',
                'example': 'Handled guest check-ins, resolved complaints, and ensured 5-star service delivery.'
            })

        if any(x in jd_lower for x in ['food safety', 'haccp']) and not any(x in cv_lower for x in ['haccp', 'food safety']):
            suggestions.append({
                'title': 'Food Safety Standards',
                'feedback': 'Mention knowledge of food safety procedures or certifications like HACCP.',
                'example': 'Maintained HACCP compliance and ensured kitchen hygiene standards were strictly followed.'
            })

        if 'event' in jd_lower and 'event' not in cv_lower:
            suggestions.append({
                'title': 'Event Management',
                'feedback': 'Mention experience with planning, setting up, or coordinating events.',
                'example': 'Coordinated banquet events and ensured smooth execution of weddings and conferences.'
            })

        if 'front desk' in jd_lower and 'front desk' not in cv_lower:
            suggestions.append({
                'title': 'Front Desk Operations',
                'feedback': 'Include tasks such as handling reservations, check-ins, or switchboard operations.',
                'example': 'Managed front desk operations including check-ins, check-outs, and room reservations.'
            })

        if 'inventory' in jd_lower and 'inventory' not in cv_lower:
            suggestions.append({
                'title': 'Inventory or Stock Management',
                'feedback': 'Mention roles involving inventory tracking, stocktaking, or supplier coordination.',
                'example': 'Maintained accurate kitchen inventory and coordinated with suppliers for daily stock replenishment.'
            })

        if any(x in jd_lower for x in ['bartender', 'bar', 'cocktail']) and not any(x in cv_lower for x in ['bartender', 'bar', 'cocktail']):
            suggestions.append({
                'title': 'Bar Service Experience',
                'feedback': 'Include roles related to bartending, drink mixing, or bar stock handling.',
                'example': 'Worked as a bartender, creating custom cocktails and managing bar supplies.'
            })

        if any(x in jd_lower for x in ['housekeeping', 'room cleaning']) and not any(x in cv_lower for x in ['housekeeping', 'room']):
            suggestions.append({
                'title': 'Housekeeping Duties',
                'feedback': 'Mention experience in room preparation, cleaning, or laundry.',
                'example': 'Performed housekeeping tasks including room cleaning, linen replacement, and restocking amenities.'
            })

    if field == "Human Resources / HR":
        if 'recruit' in jd_lower and not any(x in cv_lower for x in ['recruit', 'sourcing', 'headhunting']):
            suggestions.append({
                'title': 'Recruitment Experience',
                'feedback': 'Mention involvement in recruitment, talent sourcing, or selection.',
                'example': 'Led end-to-end recruitment processes including job posting, CV screening, and candidate interviews.'
            })

        if any(x in jd_lower for x in ['onboarding', 'orientation']) and not any(x in cv_lower for x in ['onboarding', 'orientation']):
            suggestions.append({
                'title': 'Employee Onboarding',
                'feedback': 'Highlight experience with onboarding, induction, or orientation programs.',
                'example': 'Developed and executed onboarding programs for new hires, reducing early attrition by 30%.'
            })

        if 'hr policy' in jd_lower and 'policy' not in cv_lower:
            suggestions.append({
                'title': 'HR Policies & Compliance',
                'feedback': 'Show that you’ve worked on developing or enforcing HR policies.',
                'example': 'Drafted HR policies on employee conduct and performance appraisal, ensuring compliance with labour laws.'
            })

        if 'payroll' in jd_lower and 'payroll' not in cv_lower:
            suggestions.append({
                'title': 'Payroll Management',
                'feedback': 'Mention if you’ve handled payroll processing, salary computation, or related tools.',
                'example': 'Managed payroll for 200+ employees using Sage HR and ensured timely salary disbursements.'
            })

        if 'performance' in jd_lower and 'performance' not in cv_lower:
            suggestions.append({
                'title': 'Performance Management',
                'feedback': 'Show any experience in performance reviews or appraisal systems.',
                'example': 'Coordinated annual performance reviews and implemented KPIs across departments.'
            })

        if 'employee relations' in jd_lower and 'employee relations' not in cv_lower:
            suggestions.append({
                'title': 'Employee Relations',
                'feedback': 'Highlight conflict resolution, staff engagement, or grievance handling skills.',
                'example': 'Handled staff grievances and conducted employee engagement surveys that improved satisfaction scores.'
            })

        if any(x in jd_lower for x in ['training', 'development']) and not any(x in cv_lower for x in ['training', 'development']):
            suggestions.append({
                'title': 'Training & Development',
                'feedback': 'Mention organizing or facilitating employee training or upskilling programs.',
                'example': 'Facilitated monthly training sessions on workplace ethics and customer service excellence.'
            })

    if field == "ICT / Computer":
        if any(x in jd_lower for x in ['it support', 'helpdesk', 'technical support']):
            if not any(x in cv_lower for x in ['helpdesk', 'it support', 'technical support', 'ticketing system']):
                suggestions.append({
                    'title': 'IT Support Experience',
                    'feedback': 'Highlight experience in helpdesk or end-user technical support.',
                    'example': 'Provided tier-1 IT support resolving hardware/software issues and managing ticketing systems like Zendesk.'
                })

        if any(x in jd_lower for x in ['network', 'router', 'firewall']):
            if not any(x in cv_lower for x in ['network', 'router', 'switch', 'firewall', 'lan', 'wan']):
                suggestions.append({
                    'title': 'Networking Skills',
                    'feedback': 'Demonstrate experience with routers, switches, LAN/WAN, or firewalls.',
                    'example': 'Configured Cisco routers and managed LAN/WAN infrastructure for a 50-user network.'
                })

        if 'system admin' in jd_lower or 'server' in jd_lower:
            if not any(x in cv_lower for x in ['system administrator', 'sysadmin', 'active directory', 'windows server', 'linux']):
                suggestions.append({
                    'title': 'System Administration',
                    'feedback': 'Mention your experience managing servers or user accounts.',
                    'example': 'Managed Windows Server 2019 environments and handled user access through Active Directory.'
                })

        if 'cybersecurity' in jd_lower or 'security' in jd_lower:
            if not any(x in cv_lower for x in ['cybersecurity', 'security audit', 'vulnerability', 'threat']):
                suggestions.append({
                    'title': 'Cybersecurity Awareness',
                    'feedback': 'Include experience with security protocols or threat detection tools.',
                    'example': 'Performed routine system vulnerability checks and enforced endpoint security policies.'
                })

        if 'hardware' in jd_lower:
            if not any(x in cv_lower for x in ['hardware', 'repair', 'installation', 'maintenance']):
                suggestions.append({
                    'title': 'Hardware Maintenance',
                    'feedback': 'List your experience repairing or maintaining computers or printers.',
                    'example': 'Repaired desktop computers, replaced faulty hardware components, and handled printer maintenance.'
                })

        if 'software' in jd_lower:
            if not any(x in cv_lower for x in ['software installation', 'software support', 'patching', 'license']):
                suggestions.append({
                    'title': 'Software Support',
                    'feedback': 'Show knowledge of installing, configuring, or troubleshooting applications.',
                    'example': 'Installed licensed software for 100+ users and provided troubleshooting for application errors.'
                })

        if 'cloud' in jd_lower:
            if not any(x in cv_lower for x in ['aws', 'azure', 'gcp', 'cloud']):
                suggestions.append({
                    'title': 'Cloud Technologies',
                    'feedback': 'Include cloud platforms you’ve worked with such as AWS or Azure.',
                    'example': 'Deployed virtual machines and storage services using Microsoft Azure.'
                })

        if 'script' in jd_lower or 'automation' in jd_lower:
            if not any(x in cv_lower for x in ['bash', 'powershell', 'automation', 'scripting']):
                suggestions.append({
                    'title': 'Scripting and Automation',
                    'feedback': 'Mention any use of scripting languages to automate IT tasks.',
                    'example': 'Automated server backup tasks using PowerShell scripts.'
                })

    if field == "Programming & Development":
        if 'python' in jd_lower and 'python' not in cv_lower:
            suggestions.append({
                'title': 'Python Experience',
                'feedback': 'Python is a core requirement for this role, but your CV doesn’t mention it. Include relevant experience if applicable.',
                'example': 'Developed RESTful APIs and backend services using Python and Django.'
            })
    
        if 'javascript' in jd_lower and 'javascript' not in cv_lower:
            suggestions.append({
                'title': 'JavaScript Skills',
                'feedback': 'The job calls for JavaScript expertise. Add your experience with JavaScript, frameworks, or frontend logic.',
                'example': 'Built dynamic user interfaces using JavaScript and modern frameworks like React.'
            })
    
        if 'react' in jd_lower and 'react' not in cv_lower:
            suggestions.append({
                'title': 'React Framework',
                'feedback': 'React is mentioned in the job description, but missing from your CV. Include it if you have experience.',
                'example': 'Implemented complex frontend components using React and Redux.'
            })
    
        if 'api' in jd_lower and 'api' not in cv_lower:
            suggestions.append({
                'title': 'API Integration / Development',
                'feedback': 'API knowledge is required. Mention experience with building or consuming APIs.',
                'example': 'Integrated third-party APIs and built custom RESTful services.'
            })
    
        if 'git' in jd_lower and 'git' not in cv_lower:
            suggestions.append({
                'title': 'Version Control (Git)',
                'feedback': 'Git or version control isn’t mentioned on your CV. Include this if you’ve worked with it.',
                'example': 'Used Git for version control and collaborative development via GitHub.'
            })
    
        if 'agile' in jd_lower and 'agile' not in cv_lower:
            suggestions.append({
                'title': 'Agile Methodologies',
                'feedback': 'Agile is a key work style in development. Highlight experience with sprints or Scrum practices.',
                'example': 'Collaborated in Agile teams using Scrum methodology and participated in sprint planning and reviews.'
            })
    
        if 'typescript' in jd_lower and 'typescript' not in cv_lower:
            suggestions.append({
                'title': 'TypeScript Knowledge',
                'feedback': 'TypeScript is listed but not reflected on your CV. Mention it if relevant.',
                'example': 'Built scalable frontend components using TypeScript with Angular and React.'
            })
    
        if 'node.js' in jd_lower and all(x not in cv_lower for x in ['node', 'node.js']):
            suggestions.append({
                'title': 'Node.js Backend Skills',
                'feedback': 'Node.js appears in the JD but not in your CV. Add relevant backend experience if any.',
                'example': 'Developed REST APIs and real-time services using Node.js and Express.'
            })
    
        if 'database' in jd_lower and not any(x in cv_lower for x in ['sql', 'mysql', 'postgresql', 'mongodb']):
            suggestions.append({
                'title': 'Database Technologies',
                'feedback': 'Database skills are needed for this role. Include any SQL or NoSQL experience.',
                'example': 'Designed and optimized MySQL queries; also worked with MongoDB for document-based data.'
            })
    
        if 'deployment' in jd_lower and 'deployment' not in cv_lower:
            suggestions.append({
                'title': 'Deployment Experience',
                'feedback': 'Deployment responsibilities are mentioned. Add details if you’ve deployed applications.',
                'example': 'Deployed web applications to AWS EC2 and managed CI/CD pipelines.'
            })

    if field == "UI/UX & Design":
        if 'figma' in jd_lower and 'figma' not in cv_lower:
            suggestions.append({
                'title': 'Figma Proficiency',
                'feedback': 'Figma is a core tool in UI/UX jobs. Add it if you’ve used it for design, prototyping, or collaboration.',
                'example': 'Designed interactive product prototypes and high-fidelity UI screens using Figma.'
            })
    
        if 'user research' in jd_lower and 'user research' not in cv_lower:
            suggestions.append({
                'title': 'User Research Skills',
                'feedback': 'This role involves user research, but your CV doesn’t reflect that. Include methods like interviews, surveys, or usability tests.',
                'example': 'Conducted user interviews and usability testing to inform design decisions.'
            })
    
        if 'ux writing' in jd_lower and 'ux writing' not in cv_lower:
            suggestions.append({
                'title': 'UX Writing',
                'feedback': 'UX writing is expected but missing on your CV. Include if you’ve worked on microcopy or content strategy.',
                'example': 'Wrote intuitive microcopy for app onboarding and error messages, improving user guidance.'
            })
    
        if 'wireframe' in jd_lower and 'wireframe' not in cv_lower:
            suggestions.append({
                'title': 'Wireframing Experience',
                'feedback': 'Wireframing is listed in the job, but your CV doesn’t mention it. Add tools or examples if relevant.',
                'example': 'Created wireframes using Balsamiq and Figma to map early-stage product flows.'
            })
    
        if 'design system' in jd_lower and 'design system' not in cv_lower:
            suggestions.append({
                'title': 'Design Systems',
                'feedback': 'Design system experience is requested. Include it if you’ve worked with or built one.',
                'example': 'Maintained and extended the company’s Figma-based design system for scalable product UI.'
            })
    
        if 'accessibility' in jd_lower and 'accessibility' not in cv_lower:
            suggestions.append({
                'title': 'Accessibility Standards',
                'feedback': 'Accessibility is mentioned, but not reflected on your CV. Mention WCAG compliance or accessibility testing.',
                'example': 'Designed components compliant with WCAG guidelines to ensure usability for all users.'
            })
    
        if 'adobe' in jd_lower and not any(x in cv_lower for x in ['photoshop', 'illustrator', 'adobe xd']):
            suggestions.append({
                'title': 'Adobe Tools',
                'feedback': 'Adobe tools are listed, but your CV doesn’t reflect experience with Photoshop, Illustrator, or XD.',
                'example': 'Used Adobe XD for wireframing and Illustrator for visual design assets.'
            })
    
        if 'interaction design' in jd_lower and 'interaction design' not in cv_lower:
            suggestions.append({
                'title': 'Interaction Design',
                'feedback': 'Interaction design is required, but not shown on your CV. Add if you’ve worked on animations, transitions, or flows.',
                'example': 'Designed intuitive user flows and transitions for a mobile e-commerce application.'
            })
    
        if 'prototyping' in jd_lower and 'prototype' not in cv_lower:
            suggestions.append({
                'title': 'Prototyping Skills',
                'feedback': 'Prototyping is mentioned in the JD. Add if you’ve created interactive prototypes for testing or stakeholder feedback.',
                'example': 'Built interactive prototypes in Figma to validate features with stakeholders before development.'
            })
    
        if 'ui design' in jd_lower and 'ui design' not in cv_lower:
            suggestions.append({
                'title': 'UI Design',
                'feedback': 'UI design is listed in the role but missing on your CV. Be sure to include layout or visual work.',
                'example': 'Designed responsive UI layouts for a fintech dashboard across web and mobile.'
            })

    if field == "DevOps":
        if 'ci/cd' in jd_lower and not any(x in cv_lower for x in ['ci/cd', 'jenkins', 'github actions', 'gitlab ci']):
            suggestions.append({
                'title': 'CI/CD Pipeline Experience',
                'feedback': 'CI/CD is a core requirement, but your CV doesn’t reflect experience in this area. Add relevant tools or pipelines you’ve worked with.',
                'example': 'Implemented CI/CD pipelines using GitHub Actions to automate testing and deployments.'
            })
    
        if 'docker' in jd_lower and 'docker' not in cv_lower:
            suggestions.append({
                'title': 'Docker',
                'feedback': 'Docker is mentioned, but not found on your CV. Include it if you’ve containerized applications or managed images.',
                'example': 'Containerized microservices using Docker to standardize environments across dev and prod.'
            })
    
        if 'kubernetes' in jd_lower and 'kubernetes' not in cv_lower:
            suggestions.append({
                'title': 'Kubernetes Experience',
                'feedback': 'Kubernetes is a key part of the job, but not reflected on your CV. Mention if you’ve used it for orchestration or scaling.',
                'example': 'Deployed and scaled containerized apps on Kubernetes using Helm and kubectl.'
            })
    
        if 'cloud' in jd_lower and not any(x in cv_lower for x in ['aws', 'azure', 'gcp']):
            suggestions.append({
                'title': 'Cloud Platforms',
                'feedback': 'The role requires cloud experience, but your CV doesn’t show any. Mention AWS, Azure, or GCP if applicable.',
                'example': 'Managed cloud infrastructure on AWS using EC2, S3, and CloudFormation.'
            })
    
        if 'infrastructure as code' in jd_lower and not any(x in cv_lower for x in ['terraform', 'cloudformation', 'pulumi']):
            suggestions.append({
                'title': 'Infrastructure as Code (IaC)',
                'feedback': 'IaC is mentioned but missing from your CV. Add if you’ve used Terraform, CloudFormation, etc.',
                'example': 'Used Terraform to provision and manage infrastructure as code across multiple environments.'
            })
    
        if 'monitoring' in jd_lower and not any(x in cv_lower for x in ['prometheus', 'grafana', 'datadog', 'cloudwatch']):
            suggestions.append({
                'title': 'Monitoring Tools',
                'feedback': 'Monitoring is part of the job but not reflected on your CV. Include tools like Prometheus, Grafana, or CloudWatch.',
                'example': 'Set up Prometheus and Grafana dashboards to monitor app performance and system health.'
            })
    
        if 'linux' in jd_lower and 'linux' not in cv_lower:
            suggestions.append({
                'title': 'Linux Skills',
                'feedback': 'Linux administration is a typical requirement in DevOps, but not seen on your CV.',
                'example': 'Managed server configuration and automation on Ubuntu and CentOS environments.'
            })
    
        if 'ansible' in jd_lower and 'ansible' not in cv_lower:
            suggestions.append({
                'title': 'Configuration Management',
                'feedback': 'Ansible is listed but not reflected on your CV. Include it if you’ve used it for server setup or provisioning.',
                'example': 'Automated server configuration and deployments using Ansible playbooks.'
            })
    
        if 'scripting' in jd_lower and not any(x in cv_lower for x in ['bash', 'shell', 'python']):
            suggestions.append({
                'title': 'Scripting & Automation',
                'feedback': 'Scripting is part of the JD, but your CV doesn’t show relevant skills. Include languages like Bash or Python.',
                'example': 'Wrote Bash scripts to automate log rotation, backups, and service restarts.'
            })
    
        if 'site reliability' in jd_lower and 'sre' not in cv_lower and 'site reliability' not in cv_lower:
            suggestions.append({
                'title': 'SRE Knowledge',
                'feedback': 'Site Reliability Engineering is part of the role, but your CV doesn’t reflect SRE responsibilities or mindset.',
                'example': 'Applied SRE practices to improve service uptime and manage SLIs, SLOs, and SLAs.'
            })

    if field == "Testing / QA":
        if 'test case' in jd_lower and 'test case' not in cv_lower:
            suggestions.append({
                'title': 'Test Case Development',
                'feedback': 'Mention your experience designing or writing test cases.',
                'example': 'Designed detailed manual and automated test cases for new feature rollouts.'
            })
    
        if 'bug' in jd_lower and 'bug' not in cv_lower:
            suggestions.append({
                'title': 'Bug Tracking',
                'feedback': 'Include experience identifying, logging, and tracking software bugs.',
                'example': 'Reported and tracked bugs using Jira and collaborated with developers to resolve critical issues.'
            })
    
        if any(term in jd_lower for term in ['automated testing', 'selenium', 'test automation']) and not any(x in cv_lower for x in ['automation', 'automated test', 'selenium']):
            suggestions.append({
                'title': 'Automated Testing',
                'feedback': 'Highlight experience with automation frameworks if mentioned in the job description.',
                'example': 'Implemented automated test scripts using Selenium WebDriver for regression testing.'
            })
    
        if 'qa process' in jd_lower and 'qa' not in cv_lower:
            suggestions.append({
                'title': 'Quality Assurance Process',
                'feedback': 'Explain your contribution to ensuring product quality through QA processes.',
                'example': 'Participated in end-to-end QA processes ensuring compliance with software quality standards.'
            })
    
        if 'tools' in jd_lower and not any(tool in cv_lower for tool in ['postman', 'jira', 'testrail', 'selenium', 'cypress']):
            suggestions.append({
                'title': 'QA Tools',
                'feedback': 'Mention QA tools you have used that align with the job description.',
                'example': 'Used Postman for API testing and TestRail for test case management.'
            })

    if field == "Product Management":
        if 'roadmap' in jd_lower and 'roadmap' not in cv_lower:
            suggestions.append({
                'title': 'Product Roadmap',
                'feedback': 'Mention your experience creating or managing a product roadmap.',
                'example': 'Developed quarterly product roadmaps aligned with customer needs and company goals.'
            })
    
        if 'stakeholders' in jd_lower and 'stakeholders' not in cv_lower:
            suggestions.append({
                'title': 'Stakeholder Engagement',
                'feedback': 'Show how you worked with stakeholders to shape product direction.',
                'example': 'Collaborated with engineering, sales, and marketing stakeholders to prioritize product features.'
            })
    
        if 'market research' in jd_lower and 'market' not in cv_lower:
            suggestions.append({
                'title': 'Market Research',
                'feedback': 'Include any user research or market validation efforts you’ve led.',
                'example': 'Conducted competitor and market research to guide MVP feature prioritization.'
            })
    
        if 'kpi' in jd_lower and not any(k in cv_lower for k in ['kpi', 'metrics', 'product success']):
            suggestions.append({
                'title': 'Product KPIs',
                'feedback': 'Discuss how you measured product success using KPIs or data.',
                'example': 'Defined and tracked KPIs such as user retention and conversion rate post-launch.'
            })
    
        if 'cross-functional' in jd_lower and 'cross-functional' not in cv_lower:
            suggestions.append({
                'title': 'Cross-functional Collaboration',
                'feedback': 'Mention any experience working across teams (design, dev, marketing).',
                'example': 'Led cross-functional teams to deliver product features on time and within scope.'
            })
    
        if 'agile' in jd_lower and 'agile' not in cv_lower:
            suggestions.append({
                'title': 'Agile Environment',
                'feedback': 'If you’ve worked in agile teams, include that experience.',
                'example': 'Managed backlog and sprint planning as part of an agile product development team.'
            })
    
        if 'user stories' in jd_lower and 'user stories' not in cv_lower:
            suggestions.append({
                'title': 'User Stories',
                'feedback': 'Include your experience writing or refining user stories.',
                'example': 'Created detailed user stories and acceptance criteria to align development with business needs.'
            })
    
        if any(term in jd_lower for term in ['product manager', 'product owner']) and 'product manager' not in cv_lower:
            suggestions.append({
                'title': 'Product Manager Role',
                'feedback': 'If the job is explicitly for a Product Manager, make sure your title or experience reflects this clearly.',
                'example': 'Worked as a Product Manager leading end-to-end product development from ideation to launch.'
            })

    if field == "Project Management":
        if 'project lifecycle' in jd_lower and 'project lifecycle' not in cv_lower:
            suggestions.append({
                'title': 'Project Lifecycle Understanding',
                'feedback': 'Highlight your experience managing full project lifecycles, from initiation to delivery.',
                'example': 'Oversaw end-to-end delivery of software projects from planning to deployment across agile teams.'
            })
    
        if 'scrum' in jd_lower and 'scrum' not in cv_lower:
            suggestions.append({
                'title': 'Scrum Methodology',
                'feedback': 'Include your familiarity or certification with Scrum methodologies if relevant.',
                'example': 'Facilitated daily standups and sprint reviews as a certified Scrum Master for cross-functional teams.'
            })
    
        if 'stakeholder' in jd_lower and 'stakeholder' not in cv_lower:
            suggestions.append({
                'title': 'Stakeholder Communication',
                'feedback': 'Demonstrate your ability to engage or report to stakeholders across the project lifecycle.',
                'example': 'Liaised with internal and external stakeholders to align project scope and deliverables.'
            })
    
        if 'budget' in jd_lower and 'budget' not in cv_lower:
            suggestions.append({
                'title': 'Budget or Resource Management',
                'feedback': 'If applicable, mention budget oversight or efficient resource management on tech projects.',
                'example': 'Managed project budgets of up to $250,000 and reallocated resources to meet tight timelines.'
            })
    
        if 'jira' in jd_lower and 'jira' not in cv_lower:
            suggestions.append({
                'title': 'Project Management Tools (Jira)',
                'feedback': 'Mention Jira or other PM tools if used for task tracking and sprint management.',
                'example': 'Utilized Jira for backlog grooming, sprint planning, and monitoring team velocity.'
            })
    
        if 'timeline' in jd_lower and 'timeline' not in cv_lower:
            suggestions.append({
                'title': 'Timeline Management',
                'feedback': 'Show your ability to deliver projects on time or manage shifting deadlines effectively.',
                'example': 'Delivered complex web projects 2 weeks ahead of schedule through proactive sprint planning.'
            })

    if field == "Insurance":
        if 'underwriting' in jd_lower and 'underwriting' not in cv_lower:
            suggestions.append({
                'title': 'Underwriting Knowledge',
                'feedback': 'Include your experience or familiarity with risk assessment or underwriting processes.',
                'example': 'Performed risk evaluation and underwriting for SME business clients using data-driven models.'
            })
    
        if 'claims' in jd_lower and 'claims' not in cv_lower:
            suggestions.append({
                'title': 'Claims Processing',
                'feedback': 'Mention experience with claims review, assessment, or resolution.',
                'example': 'Processed over 200 auto insurance claims, ensuring quick resolution and minimal client churn.'
            })
    
        if 'policy administration' in jd_lower and 'policy administration' not in cv_lower:
            suggestions.append({
                'title': 'Policy Administration',
                'feedback': 'Highlight tasks involving policy issuance, renewals, endorsements, or cancellations.',
                'example': 'Managed life insurance policy administration including endorsements and renewals for 500+ clients.'
            })
    
        if 'actuarial' in jd_lower and 'actuarial' not in cv_lower:
            suggestions.append({
                'title': 'Actuarial or Statistical Analysis',
                'feedback': 'If applicable, mention actuarial tasks like risk modeling or premium calculation.',
                'example': 'Supported actuarial team in pricing strategies using mortality and claims data trends.'
            })
    
        if 'insurance software' in jd_lower and 'insurance software' not in cv_lower:
            suggestions.append({
                'title': 'Insurance-Specific Software',
                'feedback': 'Include tools like Guidewire, Applied Epic, or proprietary claims/policy software.',
                'example': 'Utilized Guidewire PolicyCenter for quote generation and policy management.'
            })

    if field == "Law / Legal":
        if 'legal research' in jd_lower and 'legal research' not in cv_lower:
            suggestions.append({
                'title': 'Legal Research',
                'feedback': 'Highlight your experience with researching statutes, case law, or legal precedents.',
                'example': 'Conducted legal research to support litigation on commercial dispute cases, ensuring accurate case citations.'
            })
    
        if 'drafting' in jd_lower and 'drafting' not in cv_lower:
            suggestions.append({
                'title': 'Legal Drafting Skills',
                'feedback': 'Include your ability to draft contracts, pleadings, affidavits, or legal opinions.',
                'example': 'Drafted commercial contracts and NDAs, reducing client risk exposure in cross-border transactions.'
            })
    
        if 'litigation' in jd_lower and 'litigation' not in cv_lower:
            suggestions.append({
                'title': 'Litigation Experience',
                'feedback': 'Mention your exposure to civil/criminal litigation, court procedures, or trial preparation.',
                'example': 'Supported litigation team in case preparation, filings, and court appearances at magistrate and high courts.'
            })
    
        if 'compliance' in jd_lower and 'compliance' not in cv_lower:
            suggestions.append({
                'title': 'Regulatory Compliance',
                'feedback': 'Show familiarity with compliance frameworks, regulatory audits, or policy reviews.',
                'example': 'Monitored legal compliance with AML and data protection laws across company departments.'
            })
    
        if 'contract review' in jd_lower and 'contract review' not in cv_lower:
            suggestions.append({
                'title': 'Contract Review & Negotiation',
                'feedback': 'Add details of contract review, risk flagging, and negotiation support if applicable.',
                'example': 'Reviewed supplier contracts to flag risk clauses and negotiated terms favourable to company interests.'
            })
    
        if 'corporate law' in jd_lower and 'corporate law' not in cv_lower:
            suggestions.append({
                'title': 'Corporate Law Knowledge',
                'feedback': 'Include experience advising on business formation, governance, or shareholder issues.',
                'example': 'Advised startups on company incorporation, board structure, and regulatory filings.'
            })
    
        if 'due diligence' in jd_lower and 'due diligence' not in cv_lower:
            suggestions.append({
                'title': 'Due Diligence Support',
                'feedback': 'Mention M&A or compliance due diligence, especially for transactional or commercial law roles.',
                'example': 'Conducted legal due diligence for acquisition targets, reviewing corporate filings and liabilities.'
            })
    
        if 'law firm' in jd_lower and 'law firm' not in cv_lower:
            suggestions.append({
                'title': 'Work in Legal Practice or Chambers',
                'feedback': 'Mention prior experience in a law firm, legal clinic, or court internship.',
                'example': 'Interned at XYZ Chambers, assisting with legal drafting, file prep, and court submissions.'
            })

    if field == "Logistics":
        if 'supply chain' in jd_lower and 'supply chain' not in cv_lower:
            suggestions.append({
                'title': 'Supply Chain Knowledge',
                'feedback': 'Mention your experience coordinating or optimizing supply chain activities, if applicable.',
                'example': 'Coordinated end-to-end supply chain operations from procurement to last-mile delivery.'
            })
    
        if 'inventory' in jd_lower and 'inventory' not in cv_lower:
            suggestions.append({
                'title': 'Inventory Management',
                'feedback': 'Include experience with inventory control, stock audits, or warehouse systems.',
                'example': 'Implemented an automated inventory tracking system, reducing stock variance by 20%.'
            })
    
        if 'fleet' in jd_lower and 'fleet' not in cv_lower:
            suggestions.append({
                'title': 'Fleet Management',
                'feedback': 'Highlight any experience managing transportation fleet, maintenance, or routing.',
                'example': 'Oversaw a delivery fleet of 30 vehicles, optimizing route schedules to reduce fuel costs.'
            })
    
        if 'logistics software' in jd_lower and 'logistics software' not in cv_lower:
            suggestions.append({
                'title': 'Logistics Software Proficiency',
                'feedback': 'Mention relevant logistics or ERP software (e.g., SAP, Oracle, Odoo, TMS) you’ve used.',
                'example': 'Used SAP SCM to track inventory movement and generate logistics performance reports.'
            })
    
        if 'warehouse' in jd_lower and 'warehouse' not in cv_lower:
            suggestions.append({
                'title': 'Warehouse Operations',
                'feedback': 'Include warehouse-related duties like loading, storage, picking/packing, or layout planning.',
                'example': 'Managed warehouse layout optimization, improving picking speed and storage efficiency.'
            })
    
        if 'delivery' in jd_lower and 'delivery' not in cv_lower:
            suggestions.append({
                'title': 'Delivery Coordination',
                'feedback': 'Showcase your experience planning, tracking, or improving delivery operations.',
                'example': 'Coordinated daily deliveries across 12 states, achieving 96% on-time rate.'
            })
    
        if 'import' in jd_lower or 'export' in jd_lower:
            if 'import' in jd_lower and 'import' not in cv_lower:
                suggestions.append({
                    'title': 'Import Logistics Experience',
                    'feedback': 'Highlight your knowledge of customs, documentation, or freight handling.',
                    'example': 'Handled import documentation and clearing processes for high-value consignments.'
                })
            if 'export' in jd_lower and 'export' not in cv_lower:
                suggestions.append({
                    'title': 'Export Coordination',
                    'feedback': 'Include export compliance, shipment tracking, or vendor coordination experience.',
                    'example': 'Coordinated export shipments, ensuring all compliance documentation met customs requirements.'
                })
    
        if 'route optimization' in jd_lower and 'route optimization' not in cv_lower:
            suggestions.append({
                'title': 'Route Optimization',
                'feedback': 'Mention your ability to plan efficient routes for cost-saving and delivery speed.',
                'example': 'Used GPS and route planning tools to optimize delivery schedules and reduce turnaround time.'
            })
    
        if 'logistics coordination' in jd_lower and 'logistics coordination' not in cv_lower:
            suggestions.append({
                'title': 'Logistics Coordination',
                'feedback': 'Describe your coordination efforts across departments, vendors, or field teams.',
                'example': 'Liaised with suppliers, drivers, and warehouse teams to ensure timely order fulfillment.'
            })

    if field == "Manufacturing":
        if 'production planning' in jd_lower and 'production planning' not in cv_lower:
            suggestions.append({
                'title': 'Production Planning',
                'feedback': 'Highlight your role in scheduling production, managing timelines, or meeting output targets.',
                'example': 'Planned and executed weekly production schedules to meet 98% of customer orders on time.'
            })
    
        if 'quality control' in jd_lower and 'quality control' not in cv_lower:
            suggestions.append({
                'title': 'Quality Control',
                'feedback': 'Include responsibilities around inspecting, testing, or enforcing product standards.',
                'example': 'Performed in-process quality checks to ensure compliance with ISO 9001 standards.'
            })
    
        if 'machine operation' in jd_lower and 'machine operation' not in cv_lower:
            suggestions.append({
                'title': 'Machine Operation Skills',
                'feedback': 'Mention machines or equipment you’ve operated, maintained, or calibrated.',
                'example': 'Operated CNC machines to produce precision parts for automotive components.'
            })
    
        if 'lean manufacturing' in jd_lower and 'lean' not in cv_lower:
            suggestions.append({
                'title': 'Lean Manufacturing Knowledge',
                'feedback': 'Include familiarity with lean methods such as 5S, Kaizen, or Six Sigma.',
                'example': 'Led Kaizen events that reduced production waste by 18%.'
            })
    
        if 'health and safety' in jd_lower and 'safety' not in cv_lower:
            suggestions.append({
                'title': 'Health and Safety Compliance',
                'feedback': 'Show your experience ensuring safe working environments or adhering to HSE standards.',
                'example': 'Trained factory workers on safety procedures, resulting in a 50% drop in incidents.'
            })
    
        if 'assembly line' in jd_lower and 'assembly' not in cv_lower:
            suggestions.append({
                'title': 'Assembly Line Experience',
                'feedback': 'State your involvement in assembling, inspecting, or improving line processes.',
                'example': 'Worked on an automated assembly line, ensuring efficient part alignment and minimal defects.'
            })
    
        if 'preventive maintenance' in jd_lower and 'maintenance' not in cv_lower:
            suggestions.append({
                'title': 'Preventive Maintenance',
                'feedback': 'Include routine checks, equipment servicing, or downtime reduction efforts.',
                'example': 'Implemented preventive maintenance schedules that reduced machine breakdowns by 30%.'
            })
    
        if 'manufacturing software' in jd_lower and 'erp' not in cv_lower and 'sap' not in cv_lower:
            suggestions.append({
                'title': 'ERP or Manufacturing Software Proficiency',
                'feedback': 'Mention platforms like SAP, Oracle Manufacturing, or MES if used.',
                'example': 'Used SAP MRP to track production materials and schedule jobs efficiently.'
            })
    
        if 'technical drawings' in jd_lower and 'technical drawing' not in cv_lower and 'blueprint' not in cv_lower:
            suggestions.append({
                'title': 'Technical Drawing Interpretation',
                'feedback': 'Highlight your ability to read blueprints, schematics, or CAD diagrams.',
                'example': 'Interpreted mechanical blueprints to fabricate components to exact specifications.'
            })
    
        if 'packaging' in jd_lower and 'packaging' not in cv_lower:
            suggestions.append({
                'title': 'Packaging & Finishing',
                'feedback': 'Include tasks involving final product packaging, labeling, or dispatch.',
                'example': 'Led a packaging line team that increased throughput by 22%.'
            })

    if field == "Media / Advertising / Branding":
        if 'brand strategy' in jd_lower and 'brand strategy' not in cv_lower:
            suggestions.append({
                'title': 'Brand Strategy',
                'feedback': 'Highlight any involvement in crafting, executing, or overseeing brand strategy.',
                'example': 'Led the development of a refreshed brand strategy that increased audience engagement by 35%.'
            })
    
        if 'content creation' in jd_lower and 'content creation' not in cv_lower:
            suggestions.append({
                'title': 'Content Creation',
                'feedback': 'Include your skills or portfolio in creating compelling written, visual, or multimedia content.',
                'example': 'Produced weekly video content for YouTube and Instagram, growing followers by 20k in 6 months.'
            })
    
        if 'campaign' in jd_lower and 'campaign' not in cv_lower:
            suggestions.append({
                'title': 'Advertising Campaigns',
                'feedback': 'Emphasize your experience planning, executing, or measuring advertising campaigns.',
                'example': 'Managed digital ad campaigns across Meta and Google Ads, achieving a 4.8x return on ad spend.'
            })
    
        if 'social media' in jd_lower and 'social media' not in cv_lower:
            suggestions.append({
                'title': 'Social Media Marketing',
                'feedback': 'Mention your proficiency in managing or growing social platforms, especially with tools or strategy.',
                'example': 'Built and managed a brand’s social media presence across 5 platforms using Hootsuite and native tools.'
            })
    
        if 'copywriting' in jd_lower and 'copywriting' not in cv_lower:
            suggestions.append({
                'title': 'Copywriting',
                'feedback': 'Show your experience writing compelling ad copy, headlines, or promotional material.',
                'example': 'Crafted persuasive ad copy and landing pages that increased email signups by 42%.'
            })
    
        if 'media buying' in jd_lower and 'media buying' not in cv_lower:
            suggestions.append({
                'title': 'Media Buying / Planning',
                'feedback': 'Include experience with negotiating, purchasing, or planning media across channels.',
                'example': 'Planned and executed TV and radio media buys worth ₦50M+, optimizing for maximum reach and cost-efficiency.'
            })
    
        if 'creative direction' in jd_lower and 'creative direction' not in cv_lower:
            suggestions.append({
                'title': 'Creative Direction',
                'feedback': 'Highlight leadership in conceptualizing or overseeing visual campaigns or storytelling.',
                'example': 'Directed a team of designers and videographers to deliver a national rebranding campaign.'
            })
    
        if 'ad copy' in jd_lower and 'ad copy' not in cv_lower:
            suggestions.append({
                'title': 'Ad Copywriting',
                'feedback': 'Specify your contributions to writing copy that resonates with target audiences and meets goals.',
                'example': 'Wrote conversion-driven ad copy for ecommerce clients with CTRs exceeding industry benchmarks.'
            })

    if field == "Medical / Healthcare":
        if 'clinical' in jd_lower and 'clinical' not in cv_lower:
            suggestions.append({
                'title': 'Clinical Experience',
                'feedback': 'Emphasize your direct patient care or clinical rotation experience relevant to the role.',
                'example': 'Completed 12 months of clinical rotations in internal medicine, pediatrics, and emergency care.'
            })
    
        if 'diagnosis' in jd_lower and 'diagnosis' not in cv_lower:
            suggestions.append({
                'title': 'Diagnostic Skills',
                'feedback': 'Mention your ability to assess symptoms, interpret tests, or contribute to medical diagnosis.',
                'example': 'Skilled in diagnosing common respiratory and gastrointestinal conditions through clinical assessments.'
            })
    
        if 'patient care' in jd_lower and 'patient care' not in cv_lower:
            suggestions.append({
                'title': 'Patient Care Competence',
                'feedback': 'Include experience offering compassionate and effective care to diverse patient populations.',
                'example': 'Provided holistic patient care in outpatient and inpatient settings, ensuring high recovery rates.'
            })
    
        if 'treatment' in jd_lower and 'treatment' not in cv_lower:
            suggestions.append({
                'title': 'Treatment Administration',
                'feedback': 'Highlight your ability to recommend, prescribe, or support treatment procedures.',
                'example': 'Administered IV medications and assisted in minor surgical procedures under physician supervision.'
            })
    
        if 'emr' in jd_lower and 'emr' not in cv_lower and 'electronic medical records' not in cv_lower:
            suggestions.append({
                'title': 'Electronic Medical Records (EMR)',
                'feedback': 'Mention your proficiency with EMR systems or digital patient data entry.',
                'example': 'Maintained accurate patient data using EMR platforms like OpenMRS and Medisoft.'
            })
    
        if 'medical license' in jd_lower and 'medical license' not in cv_lower:
            suggestions.append({
                'title': 'Licensure or Certification',
                'feedback': 'List any medical license, registration, or certifications required for practice.',
                'example': 'Licensed Medical Doctor (MDCN) with valid registration and annual practice license.'
            })
    
        if 'infection control' in jd_lower and 'infection control' not in cv_lower:
            suggestions.append({
                'title': 'Infection Control Practices',
                'feedback': 'Demonstrate your adherence to safety protocols and hygiene in clinical settings.',
                'example': 'Implemented WHO-standard infection control measures, reducing ward infection rates by 20%.'
            })
    
        if 'counseling' in jd_lower and 'counseling' not in cv_lower:
            suggestions.append({
                'title': 'Patient Counseling',
                'feedback': 'Include examples of how you educate or counsel patients on treatments, medication, or lifestyle.',
                'example': 'Conducted pre- and post-operative counseling for patients undergoing elective surgeries.'
            })

    if field == "NGO / Non-Profit":
        if 'grant writing' in jd_lower and 'grant writing' not in cv_lower:
            suggestions.append({
                'title': 'Grant Writing Experience',
                'feedback': 'Highlight your experience writing or contributing to successful grant proposals.',
                'example': 'Co-authored grant proposals that secured over $150,000 in donor funding from USAID and DFID.'
            })
    
        if 'donor' in jd_lower and 'donor' not in cv_lower:
            suggestions.append({
                'title': 'Donor Relations',
                'feedback': 'Show your experience managing donor expectations or reporting progress to funders.',
                'example': 'Managed quarterly donor reports and maintained communication with institutional funders.'
            })
    
        if 'community engagement' in jd_lower and 'community engagement' not in cv_lower:
            suggestions.append({
                'title': 'Community Engagement',
                'feedback': 'Demonstrate your ability to work with local communities, stakeholders, or target beneficiaries.',
                'example': 'Led grassroots outreach programs that impacted over 3,000 rural women in Northern Nigeria.'
            })
    
        if 'monitoring and evaluation' in jd_lower and 'monitoring and evaluation' not in cv_lower and 'm&e' not in cv_lower:
            suggestions.append({
                'title': 'Monitoring & Evaluation (M&E)',
                'feedback': 'Include your M&E experience, especially designing frameworks or analyzing impact.',
                'example': 'Developed M&E tools and analyzed program KPIs to assess effectiveness of nutrition intervention.'
            })
    
        if 'proposal writing' in jd_lower and 'proposal writing' not in cv_lower:
            suggestions.append({
                'title': 'Proposal Development',
                'feedback': 'Showcase your role in developing or contributing to project proposals.',
                'example': 'Drafted project concept notes and full proposals for UNDP-funded youth empowerment initiatives.'
            })
    
        if 'advocacy' in jd_lower and 'advocacy' not in cv_lower:
            suggestions.append({
                'title': 'Advocacy or Policy Engagement',
                'feedback': 'Mention your involvement in advocacy campaigns or policy lobbying.',
                'example': 'Coordinated advocacy campaigns on sexual health rights, reaching over 5,000 adolescents.'
            })
    
        if 'partnerships' in jd_lower and 'partnerships' not in cv_lower:
            suggestions.append({
                'title': 'Stakeholder or Partnership Building',
                'feedback': 'Include your work with local/international partners or coalitions.',
                'example': 'Forged multi-sector partnerships with government and NGOs for water sanitation projects.'
            })
    
        if 'reporting' in jd_lower and 'reporting' not in cv_lower:
            suggestions.append({
                'title': 'Program or Donor Reporting',
                'feedback': 'Highlight your experience preparing technical or narrative reports.',
                'example': 'Compiled monthly project progress reports aligned with donor M&E requirements.'
            })

    if field == "Oil and Gas / Energy":
        if 'hse' in jd_lower and 'hse' not in cv_lower and 'health safety' not in cv_lower:
            suggestions.append({
                'title': 'Health, Safety & Environment (HSE)',
                'feedback': 'Mention your understanding or certification in HSE procedures and compliance.',
                'example': 'Implemented HSE protocols that reduced onsite incidents by 30% over a 12-month period.'
            })
    
        if 'upstream' in jd_lower and 'upstream' not in cv_lower:
            suggestions.append({
                'title': 'Upstream Operations Experience',
                'feedback': 'Include relevant upstream exploration or drilling activities if applicable.',
                'example': 'Worked on upstream drilling operations across multiple onshore and offshore assets.'
            })
    
        if 'downstream' in jd_lower and 'downstream' not in cv_lower:
            suggestions.append({
                'title': 'Downstream Operations',
                'feedback': 'Highlight any refining, marketing, or distribution experience in downstream segments.',
                'example': 'Supervised product distribution and depot operations for refined petroleum products.'
            })
    
        if 'rig' in jd_lower and 'rig' not in cv_lower:
            suggestions.append({
                'title': 'Rig Operations',
                'feedback': 'Mention experience working on or with drilling rigs—onshore or offshore.',
                'example': 'Assisted in rig commissioning and monitored drilling parameters during exploratory well operations.'
            })
    
        if 'pipeline' in jd_lower and 'pipeline' not in cv_lower:
            suggestions.append({
                'title': 'Pipeline Engineering / Monitoring',
                'feedback': 'Highlight roles related to pipeline inspection, construction, or maintenance.',
                'example': 'Coordinated pipeline integrity testing and ensured compliance with environmental standards.'
            })
    
        if 'compliance' in jd_lower and 'compliance' not in cv_lower:
            suggestions.append({
                'title': 'Regulatory Compliance',
                'feedback': 'Show your ability to adhere to industry regulations and environmental standards.',
                'example': 'Ensured NUPRC and DPR regulatory compliance in daily drilling operations.'
            })
    
        if 'reservoir' in jd_lower and 'reservoir' not in cv_lower:
            suggestions.append({
                'title': 'Reservoir Management',
                'feedback': 'If relevant, indicate experience with reservoir modeling, monitoring, or production optimization.',
                'example': 'Worked with geologists and reservoir engineers to optimize well performance in mature fields.'
            })
    
        if 'renewable' in jd_lower and 'renewable' not in cv_lower:
            suggestions.append({
                'title': 'Renewable Energy Exposure',
                'feedback': 'If applicable, highlight exposure to solar, wind, or hybrid energy projects.',
                'example': 'Led feasibility studies for off-grid solar installations in rural electrification projects.'
            })

    if field == "Procurement / Store-keeping / Supply Chain":
        if 'vendor management' in jd_lower and 'vendor management' not in cv_lower:
            suggestions.append({
                'title': 'Vendor Management',
                'feedback': 'Highlight your experience in sourcing, negotiating, or evaluating vendors.',
                'example': 'Managed relationships with over 20 international and local suppliers, ensuring compliance with procurement policies.'
            })
    
        if 'inventory' in jd_lower and 'inventory' not in cv_lower and 'stock' not in cv_lower:
            suggestions.append({
                'title': 'Inventory Management',
                'feedback': 'Include experience with stock control, reorder levels, or warehouse systems.',
                'example': 'Implemented inventory tracking system that reduced stockouts by 30% and minimized holding costs.'
            })
    
        if 'supply chain' in jd_lower and 'supply chain' not in cv_lower:
            suggestions.append({
                'title': 'Supply Chain Operations',
                'feedback': 'Demonstrate knowledge of end-to-end supply chain, including logistics and procurement.',
                'example': 'Coordinated cross-border supply chain operations, reducing lead times by 20%.'
            })
    
        if 'erp' in jd_lower and 'erp' not in cv_lower:
            suggestions.append({
                'title': 'ERP Software Proficiency',
                'feedback': 'Mention any ERP systems used for procurement or inventory control (e.g., SAP, Oracle).',
                'example': 'Used SAP MM module for purchase requisitions, order tracking, and vendor invoice management.'
            })
    
        if 'cost saving' in jd_lower and 'cost saving' not in cv_lower:
            suggestions.append({
                'title': 'Cost Saving Initiatives',
                'feedback': 'Show impact on procurement costs or efficiency improvements.',
                'example': 'Negotiated bulk purchasing deals, saving the company ₦15M in annual procurement costs.'
            })
    
        if 'rfq' in jd_lower and 'rfq' not in cv_lower and 'request for quotation' not in cv_lower:
            suggestions.append({
                'title': 'RFQ/RFP Process Handling',
                'feedback': 'Include experience preparing or managing Requests for Quotation or Proposals.',
                'example': 'Drafted and evaluated RFQs for construction supplies, ensuring compliance with procurement standards.'
            })
    
        if 'warehouse' in jd_lower and 'warehouse' not in cv_lower:
            suggestions.append({
                'title': 'Warehouse Management',
                'feedback': 'Highlight warehouse operations, layout optimization, or safety protocols.',
                'example': 'Led warehouse reorganization project that improved space utilization and reduced item retrieval time by 40%.'
            })
    
        if 'logistics' in jd_lower and 'logistics' not in cv_lower:
            suggestions.append({
                'title': 'Logistics Coordination',
                'feedback': 'Mention coordination of shipping, delivery, or distribution logistics.',
                'example': 'Supervised inbound and outbound logistics across 6 Nigerian states, ensuring timely product delivery.'
            })

    if field == "Real Estate":
        if 'property management' in jd_lower and 'property management' not in cv_lower:
            suggestions.append({
                'title': 'Property Management Experience',
                'feedback': 'Mention your involvement in managing residential or commercial properties.',
                'example': 'Managed a portfolio of 30+ rental properties, ensuring tenant satisfaction and timely maintenance.'
            })
    
        if 'valuation' in jd_lower and 'valuation' not in cv_lower:
            suggestions.append({
                'title': 'Real Estate Valuation',
                'feedback': 'Highlight your skills or certifications related to valuing properties.',
                'example': 'Conducted real estate appraisals and valuations for both residential and commercial properties.'
            })
    
        if 'leasing' in jd_lower and 'leasing' not in cv_lower:
            suggestions.append({
                'title': 'Leasing & Tenancy',
                'feedback': 'Include your role in tenant acquisition, lease negotiation, or agreement management.',
                'example': 'Handled lease negotiations and tenant onboarding for a mixed-use commercial building.'
            })
    
        if 'site inspection' in jd_lower and 'site inspection' not in cv_lower:
            suggestions.append({
                'title': 'Site Inspection Experience',
                'feedback': 'Mention any experience in conducting property/site inspections and reporting.',
                'example': 'Performed routine site inspections to ensure compliance with building standards and maintenance contracts.'
            })
    
        if 'title documentation' in jd_lower and 'title documentation' not in cv_lower:
            suggestions.append({
                'title': 'Title Documentation & Verification',
                'feedback': 'Highlight your involvement in verifying or handling title documents for real estate transactions.',
                'example': 'Reviewed title documents and coordinated with legal teams to validate property ownership and transfer.'
            })
    
        if 'realtor' in jd_lower and 'realtor' not in cv_lower:
            suggestions.append({
                'title': 'Realtor Certification or Experience',
                'feedback': 'State any realtor license or experience in property brokerage or sales.',
                'example': 'Certified Realtor with experience closing residential deals worth over ₦500M.'
            })

    if field == "Safety and Environment / HSE":
        if 'hse compliance' in jd_lower and 'hse compliance' not in cv_lower:
            suggestions.append({
                'title': 'HSE Compliance',
                'feedback': 'Highlight your experience ensuring Health, Safety, and Environmental (HSE) compliance in workplace operations.',
                'example': 'Ensured strict compliance with HSE policies, reducing incident rates by 40% across site operations.'
            })
    
        if 'risk assessment' in jd_lower and 'risk assessment' not in cv_lower:
            suggestions.append({
                'title': 'Risk Assessment Expertise',
                'feedback': 'Demonstrate your ability to identify and mitigate safety risks through structured assessments.',
                'example': 'Conducted regular risk assessments and implemented mitigation plans in line with ISO 45001 standards.'
            })
    
        if 'incident investigation' in jd_lower and 'incident investigation' not in cv_lower:
            suggestions.append({
                'title': 'Incident Investigation',
                'feedback': 'Include your experience in investigating and reporting workplace incidents or near misses.',
                'example': 'Led root cause analysis and reporting for on-site incidents, improving future safety protocols.'
            })
    
        if 'safety training' in jd_lower and 'safety training' not in cv_lower:
            suggestions.append({
                'title': 'Safety Training Implementation',
                'feedback': 'Mention your role in organizing or delivering safety training programs.',
                'example': 'Facilitated monthly HSE awareness sessions for over 100 staff, boosting compliance rates.'
            })
    
        if 'ppe' in jd_lower and 'ppe' not in cv_lower:
            suggestions.append({
                'title': 'Personal Protective Equipment (PPE)',
                'feedback': 'Reflect awareness or enforcement of PPE usage in safety-sensitive environments.',
                'example': 'Implemented strict PPE compliance guidelines and monitored daily adherence across worksites.'
            })
    
        if 'environmental management' in jd_lower and 'environmental management' not in cv_lower:
            suggestions.append({
                'title': 'Environmental Management',
                'feedback': 'Highlight your involvement in minimizing environmental impact or managing waste/sustainability practices.',
                'example': 'Developed site-specific environmental management plans, leading to ISO 14001 certification.'
            })
    
        if 'iso 14001' in jd_lower and 'iso 14001' not in cv_lower:
            suggestions.append({
                'title': 'ISO 14001 Compliance',
                'feedback': 'Mention any knowledge or application of ISO 14001 Environmental Management Systems.',
                'example': 'Assisted in preparing documentation and procedures for successful ISO 14001 audit clearance.'
            })
    
        if 'hse audits' in jd_lower and 'hse audits' not in cv_lower:
            suggestions.append({
                'title': 'HSE Auditing',
                'feedback': 'Showcase your role in internal or third-party HSE audit processes.',
                'example': 'Conducted quarterly HSE audits across multiple departments, ensuring continuous safety compliance.'
            })

    if field == "Sales / Marketing / Retail / Business Development":
        if 'sales target' in jd_lower and 'sales target' not in cv_lower:
            suggestions.append({
                'title': 'Sales Target Achievement',
                'feedback': 'Demonstrate your ability to meet or exceed sales targets or quotas.',
                'example': 'Achieved 120% of quarterly sales target through strategic client acquisition and upselling.'
            })
    
        if 'business development' in jd_lower and 'business development' not in cv_lower:
            suggestions.append({
                'title': 'Business Development',
                'feedback': 'Highlight experience in identifying growth opportunities or expanding market reach.',
                'example': 'Spearheaded business development initiatives that increased client base by 35% in 12 months.'
            })
    
        if 'client acquisition' in jd_lower and 'client acquisition' not in cv_lower:
            suggestions.append({
                'title': 'Client Acquisition',
                'feedback': 'Show your success in acquiring new customers or entering new markets.',
                'example': 'Secured 50+ new clients through lead generation, cold outreach, and networking strategies.'
            })
    
        if 'crm' in jd_lower and 'crm' not in cv_lower:
            suggestions.append({
                'title': 'CRM Tools (e.g., Salesforce, HubSpot)',
                'feedback': 'Mention your experience using CRM systems to manage sales pipelines or client engagement.',
                'example': 'Used HubSpot to track sales activities and nurture leads, reducing deal closure time by 20%.'
            })
    
        if 'retail sales' in jd_lower and 'retail sales' not in cv_lower:
            suggestions.append({
                'title': 'Retail Sales Management',
                'feedback': 'If relevant, include experience with point-of-sale systems, floor supervision, or retail customer service.',
                'example': 'Managed daily retail operations and improved upsell rates by training staff on customer engagement.'
            })
    
        if 'sales funnel' in jd_lower and 'sales funnel' not in cv_lower:
            suggestions.append({
                'title': 'Sales Funnel Optimization',
                'feedback': 'Include knowledge or use of strategies to move prospects through the sales funnel.',
                'example': 'Designed lead nurturing campaigns that improved conversion rates across all funnel stages.'
            })
    
        if 'negotiation' in jd_lower and 'negotiation' not in cv_lower:
            suggestions.append({
                'title': 'Negotiation & Deal Closing',
                'feedback': 'Emphasize your skill in negotiating terms and closing high-value deals.',
                'example': 'Negotiated and closed a $500,000 annual supply deal with a multinational client.'
            })
    
        if 'market research' in jd_lower and 'market research' not in cv_lower:
            suggestions.append({
                'title': 'Market Research & Analysis',
                'feedback': 'Highlight how you’ve used market insights to guide product positioning or outreach.',
                'example': 'Conducted competitive market analysis to reposition brand messaging, resulting in a 25% traffic uplift.'
            })
    
        if 'digital marketing' in jd_lower and 'digital marketing' not in cv_lower:
            suggestions.append({
                'title': 'Digital Marketing Strategy',
                'feedback': 'Mention experience running or contributing to online campaigns via SEO, ads, email, or social media.',
                'example': 'Launched integrated digital campaigns across Google Ads and Meta, generating 3X ROAS.'
            })
    
        if 'brand awareness' in jd_lower and 'brand awareness' not in cv_lower:
            suggestions.append({
                'title': 'Brand Awareness Building',
                'feedback': 'Show how you’ve contributed to increasing visibility or perception of a brand or product.',
                'example': 'Executed PR and influencer marketing campaigns that increased brand visibility by 60%.'
            })

    if field == "Science":
        if 'laboratory' in jd_lower and 'laboratory' not in cv_lower:
            suggestions.append({
                'title': 'Laboratory Experience',
                'feedback': 'Include any hands-on experience in laboratory settings, procedures, or safety protocols.',
                'example': 'Conducted chemical experiments in compliance with ISO lab standards and maintained detailed logs.'
            })
    
        if 'data analysis' in jd_lower and 'data analysis' not in cv_lower:
            suggestions.append({
                'title': 'Scientific Data Analysis',
                'feedback': 'Mention your skills in analyzing experimental or field data using scientific methods or tools.',
                'example': 'Used SPSS and R to analyze biodiversity data, leading to publishable findings on species distribution.'
            })
    
        if 'scientific research' in jd_lower and 'scientific research' not in cv_lower:
            suggestions.append({
                'title': 'Scientific Research',
                'feedback': 'Demonstrate involvement in formal research studies, publications, or investigations.',
                'example': 'Led a two-year study on antimicrobial resistance with results published in a peer-reviewed journal.'
            })
    
        if 'report writing' in jd_lower and 'report writing' not in cv_lower:
            suggestions.append({
                'title': 'Scientific Report Writing',
                'feedback': 'Show ability to draft research papers, lab reports, or technical documents for scientific audiences.',
                'example': 'Authored detailed reports on water quality assessments submitted to regulatory bodies.'
            })
    
        if 'microscopy' in jd_lower and 'microscopy' not in cv_lower:
            suggestions.append({
                'title': 'Microscopy & Imaging',
                'feedback': 'Highlight experience with microscopes (optical, electron, etc.) or image analysis tools.',
                'example': 'Used SEM and fluorescence microscopy to study cell structures and capture high-resolution images.'
            })
    
        if 'fieldwork' in jd_lower and 'fieldwork' not in cv_lower:
            suggestions.append({
                'title': 'Scientific Fieldwork',
                'feedback': 'Include experiences collecting samples or conducting experiments in real-world environments.',
                'example': 'Carried out geological surveys and sediment sampling across multiple coastal sites.'
            })
    
        if 'spectrometry' in jd_lower and 'spectrometry' not in cv_lower:
            suggestions.append({
                'title': 'Spectrometry Techniques',
                'feedback': 'Mention your use of spectroscopy (e.g., UV-Vis, IR, Mass Spec) in research or analysis.',
                'example': 'Performed GC-MS analysis to determine pollutant levels in soil and water samples.'
            })
    
        if 'research grant' in jd_lower and 'grant' not in cv_lower:
            suggestions.append({
                'title': 'Grant Writing & Funding',
                'feedback': 'Highlight experience with securing or contributing to scientific grants and research funding.',
                'example': 'Secured a $50,000 research grant from TETFund to study disease resistance in crops.'
            })
    
        if 'publication' in jd_lower and 'publication' not in cv_lower:
            suggestions.append({
                'title': 'Scientific Publications',
                'feedback': 'List any peer-reviewed articles, journals, or research papers you’ve contributed to.',
                'example': 'Co-authored 3 papers in international journals on climate variability in Sub-Saharan Africa.'
            })
    
        if 'scientific software' in jd_lower and not any(term in cv_lower for term in ['matlab', 'spss', 'r ', 'stata']):
            suggestions.append({
                'title': 'Scientific Tools & Software',
                'feedback': 'Mention software relevant to your scientific domain such as MATLAB, R, SPSS, or Stata.',
                'example': 'Analyzed molecular dynamics using MATLAB and visualized results using custom scripts.'
            })

    if field == "Security / Intelligence":
        if 'threat analysis' in jd_lower and 'threat analysis' not in cv_lower:
            suggestions.append({
                'title': 'Threat Analysis',
                'feedback': 'Showcase your ability to assess security threats and develop countermeasures.',
                'example': 'Performed threat analysis on sensitive company operations, reducing potential breaches by 60%.'
            })
    
        if 'surveillance' in jd_lower and 'surveillance' not in cv_lower:
            suggestions.append({
                'title': 'Surveillance Operations',
                'feedback': 'Include any experience monitoring environments through CCTV or physical patrols.',
                'example': 'Monitored restricted access zones using CCTV systems and performed daily security audits.'
            })
    
        if 'access control' in jd_lower and 'access control' not in cv_lower:
            suggestions.append({
                'title': 'Access Control',
                'feedback': 'Mention your handling of personnel or system access — physical or digital.',
                'example': 'Managed access control using biometric systems for over 200 employees.'
            })
    
        if 'incident response' in jd_lower and 'incident response' not in cv_lower:
            suggestions.append({
                'title': 'Incident Response',
                'feedback': 'Highlight how you respond to or report security incidents, breaches, or suspicious activity.',
                'example': 'Led rapid response to security breach resulting in zero data loss and complete system recovery within 2 hours.'
            })
    
        if 'intel gathering' in jd_lower or 'intelligence gathering' in jd_lower and 'intelligence' not in cv_lower:
            suggestions.append({
                'title': 'Intelligence Gathering',
                'feedback': 'Detail any activities involving information gathering, analysis, or reporting on threats.',
                'example': 'Compiled and analyzed regional threat intelligence for use in proactive security planning.'
            })
    
        if 'counter-terrorism' in jd_lower and 'counter-terrorism' not in cv_lower:
            suggestions.append({
                'title': 'Counter-Terrorism Strategies',
                'feedback': 'Include experience working with or developing measures against terrorism or insurgent threats.',
                'example': 'Worked with local agencies to implement counter-terrorism training protocols for staff.'
            })
    
        if 'emergency response' in jd_lower and 'emergency response' not in cv_lower:
            suggestions.append({
                'title': 'Emergency Response Readiness',
                'feedback': 'Mention drills, trainings, or real incidents where you managed emergency protocols.',
                'example': 'Coordinated fire evacuation drills and emergency response during a facility lockdown.'
            })
    
        if 'security audit' in jd_lower and 'security audit' not in cv_lower:
            suggestions.append({
                'title': 'Security Audits & Assessments',
                'feedback': 'Indicate any role in evaluating and strengthening security posture.',
                'example': 'Conducted quarterly security audits and implemented changes that improved compliance scores by 40%.'
            })
    
        if 'law enforcement' in jd_lower and not any(x in cv_lower for x in ['police', 'security officer', 'law enforcement']):
            suggestions.append({
                'title': 'Law Enforcement Background',
                'feedback': 'Mention any law enforcement training, partnerships, or experience.',
                'example': 'Worked closely with local police in apprehending trespassers and maintaining perimeter control.'
            })
    
        if 'confidentiality' in jd_lower and 'confidentiality' not in cv_lower:
            suggestions.append({
                'title': 'Confidentiality & Information Handling',
                'feedback': 'Reinforce your commitment to handling sensitive or classified data securely.',
                'example': 'Handled sensitive intelligence files under strict access and confidentiality protocols.'
            })
            
    if field == "Travels & Tours":
        if 'itinerary planning' in jd_lower and 'itinerary' not in cv_lower:
            suggestions.append({
                'title': 'Itinerary Planning',
                'feedback': 'Mention your experience creating travel plans or schedules for clients or groups.',
                'example': 'Designed detailed travel itineraries covering flights, transfers, lodging, and excursions across 5 countries.'
            })
    
        if 'visa processing' in jd_lower and 'visa' not in cv_lower:
            suggestions.append({
                'title': 'Visa Assistance & Documentation',
                'feedback': 'Highlight any experience with visa application, embassy liaison, or travel documentation.',
                'example': 'Processed over 300 visa applications for clients, ensuring 98% success rate.'
            })
    
        if 'tour guiding' in jd_lower and 'tour guide' not in cv_lower:
            suggestions.append({
                'title': 'Tour Guide Experience',
                'feedback': 'Include any roles where you guided groups, explained attractions, or coordinated tours.',
                'example': 'Led daily walking tours for up to 25 tourists across historical sites with 4.9-star average feedback.'
            })
    
        if 'flight booking' in jd_lower and not any(x in cv_lower for x in ['flight booking', 'ticketing']):
            suggestions.append({
                'title': 'Flight Booking & Ticketing',
                'feedback': 'Mention your proficiency with booking tools or airline systems.',
                'example': 'Booked domestic and international flights using Amadeus GDS system for over 200 clients monthly.'
            })
    
        if 'travel agency' in jd_lower and 'travel agency' not in cv_lower:
            suggestions.append({
                'title': 'Travel Agency Operations',
                'feedback': 'Indicate experience managing or working within a travel agency setup.',
                'example': 'Coordinated agency logistics, partnered with airlines, and handled travel bookings end-to-end.'
            })
    
        if 'customer satisfaction' in jd_lower and 'customer satisfaction' not in cv_lower:
            suggestions.append({
                'title': 'Client Experience & Satisfaction',
                'feedback': 'Include achievements or metrics showing how you enhanced client travel experiences.',
                'example': 'Achieved 95% repeat customer rate by providing personalized travel solutions and real-time support.'
            })
    
        if 'hotel booking' in jd_lower and 'hotel' not in cv_lower:
            suggestions.append({
                'title': 'Hotel Reservations',
                'feedback': 'Highlight experience booking accommodation, negotiating rates, or handling cancellations.',
                'example': 'Managed hotel bookings across 3 continents, negotiating group discounts and resolving reservation issues.'
            })
    
        if 'tour packages' in jd_lower and 'tour packages' not in cv_lower:
            suggestions.append({
                'title': 'Tour Package Design',
                'feedback': 'Mention any creative or logistical input you had in creating travel packages.',
                'example': 'Developed honeymoon and adventure tour packages that boosted sales by 30% in one year.'
            })
    
        if 'international travel' in jd_lower and 'international' not in cv_lower:
            suggestions.append({
                'title': 'International Travel Coordination',
                'feedback': 'Show familiarity with global travel routes, regulations, and client needs.',
                'example': 'Coordinated international group travel for conferences, ensuring smooth cross-border logistics.'
            })
    
        if 'travel insurance' in jd_lower and 'insurance' not in cv_lower:
            suggestions.append({
                'title': 'Travel Insurance Advisory',
                'feedback': 'Mention helping clients understand and choose travel insurance plans.',
                'example': 'Guided clients through travel insurance options, reducing claims processing delays.'
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

        cv_text = extract_text(file)

        cv_embedding = embed_text(cv_text)
        jd_embedding = embed_text(jd_text)
        similarity = util.cos_sim(cv_embedding, jd_embedding).item()

        suggestions = generate_suggestions(cv_text, jd_text, field)

        html_result = f"""
            <h2>Similarity Score: {similarity:.2f}</h2>
            <h3>Smart Suggestions</h3>
            <ul>
        """
        for s in suggestions:
            html_result += f"<li><strong>{s['title']}</strong>: {s['feedback']}<br><em>e.g., {s['example']}</em></li>"
        html_result += "</ul>"
        return html_result

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
            </select><br><br>

            <input type="submit" value="Check CV">
        </form>
    ''')

# === RUN ===
if __name__ == '__main__':
    app.run(debug=True, port=8080)
