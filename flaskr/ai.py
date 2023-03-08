import sqlite3
from werkzeug.utils import secure_filename
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
from flaskr.db import get_db
import os
from flask import send_from_directory
import spacy
nlp = spacy.load("en_core_web_sm")


ai = Blueprint('ai', __name__)  


@ai.route('/match')
def index():
    # Fetch all job roles from the database
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS job_roles")
    c.execute('''CREATE TABLE IF NOT EXISTS job_roles (
                job_name TEXT,
                skills TEXT
                )''')
     
      # Define job roles and skills
    job_roles = {
        'Software Developer': ['Python','Java','C++','JavaScript'],
        'Data Scientist': ['Python','R','SQL','Machine Learning'],
        'Marketer': ['Coding','Marketing','Digital marketing','Analytics','Content'],
        'Web development': ['Html','Css','C','Marketing','Java','Programming'],
        'HR': ['Communication','Data analysis' ]
    }

 # Insert job roles and skills into the table
    for job_name, skills in job_roles.items():
        skills_str = ', '.join(skills)
        c.execute("INSERT INTO job_roles (job_name, skills) VALUES (?, ?)", (job_name, skills_str))
        conn.commit()

    conn.close()

    return render_template('dashboard/match.html', job_roles=job_roles)

@ai.route('/matchresult', methods=['POST'])
def match_candidates():
    # Extract job role from the form
    job_role = request.form.get('job_role')

    # Fetch required skills for the job role
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute("SELECT skills FROM job_roles WHERE job_name=?", (job_role,))
    required_skills = c.fetchone()[0].split(", ")
    conn.close()

    # Fetch candidates matching the job role from the database
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    query = "SELECT name, mobile, email, skill FROM resumes WHERE "
    conditions = ["skill LIKE '%{}%'".format(skill) for skill in required_skills]
    query += " OR ".join(conditions)
    c.execute(query)
    candidates = c.fetchall()
    conn.close()

    # Calculate the percentage match for each candidate
    matches = []
    for candidate in candidates:
        candidate_skills = [skill.strip() for skill in candidate[3].split(",")]

        # Use spaCy to extract keywords and phrases from the candidate's resume
        doc = nlp(candidate[3])
        candidate_keywords = set([chunk.text for chunk in doc.noun_chunks])
        candidate_keywords.update(set([token.lemma_ for token in doc if not token.is_stop]))

        # Use set intersection to find the skills that both the candidate and job require
        matched_skills = set(candidate_skills).intersection(set(required_skills))
        matched_keywords = set(candidate_keywords).intersection(set(required_skills))

        # Calculate the match percentage based on matched skills and keywords
        match_count = 0
        for skill in required_skills:
            if skill in candidate_skills:
                match_count += 1
            elif skill in candidate_keywords:
                match_count += 1
        if len(required_skills) > 0:
            match_percentage = (match_count / len(required_skills)) * 100
        else:
            match_percentage = 0

        matches.append((candidate[0], candidate[1], candidate[2], round(match_percentage, 2)))

    # print(required_skills)
    # print(candidate_skills)
    # print(matched_skills)
    # print(matched_keywords)
    matches = sorted(matches, key=lambda x: x[3], reverse=True)

    return render_template('dashboard/airesult.html', matches=matches, job_role=job_role, candidates=candidates)
