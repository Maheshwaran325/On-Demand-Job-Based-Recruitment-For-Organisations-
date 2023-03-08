import sqlite3
from werkzeug.utils import secure_filename
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
from flaskr.db import get_db
from pyresparser import ResumeParser
import os
import sqlite3
from flask import send_from_directory
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64


bp = Blueprint('blog', __name__)  


@bp.route('/', methods=['POST', 'GET'])
def hr():
 return render_template('dashboard/hrdashboard.html')


@bp.route('/upload', methods=['POST', 'GET'])
def upload():
 if request.method == 'POST':
    files = request.files.getlist('files')
    if not files:
        return 'No files selected'
    filenames = []
    UPLOAD_FOLDER = os.path.join(bp.root_path, 'uploads')
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS resumes
             (file_name text, name text, email text, mobile text, skill text,  
             college text)''')               #, , skills text, degree text
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))
            filenames.append(file.filename)
            resume_parser = ResumeParser(os.path.join(UPLOAD_FOLDER, file.filename))
            resume_data = resume_parser.get_extracted_data()
            skills_string = ','.join(resume_data['skills'])
            c.execute("INSERT INTO resumes VALUES (?,?,?,?,?,?)", (file.filename,resume_data['name'],resume_data['email'],resume_data['mobile_number'], 
            skills_string,resume_data['college_name']))
            #resume_data['skills'], resume_data['college_name'], 
    conn.commit()
    conn.close()
    # flash("Uploaded Successfully!")
    return redirect('/parsed_data')
 return render_template('dashboard/upload_resume.html')

@bp.route('/parsed_data')
def parsed_data():
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute("SELECT * FROM resumes")
    data = c.fetchall()
    conn.close()
    return render_template('dashboard/resume_info.html', data=data)

@bp.route('/upload/<filename>')
def uploaded_file(filename):
    UPLOAD_FOLDER = os.path.join(bp.root_path, 'uploads')
    return send_from_directory(UPLOAD_FOLDER, filename)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#------------------------------------------------------------------------------------------------------------

# Load the saved Random Forest model
@bp.route("/predict", methods=["POST", 'GET'])
def predict():
    model = joblib.load("flaskr/model.pkl") 
    encoder = joblib.load("flaskr/encoder.pkl")
    
    if request.method == 'POST':
        file = request.files["file"]
        input_data = pd.read_csv(file)

        # Store Department, Gender, JobRole, MonthlyIncome in SQLite database
        conn = sqlite3.connect("results.db")
        cursor = conn.cursor()

        # Create a new table       
        cursor.execute("DROP TABLE IF EXISTS employee")
        cursor.execute('''CREATE TABLE IF NOT EXISTS employee (
                          EmployeeID INTEGER ,
                          Department TEXT,
                          Gender TEXT,
                          JobRole TEXT,
                          MonthlyIncome INTEGER
                          )''')

        # Insert the data into the table
        for index, row in input_data.iterrows():
            employee_id = row['EmployeeID']
            department = row["Department"]
            gender = row['Gender']
            job_role = row['JobRole']
            monthly_income = row['MonthlyIncome']

            cursor.execute("INSERT INTO employee (EmployeeID, Department, Gender, JobRole, MonthlyIncome) VALUES (?, ?, ?, ?, ?)", 
                           (employee_id, department, gender, job_role, monthly_income))
            conn.commit()
        conn.close()

        # Save the EmployeeID for later use
        employee_ids = input_data["EmployeeID"]

        # Drop the EmployeeID column
        input_data = input_data.drop("EmployeeID", axis=1)

        # Encode the categorical data
        X_encoded = pd.DataFrame(encoder.transform(input_data[input_data.columns[input_data.dtypes == 'object']]).toarray())

        # Convert feature names to strings
        X_encoded.columns = X_encoded.columns.astype(str)

        # Drop the original categorical columns
        input_data = input_data.drop(input_data.columns[input_data.dtypes == 'object'], axis=1)

        # Concatenate the encoded categorical columns with the rest of the data
        input_data = pd.concat([input_data, X_encoded], axis=1)

        input_data.columns = [col.strip() for col in input_data.columns]


        # Use the model to make predictions
        result = model.predict(input_data)

        conn = sqlite3.connect("results.db")
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS results")
        cursor.execute('''CREATE TABLE results (
                        EmployeeID TEXT,
                        result TEXT
                        )''')
        for i in range(len(employee_ids)):
            employee_id = employee_ids[i]
            prediction = result[i]

            cursor.execute("INSERT INTO results (EmployeeID, result) VALUES (?, ?)", 
                        (str(employee_id), prediction))      
        #cursor.executemany("INSERT INTO results (result) VALUES (?)", [(x,) for x in result])
        # cursor.execute("INSERT INTO results (result) VALUES (?)",(result[0][0]))

        conn.commit()
        conn.close()
        # Return the result to the template
        return redirect("/results")
    return render_template('dashboard/employeeattrition.html')


@bp.route("/results", methods=["GET"])
def results():
    conn = sqlite3.connect("results.db")
    cursor = conn.cursor()

    # Fetch department-wise attrition counts from the database
    cursor.execute("""SELECT e.Department, r.result, COUNT(*) AS count 
                      FROM employee e 
                      JOIN results r ON e.EmployeeID = r.EmployeeID 
                      WHERE r.result IN ('Yes', 'No') 
                      GROUP BY e.Department, r.result""")
    result = cursor.fetchall()
    
    df = pd.DataFrame(result, columns=["Department", "Attrition", "Count"])
    df['Count'] = pd.to_numeric(df['Count'])

    # Create a grouped bar chart showing department-wise attrition counts
    plt.figure(figsize=(8, 6))
    ax1 = df.pivot(index="Department", columns="Attrition", values="Count").plot(kind="bar", stacked=True)
    ax1.set_title("Attrition Result by Department")
    ax1.set_xlabel("Department")
    ax1.set_ylabel("Count")
    plt.tight_layout()

    # Convert plots to images and display on web page
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format="png")
    buffer1.seek(0)
    plot_data1 = base64.b64encode(buffer1.read()).decode()
    plt.close()

    # Fetch job role-wise attrition counts from the database
    cursor.execute("""SELECT e.JobRole, r.result, COUNT(*) AS count 
                      FROM employee e 
                      JOIN results r ON e.EmployeeID = r.EmployeeID 
                      WHERE r.result IN ('Yes', 'No') 
                      GROUP BY e.JobRole, r.result""")
    result = cursor.fetchall()

    df = pd.DataFrame(result, columns=["JobRole", "Attrition", "Count"])
    df['Count'] = pd.to_numeric(df['Count'])

    # Create a grouped bar chart showing job role-wise attrition counts
    plt.figure(figsize=(8, 6))
    ax2 = df.pivot(index="JobRole", columns="Attrition", values="Count").plot(kind="bar", stacked=True)
    ax2.set_title("Attrition Result by Job Role")
    ax2.set_xlabel("Job Role")
    ax2.set_ylabel("Count")
    plt.tight_layout()

    # Convert plots to images and display on web page
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format="png")
    buffer2.seek(0)
    plot_data2 = base64.b64encode(buffer2.read()).decode()
    plt.close()

    # Fetch monthly income-wise attrition counts from the database
    cursor.execute("""SELECT e.MonthlyIncome, r.result, COUNT(*) AS count 
                    FROM employee e 
                    JOIN results r ON e.EmployeeID = r.EmployeeID 
                    WHERE r.result IN ('Yes', 'No') 
                    GROUP BY e.MonthlyIncome, r.result""")
    result = cursor.fetchall()

    df_monthlyincome = pd.DataFrame(result, columns=["MonthlyIncome", "Attrition", "Count"])
    df_monthlyincome['Count'] = pd.to_numeric(df_monthlyincome['Count'])

    # Split MonthlyIncome into bins of 20000
    bins = pd.IntervalIndex.from_tuples([(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10000), (10000, 20000), ])
    df_monthlyincome['MonthlyIncomeRange'] = pd.cut(df_monthlyincome['MonthlyIncome'], bins=bins, include_lowest=True)

    # Create a bar chart showing monthly income-wise attrition counts
    plt.figure(figsize=(8, 6))
    ax3 = df_monthlyincome.pivot_table(index="MonthlyIncomeRange", columns="Attrition", values="Count", aggfunc=np.sum).plot(kind="bar", stacked=True)
    ax3.set_title("Attrition Result by Monthly Income")
    ax3.set_xlabel("Monthly Income Range")
    ax3.set_ylabel("Count")
    plt.tight_layout()

    # Convert plots to images and display on web page
    buffer3 = io.BytesIO()
    plt.savefig(buffer3, format="png")
    buffer3.seek(0)
    plot_data3 = base64.b64encode(buffer3.read()).decode()
    plt.close()
    
    # Fetch overall attrition count from the database
    cursor.execute("""SELECT COUNT(*) FROM employee""")
    total_count = cursor.fetchone()[0]

    cursor.execute("""SELECT COUNT(*)
                      FROM results
                      WHERE result = 'Yes'""")
    attrition_count = cursor.fetchone()[0]

    overall_attrition_rate = attrition_count / total_count
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot the bar chart
    bars = ax.bar(["Yes", "No"], [attrition_count, total_count - attrition_count], color=['tab:orange', 'tab:blue'])

    # Add count as text labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, int(height), ha='center', va='bottom')

    # Set chart title, x-axis label, and y-axis label
    ax.set_title("Overall Attrition Rate")
    ax.set_xlabel("Attrition")
    ax.set_ylabel("Count")

    # Save the chart as a PNG image in memory buffer
    buffer4 = io.BytesIO()
    plt.savefig(buffer4, format="png")
    buffer4.seek(0)
    plot_data4 = base64.b64encode(buffer4.read()).decode()
    plt.close()
    
    return render_template("dashboard/result.html", plot_data1=plot_data1, plot_data2= plot_data2, plot_data3= plot_data3, plot_data4= plot_data4)

