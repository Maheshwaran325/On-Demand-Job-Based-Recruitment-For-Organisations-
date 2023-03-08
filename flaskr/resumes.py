from flask import Blueprint, render_template, request

resumes = Blueprint('resumes', __name__)

@resumes.route("/", methods=["GET"])
def get_form():
    return render_template("dashboard/upload_employee_details_csv.html")

@resumes.route("/", methods=["POST"])
def upload_resumes():
    # code for handling the file upload and parsing goes here


