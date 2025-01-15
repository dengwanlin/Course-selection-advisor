from flask import Flask, render_template, request, redirect, url_for,session
#from pymongo import MongoClient
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from db.connection import connect_to_cluster, fetch_data
from model.cbf import get_course_recommendations, get_similarity_resources

#from mongodb.password import password

app = Flask(__name__)
app.secret_key = 'my_secret_key'

client = connect_to_cluster()
courses_data = fetch_data('processed_courses')
print(courses_data)


db = client['Course_Recommendation']
student_collection = db['student']
enums_collection = db['enums']



@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        password = request.form.get('password')

        # Get the current time as the registration time
        register_time = datetime.now()

        # Check if the student ID already exists
        existing_student = student_collection.find_one({'student_id': student_id})
        if existing_student:
            message = "The student number has been registered, please enter <a href='{}'>Login Page</a>to Log in, or you can re-fill in your information to register.".format(url_for('login'))
            return render_template('register.html', exist_message=message)
        else:
            student_info = {
                "student_id": student_id,
                "first_name": first_name,
                "last_name": last_name,
                "password": password,
                "register_time": register_time
            }
            student_collection.insert_one(student_info)
            return render_template('register_success.html', login_url=url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        password = request.form.get('password')

        # Check if the account exists in the database
        student = student_collection.find_one({'student_id': student_id})
        if student:
            if student['password'] == password:
                session['student_id'] = student_id  # Add this line to store the student number in the session
                session['username'] = student['first_name']  # Store first_name in session
                return redirect(url_for('home', username=student['first_name']))
            else:
                error = "The password and account do not match!"
        else:

            error = "The account does not exist, please enter <a href='{}'>register page</a> to register, or you can re-fill in your information to log in.".format(
                url_for('register'))
    return render_template('login.html', error=error)


@app.route('/home/<username>')
def home(username):
    return render_template('home.html', username=username)


@app.route('/logout')
def logout():
    return redirect(url_for('welcome'))


@app.route('/favicon.ico')
def favicon():
    return '', 204

# Add a new route to display the questionnaire page
@app.route('/questionnaire')
def questionnaire():
    enums_data = enums_collection.find_one({})
    terms = enums_data.get('Student_Term', [])
    languages = enums_data.get('Student_Languge', [])
    language_levels = enums_data.get('Student_Language_Level', [])
    print("In questionnaire route - language_levels type:", type(language_levels))
    if isinstance(language_levels, list):
        for item in language_levels:
            print("In questionnaire route - item type:", type(item))
    return render_template('questionnaire.html', terms=terms, languages=languages, language_levels=language_levels)


# Added a new route to handle questionnaire submission logic
@app.route('/submit_status', methods=['POST'])
def submit_status():
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('login'))

    semester = request.form.get('semester')
    # Get the raw language level data (originally a list of strings)
    enums_data = enums_collection.find_one({})
    languages = enums_data.get('Student_Languge', [])
    language_levels_data_str_list = enums_data.get('Student_Language_Level', [])

    # Convert a list of strings to a list of dictionaries (a simple example, assuming the format is {"Level": original string value})
    language_levels_data = [{"Level": item} for item in language_levels_data_str_list]

    language_levels = []
    for language in languages:
        level = request.form.get(f"{language}_level")
        for lang_level_dict in language_levels_data:

            if "Level" in lang_level_dict:
                lang_level_dict["Language"] = language
                lang_level_dict["Level"] = level
                language_levels.append(lang_level_dict)
                break

    major_name = request.form.get('major_name')
    direction_name = request.form.get('direction_name')

    student_doc = student_collection.find_one({'student_id': student_id})
    if student_doc:
        student_doc['Student_Language_Level'] = language_levels
        student_doc['Student_Major'] = {
            "Major_Name": major_name,
            "Direction_Name": direction_name
        }
        student_doc['Semester'] = semester

        try:
            result = student_collection.update_one({'student_id': student_id}, {'$set': student_doc})
            print(f"Database update result: {result.modified_count} records were modified")
        except Exception as e:
            print(f"Database update error: {e}")
    else:
        print(f"No student document found for student ID {student_id}")

    return redirect(url_for('home', username=session.get('username')))

@app.route('/course')
def show_course():
    pass

@app.route('/recommendation')
def show_recommendation():
    
#similarity_resources
    # dictionary, tfidf, termsim_matrix, tfidf_corpus, courses_data = get_similarity_resources()

    pass

@app.route('/more')
def show_more():
    pass



if __name__ == '__main__':
    app.run(debug=True)