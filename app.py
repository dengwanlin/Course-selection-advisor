import bcrypt
from flask import Flask, render_template, request, redirect, url_for,session
#from pymongo import MongoClient
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
#from mongodb.password import password

app = Flask(__name__)
app.secret_key = 'my_secret_key'

uri = "mongodb+srv://whhxsg:whhxsg@coursecluster.ecl2n.mongodb.net/?retryWrites=true&w=majority&appName=CourseCluster"
# Configure MongoDB connection (here it is assumed that MongoDB is running on the local default port)
#client = MongoClient('localhost', 27017)
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client['Course_Recommendation']
student_collection = db['student']
enums_collection = db['enums']

@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/register', methods=['GET', 'POST'])
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
    if 'student_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=username)


@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('student_id', None)
    return redirect(url_for('welcome'))


@app.route('/favicon.ico')
def favicon():
    return '', 204

# Add a new route to display the questionnaire page
@app.route('/questionnaire')
def questionnaire():
    enums_data = enums_collection.find_one({})

    student_professional_backgrounds = enums_data.get('Course_Recommended_Background', [])
    terms = enums_data.get('Student_Term', [])
    languages = enums_data.get('Student_Languge', [])
    language_levels = enums_data.get('Student_Language_Level', [])
    programming_languages = enums_data.get('Course_Required_Programming_Language', [])
    programming_levels = enums_data.get('Course_Required_Programming_Language_Level', [])
    student_majors = enums_data.get('Student_Major', [])
    for major in student_majors:
        if 'Direction_Name' not in major:
            major['Direction_Name'] = []
    student_math_levels = enums_data.get('Student_Math_Level', [])

    return render_template('questionnaire.html', terms=terms,
                           languages=languages,
                           language_levels=language_levels,
                           student_professional_backgrounds=student_professional_backgrounds,
                           programming_languages=programming_languages,
                           programming_levels=programming_levels,
                           student_majors = student_majors,
                           student_math_levels=student_math_levels)


# Added a new route to handle questionnaire submission logic
@app.route('/submit_status', methods=['POST'])
def submit_status():
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('login'))

    term = request.form.get('terms')
    selected_languages = request.form.getlist('languages')
    language_levels = []
    for language in selected_languages:
        level = request.form.get(f"{language}_level")
        language_levels.append({"Language": language, "Level": level})

    student_professional_background = request.form.getlist('student_professional_background')
    selected_programming_languages = request.form.getlist('programming_languages')

    programming_levels = {}
    for language in selected_programming_languages:
        level = request.form.get(f"{language}_level")
        programming_levels[language] = level

    major_name = request.form.get('major_name')
    direction_name = request.form.get('direction_name')
    student_math_background = request.form.get('student_math_background')
    number_courses_to_choose = request.form.get('number_courses_to_choose')
    available_exercise_time_per_week = request.form.get('available_exercise_time_per_week')


    student_doc = student_collection.find_one({'student_id': student_id})

    if student_doc:
        student_doc['Student_Language_Level'] = language_levels
        student_programming_level = []
        for language, level in zip(selected_programming_languages, programming_levels.values()):
            student_programming_level.append({"Language": language, "Level": level})
        student_doc['Student_Programming_Level'] = student_programming_level

        student_doc['Term'] = term
        student_doc['Student_Professional_Background'] = student_professional_background
        student_doc['Student_Major'] = {
            "Major_Name": major_name,
            "Direction_Name": direction_name
        }
        student_doc['Student_Math_Background'] = student_math_background
        student_doc['Number_Courses_To_Choose'] = int(number_courses_to_choose) if number_courses_to_choose else 0
        student_doc['Available_Exercise_Time_Per_Week'] = int(available_exercise_time_per_week) if available_exercise_time_per_week else 0
        try:
            result = student_collection.update_one({'student_id': student_id}, {'$set': student_doc})
            print(f"Database update result: {result.modified_count} records were modified")
        except Exception as e:
            print(f"Database update error: {e}")
    else:
        print(f"No student document found for student ID {student_id}")

    return redirect(url_for('home', username=session.get('username')))

@app.route('/course')
def course():
    return render_template('course.html', username=session.get('username'))

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html', username=session.get('username'))

@app.route('/more')
def more():
    return render_template('more.html', username=session.get('username'))


@app.route('/change_password', methods=['POST'])
def change_password():
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('login'))

    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_new_password = request.form.get('confirm_new_password')

    student = student_collection.find_one({'student_id': student_id})
    if not student:
        return redirect(url_for('login'))

    if student['password']!= current_password:
        return render_template('more.html', error='Current password is incorrect.')

    if new_password!= confirm_new_password:
        return render_template('more.html', error='New passwords do not match.')

    try:
        #hashed_password = bcrypt.hashpw(new_password.encode('utf - 8'), bcrypt.gensalt())
        student_collection.update_one({'student_id': student_id}, {'$set': {'password': confirm_new_password}})
        session.pop('student_id', None)
        session.pop('username', None)
        return redirect(url_for('welcome'))
    except Exception as e:
        return render_template('more.html', error=f'Password change failed: {str(e)}')
if __name__ == '__main__':
    app.run(debug=True)
