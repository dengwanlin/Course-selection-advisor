import bcrypt
from flask import Flask, render_template, request, redirect, url_for,session
from datetime import datetime
from cbf import *
from dash import dcc, html, Dash, Input, Output, callback


app = Flask(__name__)
app.secret_key = 'my_secret_key'


# Create the Dash app
dash_app = Dash(
    __name__,
    server=app,
    url_base_pathname='/dash/' , # Mounts the Dash app at '/dash/'
    assets_folder='static',
)

# Create the Dash app
# dash_dashboard_insights = Dash(
#     # __other__,
#     server=app,
#     url_base_pathname='/dash/' , # Mounts the Dash app at '/dash/'
#     assets_folder='static'
# )

@app.route('/', methods=['GET', 'POST'])
#@app.route('/welcome', methods=['GET', 'POST'])
def welcome():
    error = None
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        password = request.form.get('password')

        # Check if the account exists in the database
        student = fetch_single_data('students', {'student_id': student_id})
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
    return render_template('welcome.html', error=error)


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
        existing_student = fetch_single_data('students', {'student_id': student_id})
        if existing_student:
            message = "The student number has been registered, please enter <a href='{}'>welcome Page</a>to Log in, or you can re-fill in your information to register.".format(url_for('welcome'))
            return render_template('register.html', exist_message=message)
        else:
            student_info = {
                "student_id": student_id,
                "first_name": first_name,
                "last_name": last_name,
                "password": password,
                "register_time": register_time
            }
            insert_data('students', student_info)
            return render_template('register_success.html', welcome_url=url_for('welcome'))
    return render_template('register.html')


@app.route('/home/<username>')
def home(username):
    if 'student_id' not in session:
        return redirect(url_for('welcome'))
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
    enums_data = fetch_single_data('enums', {})

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
                           student_math_levels=student_math_levels,
                           username=session.get('username')
                           )

# Added a new route to handle questionnaire submission logic
@app.route('/submit_status', methods=['POST'])
def submit_status():
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))

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


    student_doc = fetch_single_data('students', {'student_id': student_id})

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
            result = update_one_data('students', {'student_id': student_id}, {'$set': student_doc})
            print(f"Database update result: {result.modified_count} records were modified")
        except Exception as e:
            print(f"Database update error: {e}")
    else:
        print(f"No student document found for student ID {student_id}")

    return redirect(url_for('course', username=session.get('username')))

@app.route('/course')
def course():
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))
    
    courses = fetch_local_data('processed_courses')
    collection_name = 'students'
    student_courses = fetch_single_data(collection_name, {"student_id": student_id})

    return render_template('questionnaire.html', 
                           courses=courses,
                           student_courses=student_courses,
                           username=session.get('username')
                           )

# Added a new route to handle questionnaire submission logic
@app.route('/course', methods=['POST'])
def get_recommendation():
    courses = fetch_local_data('courses')
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))
    
    collection_name = 'students'
    student_courses = fetch_single_data(collection_name, {"student_id": student_id})
    
    user_input = {
    "preferred_language": request.form.get('language'),
    "math_level": request.form.get('teaching_style'),
    "keywords": request.form.get('keywords'),
    "module": request.form.get('module'),
    "teaching_style": request.form.get('teaching_style'),
    "weighting": {"textual": 0.7, "categorical": 0.3},
    }

    recommendations =  get_course_recommendations(user_input)

    return render_template('questionnaire.html', recommendations=recommendations, courses=courses, username=session.get('username'), available_recommendations=True, student_courses=student_courses)



@app.route('/course/<course_id>/<course_name>/<type>', methods=['GET'])
def add_course(course_id, course_name, type):
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))
    
    collection_name = 'students'
    courses = fetch_local_data('courses')
    


    if type == 'add':
        check_course = fetch_single_data(collection_name, {"my_courses.id": course_id, "student_id": student_id, })
        student_courses = fetch_single_data(collection_name, {"student_id": student_id})
        if check_course is None:
            update_one_data(
                        collection_name,
                        {"student_id": student_id},
                        {"$push": {"my_courses": {"name": course_name, "id": course_id}}}
                    )
            student_courses = fetch_single_data(collection_name, {"student_id": student_id})
            return render_template('questionnaire.html', username=session.get('username'), student_courses=student_courses, courses=courses) #, available_recommendations=True
        else:
            return render_template('questionnaire.html', username=session.get('username'), error='course_add_clash', student_courses=student_courses, courses=courses)
        
    elif type == 'remove':
        update_one_data(
                        collection_name,
                        {"student_id": student_id},
                        {"$pull": {"my_courses": {"id": course_id}}}
                    )
        
        student_courses = fetch_single_data(collection_name, {"student_id": student_id})
        return render_template('questionnaire.html', username=session.get('username'), student_courses=student_courses, courses=courses) #, available_recommendations=True



@app.route('/course', methods=['GET'])
def get_my_courses():
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))
    
    courses = fetch_local_data('courses')
    collection_name = 'students'
    student_courses = fetch_single_data(collection_name, {"student_id": student_id})
       
    return render_template('questionnaire.html', username=session.get('username'), student_courses=student_courses, courses=courses, active_tab='myCoursesTab')



@app.route('/get_single_course/<course_id>', methods=['GET'])
def get_single_course(course_id):
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))
     
    course = fetch_single_data('courses', {"id": course_id})
    # if not course:
    #     return render_template('404.html', message="Course not found"), 404

    # Render the template with course details
    return render_template('coursedescription.html', course=course, username=session.get('username'),)



@app.route('/insights')
def insights():
        
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))
    
    return render_template('insights.html', username=session.get('username'))

@app.route('/setting')
def setting():
    return render_template('setting.html', username=session.get('username'))

@app.route('/dash/')
def dash_index():
    # Redirect directly to Dash if needed
    return redirect('/dash/')


@app.route('/change_password', methods=['POST'])
def change_password():
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))

    current_password = request.form.get('current_password')

    new_password = request.form.get('new_password')
    confirm_new_password = request.form.get('confirm_new_password')

    student = fetch_single_data('students', {'student_id': student_id})
    if not student:
        return redirect(url_for('welcome'))

    if student['password']!= current_password:
        return render_template('setting.html', error='Current password is incorrect.')

    if new_password!= confirm_new_password:
        return render_template('setting.html', error='New passwords do not match.')

    try:
        update_one_data('students', {'student_id': student_id}, {'$set': {'password': confirm_new_password}})
        session.pop('student_id', None)
        session.pop('username', None)
        return redirect(url_for('welcome'))
    except Exception as e:
        return render_template('setting.html', error=f'Password change failed: {str(e)}')

# def dash_application():
# , 'display':'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '5px'
dash_app.layout = dcc.Loading(
    type="circle", 
    children=html.Div(style={'height': '98vh' }, children=[
     html.Div(
           
            style={
                'display': 'flex',
                'flexDirection':'column'
            }
            , children= [
                html.Div(style={'display':'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '5px'},
                         children=[dcc.Dropdown(['Course Scatter','Module', 'Languages', 'Lecturers'], 'Course Scatter', id='pie_type')]),
                
                html.Div( 
                dcc.Graph(
                id="pie-output",
                ),
            )
            ]
        ),
    #  html.Div(
    #         dcc.Graph(
    #             id="scatter-graph",
    #             figure=courses_scatter()
    #         ),
    #            style={
    #             'border': '1px solid gray'
    #         }
    #     ),
    ]) 
)
  
#   return dash_graph

@callback(
    Output(component_id='pie-output', component_property='figure'),
    Input(component_id='pie_type', component_property='value')
  )
def update_output_div(pie_type):
    if pie_type=='Course Scatter':
        return courses_scatter()
    elif pie_type=='Module': 
        return course_modules()
    elif pie_type=='Lecturers': 
        return course_lecturer()
    elif pie_type=='Languages': 
        return course_languages()


# dash_application()


if __name__ == '__main__':
    app.run(debug=True)

