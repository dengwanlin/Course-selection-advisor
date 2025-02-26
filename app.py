import bcrypt
from flask import Flask, render_template, request, redirect, url_for,session, jsonify
from datetime import datetime
from utils import *
import ast
import json
from dash import dcc, html, Dash, Input, Output
from bson import json_util


app = Flask(__name__)
app.secret_key = 'my_secret_key'

# Create the Dash apps
dash_app = Dash(
    __name__,
    server=app,
    url_base_pathname='/dash/' , # Mounts the Dash app at '/dash/'
    assets_folder='static',
)

home_dash_app = Dash(
    __name__,
    server=app,
    url_base_pathname='/home_dash/' , # Mounts the Dash app at '/home_dash/'
    assets_folder='static',
)


student_professional_backgrounds, terms, languages, language_levels, programming_languages, programming_levels, student_majors, student_math_levels = get_enums_data()
def get_loggedin_student():
     student_id = session.get('student_id')
     student_data = fetch_single_data('students', {'student_id': student_id})

     return student_data


# Register the custom filter
app.jinja_env.filters['format_skills'] = format_skills
app.jinja_env.filters['format_math_level'] = format_math_level

@app.route('/', methods=['GET', 'POST'])
#@app.route('/welcome', methods=['GET', 'POST'])
def welcome():
    error = None
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        password = request.form.get('password')
        password = password.encode('utf-8') 

        # Check if the account exists in the database
        student = fetch_single_data('students', {'student_id': student_id})
        if student:
            stored_hash = student['password']
            result = bcrypt.checkpw(password, stored_hash.encode('utf-8')) 
            if result == True:
                session['student_id'] = student_id  # Add this line to store the student number in the session
                session['username'] = student['first_name']  # Store first_name in session
                return redirect(url_for('home', username=student['first_name']))
            else:
                error = "Invalid student id or password"
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
        password = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password, salt).decode('utf-8')
        
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
                "password": hashed_password,
                "register_time": register_time
            }
            insert_data('students', student_info)
        return redirect(url_for('welcome'))
    return render_template('register.html')

@app.route('/home/<username>')
def home(username):
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))
    
    courses = fetch_local_data('processed_courses')
    collection_name = 'students'
    student_courses = fetch_single_data(collection_name, {"student_id": student_id})
    student_json = json_util.dumps(student_courses)

    return render_template('home.html', username=username, courses=len(courses), my_courses=student_courses)

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('student_id', None)
    return redirect(url_for('welcome'))

@app.route('/favicon.ico')
def favicon():
    return '', 204

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
        student_doc['profile_updated'] = True
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
                           terms=terms,
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
@app.route('/course', methods=['POST'])
def get_recommendation():
    courses = fetch_local_data('processed_courses')
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

    if student_courses.get('profile_updated') is None or student_courses.get('profile_updated') == False:
        return render_template('questionnaire.html', username=session.get('username'), 
                           student_courses=student_courses, courses=courses, 
                           error='profile_not_updated', 
                           terms=terms,
                           languages=languages,
                           language_levels=language_levels,
                           student_professional_backgrounds=student_professional_backgrounds,
                           programming_languages=programming_languages,
                           programming_levels=programming_levels,
                           student_majors = student_majors,
                           student_math_levels=student_math_levels,)    
    # recommendations =  get_course_recommendations(user_input)


    recommendations = get_course_recommendations_2(user_input, student_courses)

    return render_template('questionnaire.html', 
                           recommendations=recommendations, 
                           courses=courses, username=session.get('username'), 
                           available_recommendations=True, 
                           student_courses=student_courses, 
                           terms=terms,
                           languages=languages,
                           language_levels=language_levels,
                           student_professional_backgrounds=student_professional_backgrounds,
                           programming_languages=programming_languages,
                           programming_levels=programming_levels,
                           student_majors = student_majors,
                           student_math_levels=student_math_levels,)

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
        return render_template('questionnaire.html', 
                               username=session.get('username'), 
                               student_courses=student_courses, courses=courses, terms=terms,
                               languages=languages,
                               language_levels=language_levels,
                               student_professional_backgrounds=student_professional_backgrounds,
                               programming_languages=programming_languages,
                               programming_levels=programming_levels,
                               student_majors = student_majors,
                               student_math_levels=student_math_levels,) #, available_recommendations=True


@app.route('/add_course_alt', methods=['POST'])
def add_course_alt():
    student_id = session.get('student_id')
    if student_id is None:
        return jsonify({"success": False, "error": "Not logged in"})
    
    data = request.json
    type = data.get('type')
    course = data.get('course')
    collection_name = 'students'
    
    if type == 'add':
        check_course = fetch_single_data(collection_name, {"my_courses.id": course['id'], "student_id": student_id})
        
        if check_course is None:
            # Create enhanced course object with similarity and explanation
            enhanced_course = course.copy()  # Copy the course data
            enhanced_course['added_date'] = datetime.now().isoformat()
            
            update_one_data(
                collection_name,
                {"student_id": student_id},
                {"$push": {"my_courses": enhanced_course}}
            )
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Course already added"})
    
    elif type == 'remove':
        update_one_data(
            collection_name,
            {"student_id": student_id},
            {"$pull": {"my_courses": {"id": course['id']}}}
        )
        return jsonify({"success": True})
    
    return jsonify({"success": False, "error": "Invalid operation"})

@app.route('/get_single_course/<course_id>/<radar>', methods=['GET'])
def get_single_course(course_id, radar):
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))
     
    course = fetch_single_data('processed_courses', {"id": course_id})
    check_course = fetch_single_data('students', {"my_courses.id": course_id, "student_id": student_id, })
    student_profile = fetch_single_data('students', {"student_id": student_id, })
    if check_course is not None:
        my_courses = check_course['my_courses']
        course_match = [item for item in my_courses if item["id"] == course_id]
        course_details = {"name": course_match[0]['name'], "similarity_scores": course_match[0]['similarity_scores']}
        match_explanation = explain_matching(student_profile,course)
        graph = plot_top_course_radar(course_details)
     
    # if not course:
    #     return render_template('404.html', message="Course not found"), 404

    # Render the template with course details
    return render_template('coursedescription.html', course=course, username=session.get('username'), radar=radar, graph=graph if check_course is not None else None, explanation=match_explanation if check_course is not None else None) #

@app.route('/get_single_course/<course_id>/<radar>', methods=['POST', 'GET'])
def update_course_data(course_id, radar):

    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))

    # Fetch course from database
    course = fetch_single_data('processed_courses', {"id": course_id})
    student_profile = fetch_single_data('students', {"student_id": student_id})
    if not course:
        return {"error": "Course not found"}, 404  # Handle missing course

    # Retrieve form data
    name = request.form.get("name") 
    similarity_scores = request.form.get("similarities") 
    
    match_explanation = explain_matching(student_profile,course)

    # Handle cases where data might be missing
    if similarity_scores:
        try:
            similarities_dict = ast.literal_eval(similarity_scores)  # Convert to dictionary
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing similarities: {e}")
            similarities_dict = {}  # Default to empty if parsing fails
    else:
        similarities_dict = {}  # Ensure it's always a dictionary

    # Prepare course dictionary
    course_details = {"name": name, "similarity_scores": similarities_dict}
    
    # Debugging: Print final course data
    print(f"Final course data: {course_details}")

    # Generate the radar chart
    graph = plot_top_course_radar(course_details)

    return render_template('coursedescription.html', 
                           explanation=match_explanation,
                           course=course, 
                           username=session.get('username'), 
                           radar=radar, 
                           graph=graph)

@app.route('/course', methods=['GET'])
def get_my_courses():
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))
    
    courses = fetch_local_data('processed_courses')
    collection_name = 'students'
    student_courses = fetch_single_data(collection_name, {"student_id": student_id})
       
    return render_template('questionnaire.html', username=session.get('username'), 
                           student_courses=student_courses, 
                           courses=courses, 
                           active_tab='myCoursesTab', 
                           terms=terms,
                           languages=languages,
                           language_levels=language_levels,
                           student_professional_backgrounds=student_professional_backgrounds,
                           programming_languages=programming_languages,
                           programming_levels=programming_levels,
                           student_majors = student_majors,
                           student_math_levels=student_math_levels,)

@app.route('/insights')
def insights():
        
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))
    
    return render_template('insights.html', username=session.get('username'))

@app.route('/setting')
def setting():
    student_id = session.get('student_id')
    if student_id is None:
        return redirect(url_for('welcome'))
    
    collection_name = 'students'
    student_data = fetch_single_data(collection_name, {"student_id": student_id})

    return render_template('setting.html',  
                           terms=terms,
                           student_data=student_data,
                           languages=languages,
                           language_levels=language_levels,
                           student_professional_backgrounds=student_professional_backgrounds,
                           programming_languages=programming_languages,
                           programming_levels=programming_levels,
                           student_majors = student_majors,
                           student_math_levels=student_math_levels, username=session.get('username'))


@app.route('/dash/')
def dash_index():
    # Redirect directly to Dash if needed
    return redirect('/dash/')

@app.route('/home_dash/')
def home_index():
    # Redirect directly to Dash if needed
    return redirect('/home_dash/')


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

# Sample options
module_options = ["All"] + ["Basics", "Intelligent Networked Systems", "Interactive Systems and Visualization"]
semester_options = ["All", "Winter", "Summer"]
language_options = ["All", "English", "German"]
math_mapping = {"None": 3, "Low": 1, "Moderate": 2, "High": 0, 'Very High': 4, 'All': 'All' }
math_level_options = ['All', "None", "Low", "Moderate", "High", 'Very High']



dash_app.layout = dcc.Loading(
    type="circle", 
    children=html.Div(style={'height': '80vh'}, children=[
        html.Div(
            style={'display': 'flex', 'flexDirection': 'column', 'height': '80vh'}, 
            children=[
                # Top Dropdown Selector
                html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '5px'},
                    children=[
                        dcc.Dropdown(
                            id='pie_type',
                            options=[{'label': 'Lecturers', 'value': 'Lecturers'}, 
                                     {'label': 'Network Graph', 'value': 'Network Graph'},
                                    #  {'label': 'Course Similarity Breakdown', 'value': 'Course Similarity Breakdown'},
                                     {'label': 'Self-Study Hours vs Lecture Duration', 'value': 'Self-Study Hours vs Lecture Duration'},
                                     
                                     ],
                            value='Network Graph'  # Default Selection
                        )
                        
                    ]
                ),

                # Conditional Filters (Initially Visible)
                html.Div(
                    id='filters-container',
                    style={'display':'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '5px', 'marginTop':'10px'},  # Initially visible
                    children=[
                        html.Div(children=[
                        html.Label("Filter by Module:"),
                        dcc.Dropdown(
                            id='module-dropdown',
                            options=[{'label': module, 'value': module} for module in module_options],
                            value='All',
                            multi=False
                        ),
                        ]),
                        html.Div(children=[
                             html.Label("Filter by Semester:"),
                        dcc.Dropdown(
                            id='semester-dropdown',
                            options=[{'label': semester, 'value': semester} for semester in semester_options],
                            value='All',
                            multi=False
                        ),
                        ]),
                        html.Div(children=[
                        html.Label("Filter by Language:"),
                        dcc.Dropdown(
                            id='language-dropdown',
                            options=[{'label': lang, 'value': lang} for lang in language_options],
                            value='All',
                            multi=False
                        ),
                        ]),
                        html.Div(children=[
                          html.Label("Filter by Math Level:"),
                        dcc.Dropdown(
                            id='math-level-dropdown',
                            options=[{'label': level, 'value': math_mapping.get(level)} for level in math_level_options],
                            value='All',
                            multi=False
                        ),
                        ]),  

                    ]
                ),
               


                # Graph Output
                html.Div(
                    dcc.Graph(id="pie-output"),
                    # style={'width': '100%', 'height': '80vh'} 
                )
            ]
        ),
    ])
)

@dash_app.callback(
    Output('pie-output', 'figure'),
    [
        Input('pie_type', 'value'),
        Input('module-dropdown', 'value'),
        Input('semester-dropdown', 'value'),
        Input('language-dropdown', 'value'),
        Input('math-level-dropdown', 'value')
    ]
)
def update_graph(pie_type, selected_module, selected_semester, selected_language, selected_math_level):
    if pie_type == "Lecturers":
        return course_lecturer()  # Call existing lecturer visualization
    elif pie_type == "Self-Study Hours vs Lecture Duration":
        return plot_self_study_analysis()[0] 
    # elif pie_type == "Self-Study Hours per Lecture Hour": 
    #     return plot_self_study_analysis()[1]  

    elif pie_type == "Network Graph":
        # Apply filters dynamically
        filtered_courses = [
            course for course in courses_data
            if (selected_module == "All" or course.get("encoded_module") == selected_module)
            and (selected_semester == "All" or (selected_semester in course.get("semester", "")))  
            and (selected_language == "All" or course.get("encoded_language") == selected_language)
            and (selected_math_level == "All" or course.get("math_level", "") == selected_math_level)
        ]

        # If no matching courses, return an empty graph
        if not filtered_courses:
            return {
                "data": [],
                "layout": {"title": "No courses found for the selected filters"}
            }

        # Recalculate course correlation based on filtered courses
        correlations = calculate_course_correlation(filtered_courses)
        nodes, links = prepare_plotly_data(correlations)

        # Generate updated visualization
        return visualize_course_correlation(nodes, links, filtered_courses)

    return {}  # Default empty output
  
@dash_app.callback(
    Output('filters-container', 'style'),
    Input('pie_type', 'value')
)
def toggle_filters(pie_type):
    if pie_type == "Network Graph":
        return {'display':'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '5px', 'marginTop':'10px'}  # Show filters
    else:
        return {'display': 'none'}  # Hide filters
    

def update_knowledge_graph(selected_module, selected_semester, selected_language, selected_math_level):
    # Fetch all courses

    # Apply filters dynamically
    filtered_courses = [
        course for course in courses_data
        if (selected_module == "All" or course.get("encoded_module") == selected_module)
        and (selected_semester == "All" or (selected_semester in course.get("semester", "")))  # Handles "Sommer 2024"
        and (selected_language == "All" or course.get("encoded_language") == selected_language)
        and (selected_math_level == "All" or str(course.get("math_level", "")) == selected_math_level)
    ]

    # If no matching courses, return an empty graph
    if not filtered_courses:
        return {
            "data": [],
            "layout": {"title": "No courses found for the selected filters"}
        }

    # Recalculate course correlation based on filtered courses
    correlations = calculate_course_correlation(filtered_courses)
    nodes, links = prepare_plotly_data(correlations)

    # Generate updated visualization
    return visualize_course_correlation(nodes, links, filtered_courses)


home_dash_app.layout = dcc.Loading(
    type="circle", 
    children=html.Div(children=[
     html.Div(
           
            style={
                'display': 'flex',
                'flexDirection':'column',
                'height': '80vh'

            }
            , children= [
                html.Div(style={'display':'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '5px'},
                children=[
               
            html.Div(style={'display': 'flex', 'flexDirection': 'column'}, 
                     children=[
                         dcc.Dropdown(['Module', 'Languages', 'Teaching Style'], 'Languages', id='graph'), 
                         html.Div( 
                         dcc.Graph(
                         id="graph_type",
                         ),
                         style={'width': '100%', 'height': '80vh'} 
                        ),
                     ]),
                     
                # html.Div( 
                #          dcc.Graph(
                #          id='similarity-chart'
                #          ),
                #          style={'width': '100%', 'height': '80vh'} 
                #         ),
                ]),
                    # Hidden div to store student data
                # html.Div(id='student-data-store', style={'display': 'none'})
            
            
            ]
        ),
    ]) 
)
  

@home_dash_app.callback(
    Output(component_id='graph_type', component_property='figure'),
    Input(component_id='graph', component_property='value')
  )
def update_div(graph):
    if graph=='Languages': 
        return course_languages()
    elif graph=='Module': 
        return course_modules()
    elif graph=='Teaching Style': 
        return courses_by_teaching_style()
    

# Callback to update the chart when data is available
@dash_app.callback(
    Output('similarity-chart', 'figure'),
    [Input('student-data-store', 'children')]
)
def update_chart(student_data_json):
    if not student_data_json:
        # Default empty state
        return go.Figure(layout=dict(
            title="No student data available",
            annotations=[dict(
                text="Waiting for student data...",
                showarrow=False,
                font=dict(size=20)
            )]
        ))
    
    # Parse student data
    student_data = json.loads(student_data_json)
    
    # Generate and return the chart
    return generate_course_similarity_chart(student_data)



if __name__ == '__main__':
    app.run(debug=True)

