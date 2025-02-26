
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from db.connection import *
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import networkx as nx
import pickle
import numpy as np


# Create a new client and connect to the server
client = connect_to_cluster()
db = client[get_dbname()]

#download resources
word_embeddings = api.load('word2vec-google-news-300')


#functions for crud operations
def fetch_local_data(collection, query={}):
    if query:
        return list(db[collection].find(query))
    return list(db[collection].find())

def fetch_single_data(collection_name, query):
    collection = db[collection_name]
    result = collection.find_one(query)
    # result = list(result)
    return result

def update_one_data(collection_name, filter, data):
    collection = db[collection_name]
    result = collection.update_one(filter, data)
    return result

def insert_data(collection_name, data):
    collection = db[collection_name]
    result = collection.insert_one(data)
    return result

def get_enums_data():
    enums_data = fetch_single_data('enums', {})
    student_professional_backgrounds = enums_data.get('Course_Recommended_Background', [])
    terms = enums_data.get('Course_Term', [])
    languages = enums_data.get('Student_Languge', [])
    language_levels = enums_data.get('Student_Language_Level', [])
    programming_languages = enums_data.get('Course_Required_Programming_Language', [])
    programming_levels = enums_data.get('Course_Required_Programming_Language_Level', [])
    student_majors = enums_data.get('Student_Major', [])
    for major in student_majors:
        if 'Direction_Name' not in major:
            major['Direction_Name'] = []
    student_math_levels = enums_data.get('Student_Math_Level', [])

    return student_professional_backgrounds, terms, languages, language_levels, programming_languages, programming_levels, student_majors, student_math_levels

# Custom Jinja filter to format programming skills
def format_skills(skill_list):
    if isinstance(skill_list, list) and len(skill_list) > 0 and isinstance(skill_list[0], dict):
        return ', '.join(f"{item['Language']}: {item['Level']}" for item in skill_list if 'Language' in item and 'Level' in item)
    return skill_list  # If it's not a list of dicts, return it as is

# Custom Jinja filter for mapping math levels
def format_math_level(math_level):
    math_mapping = {
        "None": 3, 
        "Low": 1, 
        "Moderate": 2, 
        "High": 0, 
        "Very High": 4, 
        "All": "All"
    }
    
    # If it's a string and exists in our mapping, return the mapped value
    if isinstance(math_level, str) and math_level in math_mapping:
        return math_mapping[math_level]
    
    # For the reverse case (if we receive the value and need to get the key)
    if math_level in math_mapping.values():
        # Find the key for this value
        for key, value in math_mapping.items():
            if value == math_level:
                return key
    
    # Return as is if no mapping found
    return math_level


#fetching courses to be used for recommendations
courses_data = fetch_local_data('processed_courses')
stop_words  = STOPWORDS

# similarity resources for calculating course similarity

# Create a dictionary and similarity matrix for courses
def create_similarity_resources(courses):
    # Preprocess text for all courses
    texts = [preprocess_text(course["description"] + " " + " ".join(course["core_concepts"])) for course in courses]
    
    # Create a dictionary and Bag-of-Words representations
    dictionary = Dictionary(texts)
    bow_corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Create a TF-IDF model
    tfidf = TfidfModel(bow_corpus)
    tfidf_corpus = [tfidf[doc] for doc in bow_corpus]
    
    # Create a WordEmbeddingSimilarityIndex and TermSimilarityMatrix
    termsim_index = WordEmbeddingSimilarityIndex(word_embeddings)
    termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)
    
    return dictionary, tfidf, termsim_matrix, tfidf_corpus

# Precompute similarity resources
print("Creating similarity resources...")
# Cache resources
def save_similarity_resources(dictionary, tfidf, termsim_matrix, tfidf_corpus, filename="similarity_resources.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"dictionary": dictionary, "tfidf": tfidf, "termsim_matrix": termsim_matrix,  "tfidf_corpus": tfidf_corpus}, f)

def load_similarity_resources(filename="similarity_resources.pkl"):
    with open(filename, "rb") as f:
        resources = pickle.load(f)
    return resources["dictionary"], resources["tfidf"], resources["termsim_matrix"], resources["tfidf_corpus"]

# Load or create resources
try:
    print("Loading cached similarity resources...")
    dictionary, tfidf, termsim_matrix, tfidf_corpus= load_similarity_resources()
    print("Cached resources loaded successfully.")
except FileNotFoundError:
    print("No cache found. Creating similarity resources...")
    dictionary, tfidf, termsim_matrix, tfidf_corpus = create_similarity_resources(courses_data)
    save_similarity_resources(dictionary, tfidf, termsim_matrix, tfidf_corpus)
    print("Resources saved for future use.")

print("Similarity resources created successfully.")



#helper function to preprocess course
def preprocess_course(course):
    course_text = course["description"] + " " + " ".join(course["core_concepts"])
    return [word for word in course_text.lower().split() if word not in stop_words]

# Text preprocessing function
def preprocess_text(text):
    tokens = simple_preprocess(text.lower())  # Tokenize and convert to lowercase
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]  # Remove stopwords and non-alphanumeric
    return tokens

# Time matching using normalized Euclidean distance
def time_match_distance(student_time, course):
    total_course_time = course['self_study_hours'] + course['lecture_duration']
    if total_course_time == 0:
        return 0
    
    # Convert to feature vectors (normalizing by typical max times)
    max_time = 40  # Assume 40 hours as max weekly time
    student_vec = np.array([student_time / max_time]).reshape(1, -1)
    course_vec = np.array([total_course_time / max_time]).reshape(1, -1)
    
    # Calculate distance and convert to similarity score (1 - normalized distance)
    distance = euclidean_distances(student_vec, course_vec)[0, 0]
    return max(0, 1 - min(1, distance))

# Language matching using vector embedding and cosine similarity
def language_match_distance(student_languages, course_language):
    # Convert language proficiency to vectors
    proficiency_values = {
        'native': 1.0,
        'fluent': 0.95,
        'proficient': 0.8,
        'intermediate': 0.6,
        'basic': 0.4,
        'beginner': 0.2,
        'none': 0
    }
    
    # Create language vectors (one-hot encoding with proficiency values)
    all_languages = set([course_language] + [lang['Language'].strip() for lang in student_languages])
    language_idx = {lang: i for i, lang in enumerate(all_languages)}
    
    # Create student vector
    student_vec = np.zeros(len(all_languages)).reshape(1, -1)
    for lang in student_languages:
        idx = language_idx[lang['Language'].strip()]
        student_vec[0, idx] = proficiency_values.get(lang['Level'].strip().lower(), 0)
    
    # Create course vector
    course_vec = np.zeros(len(all_languages)).reshape(1, -1)
    course_idx = language_idx.get(course_language, 0)
    course_vec[0, course_idx] = 1.0
    
    # Handle zero vectors
    if np.sum(student_vec) == 0 or np.sum(course_vec) == 0:
        return 0
    
    # Calculate similarity
    similarity = cosine_similarity(student_vec, course_vec)[0][0]
    return similarity 

# Programming language matching
def programming_match_multiple(student_programming, course_programming_requirements):
    # Handle case where course requires no programming
    if not course_programming_requirements or len(course_programming_requirements) == 0:
        return 1.0  # Perfect match - no programming needed
    
    # Dimensions: [language familiarity, complexity handling, algorithm knowledge]
    level_vectors = {
        'none': [0, 0, 0],
        'beginner': [0.2, 0.1, 0.1],
        'basic': [0.4, 0.3, 0.2],
        'qualified': [0.6, 0.5, 0.5],
        'proficient': [0.8, 0.7, 0.7],
        'advanced': [1.0, 0.9, 0.9],
        'expert': [1.0, 1.0, 1.0]
    }
    
    # Calculate match score for each required language
    language_scores = []
    
    for req_lang in course_programming_requirements:
          # Validate dictionary has required keys
        if not isinstance(req_lang, dict) or 'Language' not in req_lang or 'Level' not in req_lang:
            continue  # Skip invalid entries
        # Default vector for required language
        course_vec = np.array(level_vectors.get(req_lang['Level'].lower(), [0, 0, 0])).reshape(1, -1)
        best_student_vec = np.array([0, 0, 0]).reshape(1, -1)  # Start with no skills
        
        # Find best matching student skill for this language
        for prog in student_programming:
            if prog['Language'].lower() == req_lang['Language'].lower():
                student_vec = np.array(level_vectors.get(prog['Level'].lower(), [0, 0, 0])).reshape(1, -1)
                # Keep the best skill level if student has multiple entries for same language
                if np.sum(student_vec) > np.sum(best_student_vec):
                    best_student_vec = student_vec
        
        # Calculate score for this required language
        if all(best_student_vec[0][i] >= course_vec[0][i] for i in range(len(best_student_vec[0]))):
            language_scores.append(1.0)
        else:
            distance = euclidean_distances(best_student_vec, course_vec)[0,0]
            max_distance = euclidean_distances(np.array([0, 0, 0]).reshape(1,-1), np.array([1, 1, 1]).reshape(1,-1))[0,0]
            language_scores.append(max(0, 1 - (distance / max_distance)))
    
    # For multiple required languages, take the minimum score
    # This implements the "weakest link" principle
    return min(language_scores) if language_scores else 0.0

# Math level matching
def math_match_distance(student_math, course_math):
    # Student math level mapping
    student_level_values = {
        'none': 0,
        'basic': 0.25,
        'qualified': 0.5,
        'proficient': 0.75,
        'excellent': 1.0
    }
    
    # Course math level mapping
    course_level_values = {
        3: 0,      # None
        1: 0.33,   # Low
        2: 0.67,   # Moderate
        0: 0.9,    # High
        4: 1.0     # Very High
    }
    
    # Convert to numeric values
    student_val = student_level_values.get(student_math.lower(), 0)
    
    # Handle course_math as integer or string
    if isinstance(course_math, int):
        course_val = course_level_values.get(course_math, 0)
    else:
        try:
            course_val = course_level_values.get(int(course_math), 0)
        except (ValueError, TypeError):
            # Fallback if conversion fails
            course_val = 0
    
    # If student level exceeds course level, return 1.0
    if student_val >= course_val:
        return 1.0
    
    # Calculate normalized distance
    max_possible_distance = 1.0
    distance = abs(student_val - course_val)
    return max(0, 1 - (distance / max_possible_distance))

def explain_matching(student_profile, course):
    explanation = {}

    # Time Matching
    student_time = student_profile['Available_Exercise_Time_Per_Week']
    time_score = time_match_distance(student_time, course)
    explanation['Time Avaiilability Match'] = {
        'score': round(time_score, 2),
        'student_time': student_time,
        'course_time': course['encoded_lecture_duration'] + course['encoded_self_study'],
        'reason': f"Your available time is {student_time} hours/week, while this course requires {course['encoded_lecture_duration'] + course['encoded_self_study']} hours. "
                  f"A lower gap results in a higher match."
    }

    # Language Matching
    student_languages = student_profile['Student_Language_Level']
    course_language = course['encoded_language']
    language_score = language_match_distance(student_languages, course_language)
    explanation['Language Match'] = {
        'score': round(language_score, 2),
        'student_languages': student_languages,
        'course_language': course_language,
        'reason': f"Your best language match against the course language '{course_language}' determines this score."
    }

    # Programming Matching
    student_programming = student_profile.get('Student_Programming_Level', [])
    course_programming = course.get('programming_requirements', [])
    programming_score = programming_match_multiple(student_programming, course_programming)
    explanation['Programming Match'] = {
        'score': round(programming_score, 2),
        'student_skills': student_programming,
        'course_requirements': course_programming,
        'reason': "The score reflects how well your programming skills match the required levels of the course. "
                  "If you meet or exceed the required levels in all programming languages, you get a high score."
    }

    # Math Matching
    student_math = student_profile.get('Student_Math_Background', 'none')
    course_math = course.get('math_level', 3)  # Default to 'None'
    course_math_to_text = format_math_level(course_math)
    math_score = math_match_distance(student_math, course_math)
    explanation['Math Matching'] = {
        'score': round(math_score, 2),
        'student_level': student_math,
        'course_requirement': course_math_to_text,
        'reason': f"Your math level '{student_math}' is compared with the course's required level ({course_math_to_text}). "
                  f"More alignment leads to a higher score."
    }

    return explanation

def get_dynamic_weights(scores):
    """
    Dynamically assigns weights based on the variation in scores.
    """
    # Convert values to standard Python floats
    score_values = [float(v) for v in scores.values()]
    
    # Calculate standard deviations
    stds = []
    for score in score_values:
        # Calculate how far each score is from the mean
        diff = score - sum(score_values) / len(score_values)
        stds.append(abs(diff))
    
    # If all standard deviations are zero, return equal weights
    if sum(stds) == 0:
        return {k: 1.0 / len(scores) for k in scores}
    
    # Normalize the weights to sum to 1
    total_std = sum(stds)
    weights = {k: std / total_std for k, std in zip(scores.keys(), stds)}
    
    return weights

def get_course_recommendations_2(user_input, student, courses=courses_data, dictionary=dictionary, tfidf=tfidf, termsim_matrix=termsim_matrix, tfidf_corpus=tfidf_corpus, top_n=5,): 
   
    recommendations = []
    student_time = student['Available_Exercise_Time_Per_Week']
    student_languages = student['Student_Language_Level']
    student_programming = student['Student_Programming_Level']
    student_math = student['Student_Math_Background']
    user_keywords_text =  user_input["keywords"].replace(',', '') + " " + " ".join(student["Student_Professional_Background"]) 
    user_tokens = preprocess_text(user_keywords_text)  # Preprocess user keywords
    user_bow = dictionary.doc2bow(user_tokens)
    user_tfidf_vector = tfidf[user_bow]

    for idx, course in enumerate(courses):
        # Subscores
        time_score = time_match_distance(student_time, course)
        # Textual similarity
        course_tfidf_vector = tfidf_corpus[idx]
        content_similarity = termsim_matrix.inner_product(
            user_tfidf_vector, course_tfidf_vector, normalized=(True, True)
        )
        math_score = math_match_distance(student_math, course['math_level'])
        language_score = language_match_distance(student_languages, course['encoded_language'])
        course_programming_req = course.get('programming_requirements')
        programming_score = programming_match_multiple(student_programming, course_programming_req)

           # Store scores in a dictionary
        scores = {
            "content": content_similarity,
            "math": math_score,
            "time": time_score,
            "language": language_score,
            "programming": programming_score
        }

        # Get dynamic weights
        weights = get_dynamic_weights(scores)

        # Final score with weights
        final_score = sum(scores[k] * weights[k] for k in scores)

        recommendations.append({
                                "id": course["id"],
                                "name": course["name"],
                                "lecturer": course["lecturer"],
                                "semester": course["semester"],
                                "core_concepts": course["core_concepts"],
                                "description": course['description'],
                                "math_level": course["math_level"],
                                "score": round(final_score * 100, 2) ,
                                "radar": 'chart',
                                "weights": {k: float(v) for k, v in weights.items()},  # Convert weights to floats
                                'similarity_scores': {
                                'Content Similarity': float(scores['content']),
                                'Math Match': float(scores['math']),
                                'Time Match':  float(scores['time']),
                                'Language Match': float(scores['language']),
                                'Programming Match': float(scores['programming']),
                                }})
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:top_n]




























#get data for graphs
courses_df = pd.DataFrame(courses_data)

def courses_scatter():
      courses_df["core_concepts"] = courses_df["core_concepts"].apply(lambda x: ",".join(x))  # Join list of concepts into a string
      concepts_array = courses_df[["core_concepts"]]  # Convert to DataFrame with single column
      
      encoder = OneHotEncoder()
      concept_matrix = encoder.fit_transform(concepts_array).toarray()
      
      # Reduce dimensions
      pca = PCA(n_components=2)
      reduced_data = pca.fit_transform(concept_matrix)
      
      # Create a DataFrame
      df_reduced = pd.DataFrame(reduced_data, columns=["x", "y"])
      df_reduced["id"] = courses_df["id"]
      
      # Plot clustering
      fig = px.scatter(
          df_reduced, 
          x="x", 
          y="y", 
          text="id",
          size_max=200,
          title="Course Clustering based on Extracted Concepts"
      )
      
      
      # Update the font size specifically for the text labels
      fig.update_traces(
          textfont=dict(
              family="Courier New, monospace",
              size=7,  # Set the desired font size
              color="red"  # Optional: Change text color
          ),
          marker=dict(size=10)
      )

      return fig

def course_modules():
      # Count occurrences of each module
    module_name_mapping = {
        'Interactive Systems and Visualization': "IVS",
        'Intelligent Networked Systems': "INS",
        'Basics': "Basics"
    }
    module_counts = courses_df["encoded_module"].value_counts().reset_index()

    module_counts.columns = ["Module", "Count"]
    module_counts["Module"] = module_counts["Module"].map(module_name_mapping)
    print(f"Count of 'Modules' in 'Courses': {module_counts}")
    
    fig = px.pie(
        module_counts,
        names="Module",
        values="Count",
        # title="Distribution of Modules",
        color_discrete_sequence=px.colors.qualitative.Pastel1
    )

    return fig

def course_lecturer():
    fig = px.bar(courses_df, x="lecturer", title="Number of Courses by Teacher")

    return fig

def course_languages():
    # Count occurrences of each module
  language_counts = courses_df["encoded_language"].value_counts().reset_index()
  language_counts.columns = ["Language", "Count"]
  print(f"Count of 'Languages' in 'Courses': {language_counts}")

  fig = px.pie(
    language_counts,
    names="Language",
    values="Count",
    # title="Distribution of Languages",
    color_discrete_sequence=px.colors.qualitative.Pastel1
    )   
  
  return fig

#kknowlege graph
# Preprocess concepts
def preprocess_concepts(concepts):
    preprocessed = []
    for concept in concepts:
        tokens = simple_preprocess(concept.lower())
        filtered_tokens = [token for token in tokens if token not in STOPWORDS]
        preprocessed.append(filtered_tokens)
    return preprocessed

def calculate_course_correlation(courses):
    all_concepts = []
    course_name_list = []
    for course in courses:
        preprocessed_concepts = preprocess_concepts(course['core_concepts'])
        all_concepts.extend(preprocessed_concepts)
        course_name_list.append(course['name'])

    # Training the Word2Vec model
    model = Word2Vec(all_concepts, min_count=1)

    correlations = []
    for i in range(len(courses)):
        for j in range(i + 1, len(courses)):
            course1_concepts = preprocess_concepts(courses[i]['core_concepts'])
            course2_concepts = preprocess_concepts(courses[j]['core_concepts'])

            # Check if the course's core concepts list is empty
            if not course1_concepts or not course2_concepts:
                continue

            similarity = 0
            valid_comparisons = 0
            for concept1 in course1_concepts:
                for concept2 in course2_concepts:
                    if concept1 and concept2:
                        try:
                            similarity += model.wv.n_similarity(concept1, concept2)
                            valid_comparisons += 1
                        except KeyError:
                            continue
            if valid_comparisons > 0:
                similarity /= valid_comparisons
            correlations.append((course_name_list[i], course_name_list[j], similarity))

    return correlations
# Preparing data for Plotly visualization
def prepare_plotly_data(correlations):
    nodes = list(set([corr[0] for corr in correlations] + [corr[1] for corr in correlations]))
    node_indices = {node: index for index, node in enumerate(nodes)}

    links = []
    for corr in correlations:
        source = node_indices[corr[0]]
        target = node_indices[corr[1]]
        value = corr[2]
        links.append({'source': source, 'target': target, 'value': value})

    return nodes, links


# Visualization with Plotly
def visualize_course_correlation(nodes, links, courses):
    G = nx.Graph()
    node_colors = []
    for i, node in enumerate(nodes):
        G.add_node(i, name=node)
        node_color = None
        for course in courses:

         
            if course['name'] == node:
                module = course.get('encoded_module')
                if module == "Basics":
                    node_color = 'yellow'
                elif module == "Interactive Systems and Visualization":
                    node_color = 'red'
                else: node_color = 'green'
                break

        if node_color is None:
            node_color = 'gray' 
        node_colors.append(node_color)

       

    for link in links:
        G.add_edge(link['source'], link['target'], weight=link['value'])

    # Adjust layout algorithm parameters
    pos = nx.spring_layout(G, k=0.3, iterations=50)

    # Filter out low-relevance edges
    threshold = 0.05
    filtered_links = [link for link in links if link['value'] > threshold]

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [nodes[i] for i in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=20,  # Reduce node size
        ),
        textfont=dict(size=10)  # Reduce text size
    )

    link_x = []
    link_y = []
    link_text = []
    link_x_mid = []
    link_y_mid = []
    similarity_texts = []

    for link in filtered_links:
        x0, y0 = pos[link['source']]
        x1, y1 = pos[link['target']]
        link_x.extend([x0, x1, None])
        link_y.extend([y0, y1, None])
        
        # Compute midpoint for similarity text
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        link_x_mid.append(mid_x)
        link_y_mid.append(mid_y)
        similarity_texts.append(f"{link['value']:.2f}")

    # Edge (Link) Trace
    link_trace = go.Scatter(
        x=link_x, y=link_y,
        mode='lines',
        hoverinfo='text',
        text=similarity_texts,
        line=dict(
            color='rgba(128, 128, 128, 0.5)',
            width=1
        )
    )

    # Similarity Score Labels
    text_trace = go.Scatter(
        x=link_x_mid, y=link_y_mid,
        mode='text',
        text=similarity_texts,
        textposition='middle right',
        hoverinfo='none',
        textfont=dict(size=10, color="black")  # Adjust font size & color
    )

    fig = go.Figure(data=[link_trace, node_trace, text_trace])
    fig.update_layout(showlegend=False,  height=1000, autosize=False,)
    return fig

def plot_top_course_radar(course):
    # Get the top course for the first student

    if course == {}:
        return go.Figure()
    
    course_name = course['name']
    course_details = course['similarity_scores']

    # Radar chart data
    categories = list(course_details.keys())
    values = list(course_details.values())
    values.append(values[0])  # Close the radar chart

    

    # Draw radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],  # Close
        fill='toself',
        name=course_name
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Course Recommendation Radar Chart"
    )

    return fig.to_html(full_html=False, default_height=500, default_width=700)
    fig.show()


def plot_self_study_analysis(df=courses_df):
    figures = []

    # Bar Chart: Self-Study Hours vs. Lecture Duration
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=df["name"],
        y=df["encoded_self_study"],
        name="Self-Study Hours",
        marker=dict(color="royalblue")
    ))
    fig1.add_trace(go.Bar(
        x=df["name"],
        y=df["encoded_lecture_duration"],
        name="Lecture Duration",
        marker=dict(color="orange")
    ))
    fig1.update_layout(
        title="Comparison of Self-Study Hours and Lecture Duration",
        xaxis_title="Courses",
        yaxis_title="Hours",
        barmode="group",
        xaxis_tickangle=-45
    )
    figures.append(fig1)

    # Scatter Plot: Self-Study Efficiency vs. Courses
    df["self_study_efficiency"] = df["encoded_self_study"] / df["encoded_lecture_duration"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["name"],
        y=df["self_study_efficiency"],
        mode='markers+text',
        text=df["self_study_efficiency"].round(2),
        textposition="top center",
        marker=dict(size=10, color="green")
    ))
    fig2.update_layout(
        title="Self-Study Efficiency (Self-Study Hours per Lecture Hour)",
        xaxis_title="Courses",
        yaxis_title="Efficiency Ratio",
        xaxis_tickangle=-45
    )
    figures.append(fig2)

    return figures

def courses_by_teaching_style(df=courses_df):
    # Define the mapping of numerical values to their string labels
    teaching_style_labels = {
        0.0: "Course+Exercise+Project",
        1.0: "Course+Exercise",
        2.0: "Course+Project"
    }

    # Replace the numerical values with string labels in the 'Course_Teaching_Style' column
    df["teaching_style_label"] = df["teaching_style"].replace(teaching_style_labels)

    # Get the updated counts with labels
    teaching_style_counts = df["teaching_style_label"].value_counts().reset_index()
    teaching_style_counts.columns = ["Style", "Count"]

    # Print the updated counts
    print(f"Count of 'Teaching Styles' in 'Courses':\n{teaching_style_counts}")

    fig = px.pie(
        teaching_style_counts,
        names="Style",
        values="Count",
        # title="Distribution of Course Teaching Styles",
        color_discrete_sequence=px.colors.qualitative.Pastel1)
    
    return fig

def generate_course_similarity_chart(student_data):

   
    # Extract similarity data
    categories = []
    scores = []
    
    if 'my_courses' in student_data and student_data['my_courses']:
        for course in student_data['my_courses']:
            if 'similarity_scores' in course:
                for category, score in course['similarity_scores'].items():
                    categories.append(f"{category} - {course['name']}")
                    scores.append(score * 100)  # Convert to percentage
    
    # Create the bar chart
    if categories and scores:
        fig = go.Figure(go.Bar(
            x=scores,
            y=categories,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale='Blues',
                colorbar=dict(title="Match %"),
            )
        ))
        
        fig.update_layout(
            title="Course Match Metrics",
            xaxis_title="Match Percentage (%)",
            yaxis_title="Match Categories",
            xaxis=dict(range=[0, 100]),
            height=max(400, len(categories) * 40),  # Dynamic height based on categories
            margin=dict(l=20, r=20, t=50, b=20),
        )
    else:
        # Empty chart with message
        fig = go.Figure()
        fig.update_layout(
            title="No courses added yet",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text="Student has not added any courses yet",
                showarrow=False,
                font=dict(size=20)
            )]
        )
    
    return fig