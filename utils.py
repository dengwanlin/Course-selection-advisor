
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
import json
import plotly.io as pio




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

    return student_professional_backgrounds, terms, languages, language_levels, programming_languages, programming_levels, student_majors, student_math_levels


courses_data = fetch_local_data('processed_courses')
stop_words  = STOPWORDS


#helper function to preprocess course
def preprocess_course(course):
    course_text = course["description"] + " " + " ".join(course["core_concepts"])
    return [word for word in course_text.lower().split() if word not in stop_words]

# Text preprocessing function
def preprocess_text(text):
    tokens = simple_preprocess(text.lower())  # Tokenize and convert to lowercase
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]  # Remove stopwords and non-alphanumeric
    return tokens

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

# Calculate soft cosine similarity

def calculate_soft_cosine_similarity(user_text, courses=courses_data):
    # Preprocess and convert user input into Bag-of-Words and TF-IDF
    user_tokens = preprocess_text(user_text)
    user_bow = dictionary.doc2bow(user_tokens)
    user_tfidf = tfidf[user_bow]
    
    # Compute similarities for all courses
    similarities = []
    for idx, course_id in enumerate(courses):
        course_tfidf = tfidf_corpus[idx]
        similarity = termsim_matrix.inner_product(user_tfidf, course_tfidf, normalized=(True, True))
        similarities.append((course_id, similarity))
    return similarities

# Helper functions for encoding categorical and numerical features
def encode_teaching_style(style):
    mapping = {"Course+Exercise+Project": 0.0, "Course+Exercise": 1.0, "Course+Project": 2.0}
    return mapping.get(style, 0.0)

def encode_course_language(lang):
    mapping = {"English": 0.0, "German": 1.0, "Both": 2.0}
    return mapping.get(lang, 2.0)

def encode_math_level(level):
    mapping = {"None": 1.0, "Low": 2.0, "Moderate": 3.0, "High": 4.0, 'Very High': 5.0 }
    return mapping.get(level, 2.0)

def encode_course_module(module):
    mapping = {
        "Interactive Systems and Visualization": 0.0,
        "Intelligent Networked Systems": 1.0,
        "Basic": 2.0,
    }
    return mapping.get(module, 0.0)

def encode_user_input(user_input):
    return {
        "teaching_style": encode_teaching_style(user_input.get("teaching_style")),
        "language": encode_course_language(user_input.get("preferred_language")),
        "module": encode_course_module(user_input.get("module")),
        "math_level": encode_course_module(user_input.get("math_level")),
    }

#jaccard similarity between categorical values
def calculate_jaccard_similarity(vec1, vec2):
    intersection = sum(1 for x, y in zip(vec1, vec2) if x == y)
    union = len(vec1)
    return intersection / union if union else 0

# Recommendation function
def get_course_recommendations(user_input, courses=courses_data, dictionary=dictionary, tfidf=tfidf, termsim_matrix=termsim_matrix, tfidf_corpus=tfidf_corpus, top_n=5,):
    # Encode user input
    user_encoded_input = encode_user_input(user_input)

    # Compute user vector
    user_categorical_vector = [
        user_encoded_input.get("language", 0.0),
        user_encoded_input.get("module", 0.0),
        user_encoded_input.get("teaching_style", 0.0),
        user_encoded_input.get("math_level", 0.0),
    ]

    # User TF-IDF vector
    user_keywords_text = user_input["keywords"].replace(',', '')
    user_tokens = preprocess_text(user_keywords_text)  # Preprocess user keywords
    user_bow = dictionary.doc2bow(user_tokens)
    user_tfidf_vector = tfidf[user_bow]

    recommendations = []
    for idx, course in enumerate(courses):
        course_tfidf_vector = tfidf_corpus[idx]
        text_similarity = termsim_matrix.inner_product(
            user_tfidf_vector, course_tfidf_vector, normalized=(True, True)
        )

        # Categorical similarity
        course_categorical_vector = [
            encode_course_language(course.get("language", "")),
            encode_course_module(course.get("module", "")),
            encode_teaching_style(course.get("teaching_style", "")),
            encode_math_level(course.get("math_level", "")),
        ]
        categorical_similarity = calculate_jaccard_similarity(
            course_categorical_vector, user_categorical_vector
        )

        # Weighted score
        weight = user_input["weighting"]
        final_score = (
            weight["textual"] * text_similarity +
            weight["categorical"] * categorical_similarity
        )

        recommendations.append({
            "id": course["id"],
            "name": course["name"],
            "lecturer": course["lecturer"],
            "semester": course["semester"],
            "core_concepts": course["core_concepts"],
            "math_level": course["math_level"],
            "score": round(final_score * 100, 2) ,
        })

    # Sort recommendations by score
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    return recommendations[:top_n]




# Time matching
def time_match(student_time, course):
    total_course_time = course['self_study_hours'] + course['lecture_duration']
      # Handle the case where total_course_time is zero
    if total_course_time == 0:
        return 0  # Return 0 as there's no valid course time to compare with

    return min(1.0, student_time / total_course_time)

# Language matching
def language_match(student_languages, course_language):

    for lang in student_languages:
        if lang['Language'].strip() == course_language:
            if lang['Level'].strip() in ['native', 'fluent']:
                return 1.0
            elif lang['Level'].strip() in ['proficient']:
                return 0.8
            elif lang['Level'].strip() in ['basic']:
                return 0.5
    return 0.0

# Programming language matching
# def programming_match(student_programming, course_programming):
#     for i, prog in student_programming:
#         if prog['Language'] == course_programming['Language']:
#             levels = {"basic": 1, "qualified": 2, "proficient": 3, "advanced": 4}
#             student_level = levels.get(prog['Level'], 0)
#             course_level = levels.get(course_programming['Level'], 0)
#             return min(1.0, student_level / course_level) if course_level > 0 else 0.0
#     return 0.0

# Math level matching
def math_match(student_math, course_math):
    levels = {"basic": 1, "qualified": 2, "proficient": 3, "excellent": 4}
    # student_level = max([levels.get(math['Level'], 0) for math in student_math], default=0)
    student_level = levels.get(student_math, 0)
    course_level = levels.get(course_math, 0)
    if student_level >= course_level:
        return 1.0  # Meets or exceeds the requirement
    return student_level / course_level if course_level > 0 else 0.0


def get_course_recommendations_2(user_input, student, courses=courses_data, dictionary=dictionary, tfidf=tfidf, termsim_matrix=termsim_matrix, tfidf_corpus=tfidf_corpus, top_n=5,): 
   
    recommendations = []
    # for i, student in enumerate(students):
    student_time = student['Available_Exercise_Time_Per_Week']
    student_languages = student['Student_Language_Level']
    student_programming = student['Student_Programming_Level']
    student_math = student['Student_Math_Background']
    # student_scores = []
    user_keywords_text =  user_input["keywords"].replace(',', '') + " " + " ".join(student["Student_Professional_Background"]) 
        # user_tfidf_vector = tfidf_vectorizer.transform([user_keywords_text])
        # User TF-IDF vector
    user_tokens = preprocess_text(user_keywords_text)  # Preprocess user keywords
    user_bow = dictionary.doc2bow(user_tokens)
    user_tfidf_vector = tfidf[user_bow]

    for idx, course in enumerate(courses):
        # Subscores
        time_score = time_match(student_time, course)

        # Textual similarity
        

        course_tfidf_vector = tfidf_corpus[idx]
        content_similarity = termsim_matrix.inner_product(
            user_tfidf_vector, course_tfidf_vector, normalized=(True, True)
        )

        # content_similarity = similarity_matrix[i, j]
        math_score = math_match(student_math, course['math_level'])
        language_score = language_match(student_languages, course['encoded_language'])
        # programming_score = programming_match(student_programming, course.get('programming_requirements', {})[0])
        # Final score with weights
        final_score = (
            0.5 * content_similarity +
            0.2 * math_score +
            0.2 * time_score +
            0.1 * language_score
            # 0.15 * programming_score
        )
        recommendations.append({
                                "id": course["id"],
                                "name": course["name"],
                                "lecturer": course["lecturer"],
                                "semester": course["semester"],
                                "core_concepts": course["core_concepts"],
                                "math_level": course["math_level"],
                                "score": round(final_score * 100, 2) ,
                                'similarity_scores': {
                                'Content Similarity': float(content_similarity),
                                'Math Match': float(math_score),
                                'Time Match':  float(time_score),
                                'Language Match': float(language_score),
                                # 'Programming Match': programming_score
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

calculate_soft_cosine_similarity('Data Science')




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

course = {'id': 'ZKD50068', 'name': 'Intelligent Learning Environments', 'lecturer': 'Prof. Dr. Irene-Angelica Chounta', 'semester': 'Sommer 2024', 'core_concepts': ['Educational technology', 'Cognitive modeling', 'Learning environment', 'Technology application'], 'math_level': 1, 'score': 63.63, 'similarity_scores': {'Content Similarity': 0.1918661, 'Math Match': 1.0, 'Time Match': 1.0, 'Language Match': 1.0}}

def plot_top_course_radar(course):
    # Get the top course for the first student

    if course == {}:
        return go.Figure()
    
    course_name = course['name']
    course_details = course['similarity_scores']

    print(course_name, course_details)

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