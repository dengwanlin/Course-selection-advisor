from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from db.connection import *
import pandas as pd
import plotly.express as px






# Create a new client and connect to the server
client = connect_to_cluster()
db = client[get_dbname()]

#functions for crud operations
def fetch_local_data(collection):
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


courses_data = fetch_local_data('processed_courses')
stop_words  = ENGLISH_STOP_WORDS

#helper function to preprocess course
def preprocess_course(course):
    course_text = course["description"] + " " + " ".join(course["core_concepts"])
    return [word for word in course_text.lower().split() if word not in stop_words]


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

# Vectorizer initialization (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Precompute text embeddings for all courses
course_texts = [" ".join(preprocess_course(course)) for course in courses_data]
tfidf_matrix = tfidf_vectorizer.fit_transform(course_texts)

# Recommendation function
def get_course_recommendations(user_input, courses=courses_data, top_n=5,):
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
    user_tfidf_vector = tfidf_vectorizer.transform([user_keywords_text])

    recommendations = []
    for idx, course in enumerate(courses):
        # # Textual similarity
        course_tfidf_vector = tfidf_matrix[idx]
        text_similarity = cosine_similarity(user_tfidf_vector, course_tfidf_vector)[0][0]

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
    module_counts = courses_df["encoded_module"].value_counts().reset_index()
    module_counts.columns = ["Module", "Count"]
    print(f"Count of 'Modules' in 'Courses': {module_counts}")
    
    fig = px.pie(
        module_counts,
        names="Module",
        values="Count",
        title="Distribution of Modules",
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
    title="Distribution of Languages",
    color_discrete_sequence=px.colors.qualitative.Pastel1
    )   
  
  return fig

