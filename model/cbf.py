import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymongo import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://whhxsg:whhxsg@coursecluster.ecl2n.mongodb.net/?retryWrites=true&w=majority&appName=CourseCluster"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['Course_Recommendation']
collection = db['processed_courses']
X = collection.find()
courses_data = list(X)


stop_words  = ENGLISH_STOP_WORDS

def preprocess_course(course):
    course_text = course["Course_Description"] + " " + " ".join(course["Extracted_Concepts"])
    return [word for word in course_text.lower().split() if word not in stop_words]


# Helper functions for encoding categorical and numerical features
def encode_teaching_style(style):
    mapping = {"Course+Exercise+Project": 0.0, "Course+Exercise": 1.0, "Course+Project": 2.0}
    return mapping.get(style, 0.0)

def encode_course_language(lang):
    mapping = {"English": 0.0, "German": 1.0, "Both": 2.0}
    return mapping.get(lang, 2.0)

def encode_course_module(module):
    mapping = {
        "Interactive Systems and Visualization": 0.0,
        "Intelligent Networked Systems": 1.0,
        "Basic": 2.0,
    }
    return mapping.get(module, 0.0)

def encode_user_input(user_input):
    return {
        "Course_Teaching_Style": encode_teaching_style(user_input.get("teaching_style")),
        "Course_Language": encode_course_language(user_input.get("preferred_language")),
        "Course_Module": encode_course_module(user_input.get("module")),
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
def get_course_recommendations(courses, user_input, top_n=5):
    # Encode user input
    user_encoded_input = encode_user_input(user_input)

    # Compute user vector
    user_categorical_vector = [
        user_encoded_input.get("Course_Language", 0.0),
        user_encoded_input.get("Course_Module", 0.0),
        user_encoded_input.get("Course_Teaching_Style", 0.0),
    ]

    # User TF-IDF vector
    user_keywords_text = " ".join(user_input["keywords"])
    user_tfidf_vector = tfidf_vectorizer.transform([user_keywords_text])

    recommendations = []
    for idx, course in enumerate(courses):
        # # Textual similarity
        course_tfidf_vector = tfidf_matrix[idx]
        text_similarity = cosine_similarity(user_tfidf_vector, course_tfidf_vector)[0][0]

        # Categorical similarity
        course_categorical_vector = [
            encode_course_language(course.get("Course_Language", "")),
            encode_course_module(course.get("Course_Module", "")),
            encode_teaching_style(course.get("Course_Teaching_Style", "")),
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
            "Course_ID": course["Course_ID"],
            "Course_Name": course["Course_Name"],
            "Course_Teacher": course["Course_Teacher"],
            "Term": course["Term"],
            "Score": final_score,
        })

    # Sort recommendations by score
    recommendations.sort(key=lambda x: x["Score"], reverse=True)

    return recommendations[:top_n]

# User input
user_input = {
    "preferred_language": "English",
    "math_level": "High",
    "keywords": ["Data Analysis"],
    "module": "Basics",
    "teaching_style": "Course+Exercise",
    "weighting": {"textual": 0.7, "categorical": 0.3},
}

recommendation = get_course_recommendations(courses_data, user_input)

for idx, rec in enumerate(recommendation):
    print(f"{idx + 1}. {rec['Course_Name']} - Score: {rec['Score']:.2f} - Lecturer: {rec['Course_Teacher']} - Semester: {rec['Term']}")




