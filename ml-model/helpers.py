import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
import gensim.downloader as api
from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import json


# Load resources
# Download resources
download('stopwords')
download('punkt')
download('punkt_tab')
word_embeddings = api.load('word2vec-google-news-300')

# Load the JSON file
with open("files/processed_courses.json", "r") as file:
    courses_data = json.load(file)

# Preprocess course text
stop_words = set(stopwords.words('english'))


def preprocess_course(course):
    course_text = course["Course_Description"] + " " + " ".join(course["Extracted_Concepts"])
    return [word for word in course_text.lower().split() if word not in stop_words]

# Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]  # Remove stopwords and non-alphanumeric
    return tokens

# Create a dictionary and similarity matrix for courses
def create_similarity_resources(courses):
    # Preprocess text for all courses
    texts = [preprocess_text(course["Course_Description"] + " " + " ".join(course["Extracted_Concepts"])) for course in courses]
    
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

# Calculate soft cosine similarity
def calculate_soft_cosine_similarity(user_text, courses, dictionary, tfidf, termsim_matrix):
    # Preprocess and convert user input into Bag-of-Words and TF-IDF
    user_tokens = preprocess_text(user_text)
    user_bow = dictionary.doc2bow(user_tokens)
    user_tfidf = tfidf[user_bow]
    
    # Compute similarities for all courses
    similarities = []
    for course_id, course_tfidf in enumerate(courses):
        similarity = termsim_matrix.inner_product(user_tfidf, course_tfidf, normalized=(True, True))
        similarities.append((course_id, similarity))
    
    return similarities


# Precompute similarity resources
dictionary, tfidf, termsim_matrix, tfidf_corpus = create_similarity_resources(courses_data)


# Vectorizer initialization (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Precompute text embeddings for all courses
course_texts = [" ".join(preprocess_course(course)) for course in courses_data]
tfidf_matrix = tfidf_vectorizer.fit_transform(course_texts)

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

def calculate_jaccard_similarity(vec1, vec2):
    intersection = sum(1 for x, y in zip(vec1, vec2) if x == y)
    union = len(vec1)
    return intersection / union if union else 0

# Recommendation function
def get_course_recommendations(courses, user_input, dictionary, tfidf, termsim_matrix, tfidf_corpus, top_n=5):
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
    # user_tfidf_vector = tfidf_vectorizer.transform([user_keywords_text])
    user_tokens = preprocess_text(user_keywords_text)  # Preprocess user keywords
    user_bow = dictionary.doc2bow(user_tokens)
    user_tfidf_vector = tfidf[user_bow]

    recommendations = []
    for idx, course in enumerate(courses):
        # # Textual similarity
        # course_tfidf_vector = tfidf_matrix[idx]
        # text_similarity = cosine_similarity(user_tfidf_vector, course_tfidf_vector)[0][0]

        #   Textual similarity using soft cosine
        course_tfidf_vector = tfidf_corpus[idx]
        text_similarity = termsim_matrix.inner_product(
            user_tfidf_vector, course_tfidf_vector, normalized=(True, True)
        )

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
            "Score": final_score,
        })

    # Sort recommendations by score
    recommendations.sort(key=lambda x: x["Score"], reverse=True)

    return recommendations[:top_n]

# User input
user_input = {
    "preferred_language": "German",
    "math_level": "High",
    "keywords": ["begins",
            "synchronization",
            "calls",
            "soap",
            "foundational",
            "transactions"],
    "module": "Basics",
    "teaching_style": "Course+Exercise",
    "weighting": {"textual": 0.8, "categorical": 0.2},
}

# Get recommendations
recommended_courses = get_course_recommendations(courses_data, user_input, dictionary, tfidf, termsim_matrix, tfidf_corpus)

# Output recommendations
for idx, rec in enumerate(recommended_courses, 1):
    print(f"{idx}. {rec['Course_Name']} - Score: {rec['Score']:.2f}")
