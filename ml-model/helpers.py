from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
import gensim.downloader as api
from gensim.matutils import corpus2csc
from nltk.corpus import stopwords
from nltk import download
import transformers
import numpy as np
import json


# Load resources
# word_embeddings = models.KeyedVectors.load_word2vec_format("files/GoogleNews-vectors-negative300.bin", binary=True)
word_embeddings = api.load('word2vec-google-news-300')
bert_tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()


download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')


# Load the JSON file
with open("files/processed_courses.json", "r") as file:
    courses_data = json.load(file)


# Combine descriptions and concepts for each course
def preprocess_course(course):

    course_text = course["Course_Description"] + " " + " ".join(course["Extracted_Concepts"])
    return [word for word in course_text.lower().split() if word not in stop_words]


def course_texts(courses):
    texts = [preprocess_course(course) for course in courses]
    dictionary = Dictionary(texts)
    documents = [dictionary(text) for text in texts]
    return documents

# Step 1: Soft Cosine Similarity
def compute_soft_cosine_measure(course1, course2):
    preprocess_text1 = preprocess_course(course1)
    preprocess_text2 = preprocess_course(course2)

    documents = [preprocess_text1, preprocess_text2]
    dictionary = Dictionary(documents)

    preprocess_text1 = dictionary.doc2bow(preprocess_text1)
    preprocess_text2 = dictionary.doc2bow(preprocess_text2)

    documents = [preprocess_text1, preprocess_text2]

    tfidf = TfidfModel(documents)

    preprocess_text1 = tfidf[preprocess_text1]
    preprocess_text2 = tfidf[preprocess_text2]

    termsim_index = WordEmbeddingSimilarityIndex(word_embeddings)
    termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)

    similarity = termsim_matrix.inner_product(preprocess_text1, preprocess_text2, normalized=(True, True))

    return similarity

# Function to calculate Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = len(set(set1).intersection(set(set2)))
    union = len(set(set1).union(set(set2)))
    return intersection / union if union != 0 else 0

# Function to normalize and calculate numerical similarity
def numerical_similarity(value1, value2):
    diff = abs(value1 - value2)
    return 1 - diff  # Inverse to make higher similarity values better

# Overall similarity function
def get_course_similarity(course1, course2, weights):
    # Text similarity (concepts or descriptions)
    text_sim = compute_soft_cosine_measure(course1, course2,)

    # Categorical similarity (teaching language)
    lang_sim = jaccard_similarity(course1["Encoded_Language"], course2["Encoded_Language"])

    # Numerical similarity (credit hours)
    credit_sim = numerical_similarity(course1["Course_Credit"], course2["Course_Credit"])

    # Combine with weights
    overall_similarity = (
        weights["text"] * text_sim +
        weights["categorical"] * lang_sim +
        weights["numerical"] * credit_sim
    )
    return overall_similarity


weights = {
    "text": 0.8, # Weight for text similarity (soft cosine)
    "categorical": 0.1, # Weight for categorical similarity (e.g., teaching language)
    "numerical": 0.1   # Weight for numerical similarity (e.g., credit hours)
}

similarity_matrix = []

for i, course1 in enumerate(courses_data):
    similarities = {}
    for j, course2 in enumerate(courses_data):
        # if i < j:  # Only calculate for unique pairs
            similarity_score = get_course_similarity(course1, course2, weights)
            similarities[course2["Course_ID"]] = similarity_score

    similarity_matrix.append({
        "_id": course1["Course_ID"],
        "similarities": similarities
    })

# Save the similarity matrix to a new JSON file
with open("files/similarity-matrix.json", "w") as file:
    json.dump(similarity_matrix, file, indent=4)

# # Load similarity matrix (if needed)
# with open("similarity-matrix.json", "r") as file:
#     similarity_matrix = json.load(file)

# Define recommendation function
def get_recommendations(course_id, top_n=5):
    # Find the course in the similarity matrix
    course_similarity = next((entry for entry in similarity_matrix if entry["Course_ID"] == course_id), None)
    if not course_similarity:
        return []

    # Sort similarities in descending order
    sorted_courses = sorted(
        course_similarity["similarities"].items(),
        key=lambda x: x[1],  # Sort by similarity score
        reverse=True
    )

    # Retrieve details of the top N recommended courses
    recommendations = [
        next(course for course in courses_data if course["_id"] == course_id)
        for course_id, score in sorted_courses[:top_n]
    ]
    return recommendations

# # Example usage
# course_id = courses_data[0]["Course_ID"]  # Replace with an actual course ID
# top_recommendations = get_recommendations(course_id, top_n=5)

# print("Recommended Courses:")
# for rec in top_recommendations:
#     print(rec["Course_Name"], rec["Course_Description"])





