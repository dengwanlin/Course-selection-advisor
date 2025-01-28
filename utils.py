
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from db.connection import *
import pandas as pd
import plotly.express as px
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
import gensim.downloader as api
from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



# Create a new client and connect to the server
client = connect_to_cluster()
db = client[get_dbname()]

#download resources
word_embeddings = api.load('word2vec-google-news-300')


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

# Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
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
dictionary, tfidf, termsim_matrix, tfidf_corpus = create_similarity_resources(courses_data)

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
    
    print(similarities
          )
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

# Vectorizer initialization (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Precompute text embeddings for all courses
course_texts = [" ".join(preprocess_course(course)) for course in courses_data]
tfidf_matrix = tfidf_vectorizer.fit_transform(course_texts)

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
    # user_tfidf_vector = tfidf_vectorizer.transform([user_keywords_text])
        # User TF-IDF vector
    user_tokens = preprocess_text(user_keywords_text)  # Preprocess user keywords
    user_bow = dictionary.doc2bow(user_tokens)
    user_tfidf_vector = tfidf[user_bow]

    recommendations = []
    for idx, course in enumerate(courses):
        # # Textual similarity
        # course_tfidf_vector = tfidf_matrix[idx]
        # text_similarity = cosine_similarity(user_tfidf_vector, course_tfidf_vector)[0][0]
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

calculate_soft_cosine_similarity('Data Science')
