# SMART COURSE SELECTOR

<img src="static/images/banner.png">
<img src="https://img.shields.io/badge/language-python-green"/>
<img src="https://img.shields.io/badge/Course-Learning Analytics-red"/>
<img src="https://img.shields.io/badge/Recommendations Type-Content Based Filtering-yellow"/>
<img src="https://img.shields.io/badge/University-Uni DUE-orange"/>

## Table of Contents

- [About](#-about)
- [Architecture](#%EF%B8%8F-architecture)
- [Libraries and Algorithms](#-libraries-and-algorithms)
- [A Look at the App](#-a-look-at-the-app)
- [Running the app](#-running-the-app)
- [Demo](#-demo)
- [Feedback and Contributions](#-feedback-and-contributions)
- [Authors](#-authors)
- [License](#-license)

## 🚀 About

**Smart Course Selector** is an intelligent course recommendation tool designed to help students of the MSc. Computer Engineering (UDE) program find courses that best match their preferences, interests, and academic goals. By analyzing various factors such as preferred semester, past programming experiences, language proficiency among others, the app provides personalized course recommendations to optimize the learning experience.

**Key Features**

- **Personalized Course Suggestions**: Tailored recommendations based on your unique learning preferences.
- **Interest & Concept-Based Selection**: Find courses that align with topics and subjects you enjoy.
- **Data-Driven Insights**: Leverages intelligent algorithms to provide meaningful and accurate recommendations.

## ⚙️ Architecture

<img src="static/images/tech_architecture.jpg"/>

## 📚 Libraries and Algorithms

### Core Libraries

- **Flask**: Web framework for serving the application
- **PyMongo**: MongoDB integration for Python
- **Plotly**: Interactive data visualization
- **Dash**: Framework for building analytical web applications
- **Flask-WTF**: Form handling and validation
- **scikit-learn**: Machine learning utilities
- **NumPy**: Numerical computing
- **python-dotenv**: Environment variable management
- **pickle**: Serialization and deserialization of Python objects, used for storing trained models

### Recommendation System Architecture

This system implements a content-based recommendation algorithm that suggests courses to students based on their profile data, preferences, and course descriptions.

### Core Algorithm: Content-Based Filtering with Word2Vec

The recommendation engine employs Word2Vec models to capture semantic relationships between course descriptions and student preferences. This approach goes beyond simple keyword matching by:

1. **Semantic Understanding**: Word2Vec transforms text into vector representations that capture the semantic meaning of words and concepts
2. **Contextual Similarity**: Measures how closely course content aligns with student interests and background
3. **Feature Extraction**: Converts unstructured text data into meaningful numerical representations

### Similarity Metrics and Distance Calculations

#### Cosine Similarity

Our primary similarity metric for content matching is cosine similarity, which measures the cosine of the angle between two non-zero vectors:

```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

This algorithm was chosen because:

- It effectively captures semantic similarity regardless of magnitude
- It works well with high-dimensional sparse vectors typical in text analysis
- It produces normalized results between -1 and 1, where 1 represents identical direction
- It emphasizes the orientation rather than magnitude of text feature vectors

#### Euclidean Distance

For certain matching criteria (like math level and time availability), we employ Euclidean distance to measure the direct distance between feature points:

```
euclidean_distance(A, B) = √∑(Aᵢ - Bᵢ)²
```

This metric is particularly useful for:

- Calculating differences in numeric attributes
- Measuring absolute distances in multi-dimensional feature space
- Providing intuitive distance measurements for non-textual features

### Vector Processing Pipeline

1. **Text Preprocessing**: Tokenization, stopword removal, and lemmatization
2. **Vector Embedding**: Conversion of processed text to numerical vectors using Word2Vec
3. **Feature Weighting**: Applying TF-IDF weighting to emphasize important terms
4. **Dimension Reduction**: Optional PCA for feature space optimization
5. **Similarity Computation**: Calculating similarity scores using the metrics above
6. **Score Normalization**: Scaling scores to a consistent range for comparison
7. **Weighted Aggregation**: Combining individual criterion scores into a final recommendation score

### Matching Criteria

The system calculates a composite matching score based on multiple factors:

- **Content Similarity**: Semantic matching between student interests and course content (cosine similarity of Word2Vec vectors)
- **Math Level Compatibility**: Alignment between course mathematical requirements and student background (Euclidean distance + Threshold-Based Scoring)
- **Time Availability Match**: Comparison of course workload to student's available time (Normalized Euclidean distance)
- **Language Proficiency**: Match between course language and student's language abilities (Proficiency-Weighted Cosine Similarity)
- **Programming Requirements**: Alignment of student's programming skills with course needs (Multi-dimensional Comparison)

### Model Persistence

The trained Word2Vec model is serialized using Python's pickle module for efficient storage and retrieval, allowing for:

- Fast loading of pre-trained models without retraining
- Consistent semantic relationship calculations across system restarts
- Reduced computational overhead during recommendation generation

## 📸 A Look at the App

### Application Flow

<img src="static/images/screenshots/flow/final-flow.jpg"/>

### Closer look at visualizations

<img src="static/images/screenshots/visualization/v1.png"/>
<img src="static/images/screenshots/visualization/v2.png"/>
<img src="static/images/screenshots/visualization/v3.png"/>

## 📝 Running the app

To run the app locally, follow these steps:

```shell
# Open a terminal (Command Prompt or PowerShell for Windows, Terminal for macOS or Linux)

# Ensure Git is installed
# Visit https://git-scm.com to download and install console Git if not already installed

# Clone the repository
git clone https://github.com/dengwanlin/Course-selection-advisor.git

# Navigate to the project directory
cd Course-selection-advisor

# Install required libraries
pip install -r requirements.txt

# Contact any of the authors for our `.env` file to access the database
# Paste the `.env` file in the root of the  Course-selection-advisor folder

# Run app
python app.py

```

## 🤝 Demo Video

[![Smart Course Advisor](https://img.youtube.com/vi/LqJigVvI2raKsFcs/0.jpg)](https://youtu.be/fjyngh9PJAY?si=LqJigVvI2raKsFcs)

_Click the image above to watch a demonstration of the Course Recommendation System in action._

## 🤝 Feedback and Contributions

> [!IMPORTANT]
> Whether you have feedback on features, have encountered any bugs, or have suggestions for enhancements, we're eager to hear from you. Your insights help us make the Smart Course Advisor better for students

We appreciate your support and look forward to making our product even better with your help!

## 👥 Authors

- Clement Ankomah - [@kojobaffoe011](https://github.com/@kojobaffoe011)
- Shafika Islam
- Haihua Wang
- Marta Zhao Ladrón de Guevara Cano
- Laura María García Pulido
- Hazem Al Massalmeh

## 📃 License

Distributed under the MIT License. See `LICENSE.txt` for more information.

[Back to top](#top)
