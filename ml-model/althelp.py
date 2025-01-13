
# def get_bert_embedding(text):
#     # Tokenize and encode the text
#     inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
#     # Pass the inputs through BERT
#     with torch.no_grad():
#         outputs = bert_model(**inputs)
    
#     # Use the [CLS] token's embedding as the sentence embedding
#     embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (1, 768)
#     return embedding.squeeze(0)  # Shape: (768,)

# def get_courses_similarity(course_1, course_2):
#     # Get BERT embeddings for both sentences
#     embedding1 = get_bert_embedding(course_1)
#     embedding2 = get_bert_embedding(course_2)
    
#     # Calculate cosine similarity
#     embedding1 = embedding1.numpy().reshape(1, -1)
#     embedding2 = embedding2.numpy().reshape(1, -1)
#     calculated_cosine_similarity = cosine_similarity(embedding1, embedding2)
#     return calculated_cosine_similarity[0][0]


# # Function to get BERT embeddings for text
# def get_bert_embedding(text):
#     inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = bert_model(**inputs)
#     return outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()

# # Function to calculate Jaccard similarity
# def jaccard_similarity(set1, set2):
#     intersection = len(set(set1).intersection(set(set2)))
#     union = len(set(set1).union(set(set2)))
#     return intersection / union if union != 0 else 0

# # Function to normalize and calculate numerical similarity
# def numerical_similarity(value1, value2):
#     diff = abs(value1 - value2)
#     return 1 - diff  # Inverse to make higher similarity values better

# # Overall similarity function
# def get_course_similarity(course1, course2, weights):
#     # Text similarity (concepts or descriptions)
#     text_sim = cosine_similarity(
#         [get_bert_embedding(course1["description"])],
#         [get_bert_embedding(course2["description"])]
#     )[0][0]

#     # Categorical similarity (teaching language)
#     lang_sim = jaccard_similarity(course1["languages"], course2["languages"])

#     # Numerical similarity (credit hours)
#     credit_sim = numerical_similarity(course1["credits"], course2["credits"])

#     # Combine with weights
#     overall_similarity = (
#         weights["text"] * text_sim +
#         weights["categorical"] * lang_sim +
#         weights["numerical"] * credit_sim
#     )
#     return overall_similarity

# # Example course data
# course1 = {
#     "description": "Introduction to machine learning concepts",
#     "languages": ["English", "German"],
#     "credits": 5
# }

# course2 = {
#     "description": "Basics of artificial intelligence and machine learning",
#     "languages": ["English"],
#     "credits": 6
# }


# def clean_course(course):
#     """
#     Cleans a single course entry.
#     """
#     # Normalize text fields
#     course["Course_Description"] = course.get("Course_Description", "").capitalize()
#     course["Course_Teacher"] = course.get("Course_Teacher", "").replace("Professor, ", "").strip()
#     course["Course_Exam_Difficulty"] = course.get("Course_Exam_Difficulty", "").capitalize()
#     course["Course_Required_Math_Level"] = course.get("Course_Required_Math_Level", "").replace("very high", "High")
    
#     # Convert numerical fields
#     if "Course_Credit" in course:
#         course["Course_Credit"] = int(course["Course_Credit"]) if course["Course_Credit"].isdigit() else None
#     if "Course_Class_Hours" in course:
#         course["Course_Class_Hours"] = int(course["Course_Class_Hours"]) if course["Course_Class_Hours"].isdigit() else None
#     if "Homework_mandatorily required_hours" in course:
#         course["Homework_mandatorily required_hours"] = int(course["Homework_mandatorily required_hours"]) if course["Homework_mandatorily required_hours"].isdigit() else None

#     # Normalize boolean fields
#     if "With_Course_Videos" in course:
#         course["With_Course_Videos"] = course["With_Course_Videos"].lower() == "yes"
    
#     # Return cleaned course
#     return course

# def clean_courses(file_path, output_path):
#     """
#     Reads a JSON file with courses, cleans each course, and saves the cleaned data to a new file.
#     """
#     try:
#         # Load the JSON file
#         with open(file_path, "r", encoding="utf-8") as file:
#             courses = json.load(file)
        
#         # Clean each course
#         cleaned_courses = [clean_course(course) for course in courses]

#         # Save the cleaned courses to a new file
#         with open(output_path, "w", encoding="utf-8") as file:
#             json.dump(cleaned_courses, file, indent=4, ensure_ascii=False)

#         print(f"Cleaned courses saved to {output_path}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # File paths
# input_file = "courses.json"
# output_file = "cleaned_courses.json"

# # Clean the courses
# clean_courses(input_file, output_file)


# # Calculate similarity
# weights = {"text": 0.5, "categorical": 0.3, "numerical": 0.2}
# similarity = get_course_similarity(course1, course2, weights)
# print(f"Overall Similarity: {similarity:.4f}")


# # Example usage
# similarity = get_courses_similarity('Advanced Image Sythesis','Learning Analytics')
# print(f"Cosine Similarity: {similarity}")
