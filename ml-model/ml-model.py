from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def main():
   course_df = pd.read_json("courses.json")
   students_df = pd.read_json("students.json")

   course_df.columns = course_df.columns.str.strip()  # Remove extra spaces
   course_df.rename(columns=lambda x: x.lstrip('\ufeff'), inplace=True)
   students_df.columns = students_df.columns.str.strip()  # Remove extra spaces
   students_df.rename(columns=lambda x: x.lstrip('\ufeff'), inplace=True)

   # Merge data to create student-course combinations
   combined_data = pd.merge(course_df, students_df, how="cross")


    # Generate similarity features (e.g., matching language, major)
   combined_data['Language_Match'] = (combined_data['Preferred_Language'] == combined_data['Course_Language']).astype(int)
   combined_data['Major_Match'] = (combined_data['Major'] == combined_data['Course_Module']).astype(int)     

   # One-hot encode categorical variables
   encoder = OneHotEncoder()
   encoded_features = encoder.fit_transform(combined_data[['Course_Teaching_Style', 'Course_Module']])  

   # Normalize numerical fields (e.g., credit hours, difficulty level)
   scaler = StandardScaler()
   scaled_features = scaler.fit_transform(combined_data[['Course_Credit']])

   combined_data['Matched'] = False

   # Combine all features
   X = pd.concat([combined_data[['Language_Match', 'Major_Match']], pd.DataFrame(encoded_features.toarray())], axis=1)
   X.columns = X.columns.astype(str)
   y = combined_data['Matched']

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Train Random Forest
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   
   # Evaluate the model
   accuracy = model.score(X_test, y_test)
   print(f"Model Accuracy: {accuracy:.2f}")

#    # Predict probabilities for each course
#    student_profile = students_df.iloc[0]  # Example student
#    student_courses = course_df.copy()
#    student_courses['Predicted_Score'] = model.predict_proba(X_test)[:, 1]
   
#    # Sort courses by predicted score
#    recommendations = student_courses.sort_values(by='Predicted_Score', ascending=False)
#    print(recommendations[['Course_Name', 'Predicted_Score']])



   # Print the DataFrame
#    print(combined_data.columns.tolist())  # Access as a Series
  

if __name__ == "__main__":
    main()