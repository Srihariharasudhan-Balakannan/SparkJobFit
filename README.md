# SparkJobFit
A machine learning-based resume analyzer that evaluates job seeker's resumes against job descriptions to assess job fit. Built using Spark ML for model training, NLP for skill extraction, and Streamlit for an interactive UI.

# Defining the problem
* Goal is to compare resume with job description and provide a matching score (0-100). 
* This is a text similarity problem, which requires text processing, feature extraction, and machine learning modeling.

### 1. Step 1 - Data Preprocessing
* Before training the learning model, we need to clean and process text data.
* Tasks in this step:
    * Tokenization - Convert text into words.
    * Reove Stopwords - Remove common words (e.g., "the", "is", "and").
    * Vectorization (TF-IDF) - Convert text into numerical format for ML.
