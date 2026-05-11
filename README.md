# Resume Ranking System using NLP

This project is an NLP-based Resume Ranking System built using Python and Machine Learning techniques. The system analyzes resumes and ranks them according to their similarity with a given job description.

## Features
- Resume text preprocessing
- Stopword removal using NLTK
- TF-IDF vectorization
- Cosine similarity-based ranking
- Precision@K and Recall@K evaluation metrics
- Automatic ranking of top matching resumes

## Technologies Used
- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

## Dataset
The project uses a resume dataset containing resumes and their corresponding job categories.

Dataset columns:
- Resume
- Category

## Working Process

### 1. Data Preprocessing
The resumes are cleaned using:
- Lowercasing
- Removing special characters
- Removing stopwords
- Removing extra spaces

### 2. Feature Extraction
TF-IDF Vectorizer converts text data into numerical vectors for machine learning processing.

### 3. Similarity Calculation
Cosine Similarity is used to compare resumes with the job description and calculate matching scores.

### 4. Ranking
The resumes are sorted based on similarity scores and the top matching resumes are displayed.

### 5. Evaluation
The project evaluates recommendation performance using:
- Precision@5
- Recall@5

## Example Job Description
```python
Looking for a data scientist with skills in python, machine learning,
data analysis, and natural language processing
