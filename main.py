import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/ResumeDataSet.csv")
df.head()
df.columns
print(df['Resume'][0])
print(df['Category'][0])

stop_words = set(stopwords.words('english'))
def preprocess_text(text):
 text = text.lower()
 text = re.sub(r'[^a-z\s]', '', text)
 text = re.sub(r'\s+', ' ', text).strip()
 words = text.split()
 words = [word for word in words if word not in stop_words]
 return " ".join(words)
df['cleaned_resume'] = df['Resume'].apply(preprocess_text)
print(df['Resume'][0])
print("----- CLEANED -----")
print(df['cleaned_resume'][0])

job_description = """
Looking for a data scientist with skills in python, machine learning,
data analysis, and natural language processing
"""
all_texts = df['cleaned_resume'].tolist()
all_texts.append(job_description)

vectorizer = TfidfVectorizer(max_features = 5000)
tfidf_matrix = vectorizer.fit_transform(all_texts)
similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

scores = similarity_scores.flatten()
ranked_indices = np.argsort(scores)[::-1]
top_n = 5

for i in range(top_n):
    idx = ranked_indices[i]
    print(f"Rank {i+1} | Score: {scores[idx]:.4f}")
    print(df['Resume'][idx][:300])  # first 300 chars
    print("-"*50)

job_category = "Data Science"
relevant_indices = df[df['Category']==job_category].index.tolist()

def precision_at_k(ranked_indices, relevant_indices, k=5):
    top_k = ranked_indices[:k]

    relevant_count = 0
    for idx in top_k:
        if idx in relevant_indices:
            relevant_count += 1

    return relevant_count / k

p_at_5 = precision_at_k(ranked_indices, relevant_indices, k=5)
print("Precision@5:", p_at_5)

def recall_at_k(ranked_indices, relevant_indices, k=5):
    top_k = ranked_indices[:k]

    relevant_count = 0
    for idx in top_k:
        if idx in relevant_indices:
            relevant_count += 1

    return relevant_count / len(relevant_indices)

r_at_5 = recall_at_k(ranked_indices, relevant_indices, k=5)
print("Recall@5:", r_at_5)