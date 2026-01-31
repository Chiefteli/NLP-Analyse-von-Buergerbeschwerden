# Imports and Setup 

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

nltk.download('punkt')
nltk.download('punkt_tab')


# Daten laden und Prüfen 

csv_file = r'<Dateipfad>' # <-- DATEIPFAD ANPASSEN! 

df = pd.read_csv(csv_file, sep=',') 

text_col = 'MajorCategory'

df = df.dropna(subset=[text_col])

print("Daten sind geladen und geprüft.")


# Preprocessing 

def preprocess(text):
    text = str(text).replace('/', ' ')
    tokens = word_tokenize(text)
    clean_tokens = [w.lower() for w in tokens if w.isalpha()]
    return " ".join(clean_tokens)

df['clean_text'] = df[text_col].apply(preprocess)

print("Original:", df[text_col].iloc[0])
print("Clean:   ", df['clean_text'].iloc[0])


# Vektorisierung & Vergleich 


bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(df['clean_text'])

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

print("Vektorisierung abgeschlossen.")


# Themenextraktion 

k = 5

def show_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nThema {topic_idx+1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

print(f"--- LSA Themen (basierend auf TF-IDF) ---")
lsa = TruncatedSVD(n_components=k, random_state=1)
lsa.fit(tfidf_matrix)
show_topics(lsa, tfidf_vectorizer.get_feature_names_out(), 3)

print(f"\n--- LDA Themen (basierend auf BoW) ---")
lda = LatentDirichletAllocation(n_components=k, random_state=1)
lda.fit(bow_matrix)
show_topics(lda, bow_vectorizer.get_feature_names_out(), 3)







