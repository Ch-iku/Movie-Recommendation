import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("C:\\codinghai\\APLAB\\gproject\\netflix_titles.csv")


df.fillna('', inplace=True)

# Combine genre and description
df["content"] = df["listed_in"] + " " + df["description"]

# Convert text to numbers using TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["content"])

# Calculate similarity between shows
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend(title):

   
    if title not in df["title"].values:
        print("Show not found in dataset")
        return

    index = df[df["title"] == title].index[0]

    scores = list(enumerate(similarity[index]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    scores = scores[1:8]

    movie_indices = [i[0] for i in scores]

    print("Recommended Shows:")
    print(df["title"].iloc[movie_indices])


movie = input("Enter a Netflix show or movie: ")

recommend(movie)