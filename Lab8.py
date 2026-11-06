
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import PorterStemmer
!unzip -q -o data_exp8.zip
!ls data_exp8/Summaries

# 1. Load news articles
files = glob.glob("data_exp8/Summaries/**/*.txt", recursive=True)

df = pd.DataFrame({
    "filename": files,
    "text": [open(f, encoding="utf-8").read() for f in files]
})

print("Total Articles:", len(df))

# 2. Preprocess text
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    words = [stemmer.stem(w) for w in text.split()
             if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

# 3. Convert text into numerical format
vectorizer = CountVectorizer(max_df=0.9, min_df=2)
X = vectorizer.fit_transform(df["clean_text"])

words = vectorizer.get_feature_names_out()

# 4. Find top 20 frequent words
freq = X.sum(axis=0).A1
idx = freq.argsort()[-20:][::-1]

plt.bar(words[idx], freq[idx])
plt.xticks(rotation=60)
plt.title("Top 20 Frequent Words")
plt.tight_layout()
plt.show()

# 5. Apply LDA topic modelling
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# 6. Display topics
for i, topic in enumerate(lda.components_):
    top_words = words[topic.argsort()[-10:][::-1]]
    print("Topic", i+1, ":", ", ".join(top_words))

# 7. Display topic distribution
topics = lda.transform(X)

topic_df = pd.DataFrame(topics, columns=[
    "Topic_1", "Topic_2", "Topic_3", "Topic_4", "Topic_5"
])

topic_df.insert(0, "filename", df["filename"])

print("\nTopic Distribution:")
print(topic_df.head(10))
