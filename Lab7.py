import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# 1. Web page URLs
urls = [
    "https://www.bbc.com/news/technology",
    "https://www.cnn.com/tech"
]

# 2. Extract web page text
documents = []

for url in urls:
    page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(page.text, "html.parser")
    text = " ".join(p.get_text() for p in soup.find_all("p"))
    documents.append(text)

# 3. Convert text into TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(documents)

# 4. Apply topic modelling
model = NMF(n_components=2, random_state=42)
model.fit(X)

# 5. Display topics
words = vectorizer.get_feature_names_out()

for i, topic in enumerate(model.components_):
    top_words = [words[j] for j in topic.argsort()[-10:][::-1]]
    print("Topic", i+1, ":", ", ".join(top_words))
