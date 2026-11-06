import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from textblob import TextBlob

# 1. Web page URL
url = "https://www.sahyadri.edu.in/"

# 2. Scrape comments
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(response.text, "html.parser")

comments = [p.get_text(strip=True) for p in soup.find_all("p")]

# 3. Sample comments if no comments are found
if not comments:
    comments = [
        "I love this post! Very informative.",
        "This article is terrible.",
        "Pretty good explanation.",
        "I completely disagree.",
        "Excellent explanation, thanks!"
    ]

# 4. Perform setniment analysis
sentiments = []

for comment in comments:
    polarity = TextBlob(comment).sentiment.polarity

    if polarity > 0:
        sentiments.append("Positive")
    elif polarity < 0:
        sentiments.append("Negative")
    else:
        sentiments.append("Neutral")

# 5. Store results in DataFrame
df = pd.DataFrame({
    "Comment": comments,
    "Sentiment": sentiments
})

print(df)

# 6. Count sentiments
counts = df["Sentiment"].value_counts()
print("\nSentiment Counts:")
print(counts)

# 7. Bar chart
counts.plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")
plt.show()

# 8. Pie chart
counts.plot(kind="pie", autopct="%1.1f%%")
plt.title("Sentiment Distribution")
plt.ylabel("")
plt.show()
