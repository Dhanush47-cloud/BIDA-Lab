
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

# 1. Load dataset
df = pd.read_csv("data_exp1.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2. Encode data
X = pd.get_dummies(X)

if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average="weighted", zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted", zero_division=0))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 7. Plot results
metrics = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred, average="weighted", zero_division=0),
    recall_score(y_test, y_pred, average="weighted", zero_division=0),
    f1_score(y_test, y_pred, average="weighted", zero_division=0)
]

plt.bar(["Accuracy", "Precision", "Recall", "F1 Score"], metrics)
plt.ylim(0, 1.05)
plt.title("Decision Tree Performance")
plt.show()
