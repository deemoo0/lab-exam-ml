# Logistic Regression Performance Evaluation (Wine Dataset)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             precision_recall_fscore_support)

# Load Dataset
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names
target_names = wine.target_names

# Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train Model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Accuracy:", round(acc, 3))
print("Precision:", round(prec, 3))
print("Recall:", round(rec, 3))
print("F1 Score:", round(f1, 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix - Wine Dataset")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Class-wise F1 Score Bar Plot
precisions, recalls, f1s, _ = precision_recall_fscore_support(y_test, y_pred)
plt.figure(figsize=(6,4))
plt.bar(target_names, f1s, color=['red', 'green', 'blue'])
plt.title("Class-wise F1 Scores - Wine Dataset")
plt.xlabel("Wine Class")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
for i, v in enumerate(f1s):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.show()
