# K-Means Clustering on Wine Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.stats import mode
from sklearn.model_selection import train_test_split

# Load Dataset
data = load_wine()
X, y = data.data, data.target

# Scale Data
X = StandardScaler().fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train K-Means Model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
y_train_pred = kmeans.fit_predict(X_train)
y_test_pred = kmeans.predict(X_test)

# Relabel clusters to match real classes
mapping = {}
for i in range(3):
    mapping[i] = mode(y_train[y_train_pred == i], keepdims=True).mode[0]
y_test_mapped = np.array([mapping[label] for label in y_test_pred])

# Evaluation
acc = accuracy_score(y_test, y_test_mapped)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_test_mapped))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_mapped)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - K-Means (Wine Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans.predict(X), cmap='viridis', s=50)
plt.title(f"K-Means Clustering (Wine Data)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Cluster ID")
plt.show()
