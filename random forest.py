# Titanic Survival Prediction using Random Forest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve

# Load dataset
data = sns.load_dataset("titanic").dropna(subset=["survived"])

# Features & Target
X = data[["pclass", "sex", "age", "sibsp", "parch", "fare"]]
y = data["survived"]

# Preprocessing
X["sex"] = X["sex"].map({"female": 0, "male": 1})
X["age"].fillna(X["age"].median(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,max_features="sqrt", random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision, color="purple")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict new passenger
new_data = [[3, 1, 55.0, 0, 0, 71.28]]
print("New Passenger Survival Prediction:", model.predict(new_data))
