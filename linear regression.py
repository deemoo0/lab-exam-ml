
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv(r"https://raw.githubusercontent.com/deemoo0/lab-exam-ml/refs/heads/main/bmi.csv")


print(data.head())
print(data.info())
print(data.describe())
data["Height"] = 100 * data["Height"]


X = data[["Age", "Height", "Weight"]]
y = data["Bmi"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


new_person = [[25, 170, 70]]
predicted_bmi = model.predict(new_person)
print("Predicted BMI:", predicted_bmi[0])


plt.scatter(y_test, y_pred, color="blue", label="Predicted points")
m, b = np.polyfit(y_test, y_pred, 1)
plt.plot(y_test, m*y_test + b, color="red", label="Best Fit Line")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="green", linestyle="--", label="Ideal Line (y=x)")
plt.xlabel("Actual BMI")
plt.ylabel("Predicted BMI")
plt.title("Actual vs Predicted BMI")
plt.show()
