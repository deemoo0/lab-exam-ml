from flask import Flask, request, render_template_string
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# HTML Template
template = """
<!DOCTYPE html>
<html>
<head>
    <title>Iris Flower Prediction</title>
</head>
<body style="text-align:center;">
    <h2>Iris Flower Species Prediction (Logistic Regression)</h2>
    <form method="POST">
        <label>Sepal Length:</label><input type="text" name="slen"><br><br>
        <label>Sepal Width:</label><input type="text" name="swid"><br><br>
        <label>Petal Length:</label><input type="text" name="plen"><br><br>
        <label>Petal Width:</label><input type="text" name="pwid"><br><br>
        <input type="submit" value="Predict">
    </form>

    {% if prediction %}
        <h3>Predicted Species: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            slen = float(request.form['slen'])
            swid = float(request.form['swid'])
            plen = float(request.form['plen'])
            pwid = float(request.form['pwid'])
            features = np.array([[slen, swid, plen, pwid]])
            pred = model.predict(features)[0]
            prediction = iris.target_names[pred]
        except:
            prediction = "Invalid input! Please enter numeric values."
    return render_template_string(template, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
