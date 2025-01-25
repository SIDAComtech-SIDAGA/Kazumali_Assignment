import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
import joblib
import os

file_path = "data/student-por.csv"
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

df = pd.read_csv(file_path)
print("Data loaded successfully. First 5 rows:")
print(df.head())
print(df.columns)

# Separate features and target
X = df.drop("G3", axis=1)
y = df["G3"]

# Handle cat
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
model_path = "student_performance_model.pkl"
joblib.dump((model, X.columns), "student_performance_model.pkl")

print(f"Trained model saved at {model_path}")
