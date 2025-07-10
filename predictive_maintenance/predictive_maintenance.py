import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Simulate sensor data for a nuclear plant pump
np.random.seed(42)
n_samples = 1000
data = {
    'vibration': np.random.normal(0.5, 0.1, n_samples),
    'temperature': np.random.normal(60, 5, n_samples),
    'pressure': np.random.normal(100, 10, n_samples),
    'failure': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 10% failure rate
}
df = pd.DataFrame(data)

# Preprocess data
X = df[['vibration', 'temperature', 'pressure']]
y = df['failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'pump_failure_model.pkl')

# Example prediction
sample = np.array([[0.6, 65, 105]])
prediction = model.predict_proba(sample)
print(f"Failure probability: {prediction[0][1]:.2f}")
