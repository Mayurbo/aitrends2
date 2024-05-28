# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('credit-card-default.csv')
df = df.drop('ID', axis=1)

# Split data
X = df.drop('defaulted', axis=1)
y = df['defaulted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Train model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Predictions
predictions = rfc.predict(X_test)

# Evaluation
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
