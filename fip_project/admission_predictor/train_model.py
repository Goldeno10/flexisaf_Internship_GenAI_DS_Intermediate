import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Load Data
df = pd.read_csv('university_admission.csv')

# CLEAN COLUMN NAMES (Fixes the "Feature names seen at fit time" error)
# This removes leading/trailing spaces and replaces middle spaces with underscores
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Create Target Variable
threshold = 0.60
df['Admitted'] = (df['Chance_of_Admission'] >= threshold).astype(int)

# Define Features (X) and Target (y)
X = df.drop(['Chance_of_Admission', 'Serial_No.', 'Admitted'], axis=1, errors='ignore')
y = df['Admitted']

print(f"Features being used for training: {list(X.columns)}")

# Build and Train the Pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# Save Model & Features
joblib.dump(pipeline, 'admission_pipeline.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')

print("✅ Pipeline trained locally and saved successfully!")
