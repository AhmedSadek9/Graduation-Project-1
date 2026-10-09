import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load the generated dataset
data_file = "wanas_augmented_dataset.csv"
df = pd.read_csv(data_file)

X = df["text"]
y = df["intent"]

# 2. Split dataset for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build lightweight, offline ML pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=200))
])

# 4. Train model
print("Training Intent Classification model...")
pipeline.fit(X_train, y_train)

# 5. Evaluate and display metrics
y_pred = pipeline.predict(X_test)
print("\nModel Evaluation Report:")
print(classification_report(y_test, y_pred))

# 6. Save model to disk
model_path = "intent_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print(f"\nModel successfully saved to: {model_path}")