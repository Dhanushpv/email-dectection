import pandas as pd
import nltk
import pickle
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load dataset
# The CSV is stored in the `Set` folder in the workspace
data = pd.read_csv("Set/spam.csv")

# Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Use shared preprocess implementation from utils
from utils import preprocess

# Use raw messages (pipeline will call preprocess)
X = data['message'].values
y = data['label']

# Adjust test size and cv to dataset size (avoid too-small test set or CV folds)
n_samples = len(data)
min_class_count = data['label'].value_counts().min()
# ensure there are at least 2 samples in the test set
test_size = max(0.2, 2.0 / n_samples)
# choose cv as min(5, min_class_count) but at least 2
cv = min(5, min_class_count) if min_class_count >= 2 else 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

# Pipeline: TfidfVectorizer (uses our preprocess) + classifier
pipeline = Pipeline([
    ('vect', TfidfVectorizer(preprocessor=preprocess)),
    ('clf', MultinomialNB())
])

# Grid for MultinomialNB
param_grid_nb = {
    'vect__ngram_range': [(1,1), (1,2)],
    'vect__min_df': [1, 2],
    'clf__alpha': [0.1, 0.5, 1.0]
}

# Decide whether we can run GridSearch (need at least 2 samples per class in training set)
min_train_class = min([(y_train == cls).sum() for cls in set(y_train)])
if min_train_class >= 2:
    grid_nb = GridSearchCV(pipeline, param_grid_nb, scoring='f1', cv=min(5, min_train_class), n_jobs=-1)
    print("Training MultinomialNB with GridSearchCV...")
    grid_nb.fit(X_train, y_train)
    print("Best params (NB):", grid_nb.best_params_)
    best_nb = grid_nb.best_estimator_
else:
    print("Too few samples per class in training set; skipping GridSearch for NB and using default pipeline parameters")
    pipeline.fit(X_train, y_train)
    best_nb = pipeline

# Quick try: Logistic Regression (different param grid)
pipeline_lr = Pipeline([
    ('vect', TfidfVectorizer(preprocessor=preprocess)),
    ('clf', LogisticRegression(max_iter=1000))
])
param_grid_lr = {
    'vect__ngram_range': [(1,1), (1,2)],
    'vect__min_df': [1, 2],
    'clf__C': [0.1, 1.0, 10.0]
}

# Decide whether to run GridSearch for LogisticRegression
if min_train_class >= 2:
    grid_lr = GridSearchCV(pipeline_lr, param_grid_lr, scoring='f1', cv=min(5, min_train_class), n_jobs=-1)
    print("Training LogisticRegression with GridSearchCV...")
    grid_lr.fit(X_train, y_train)
    print("Best params (LR):", grid_lr.best_params_)
    best_lr = grid_lr.best_estimator_
else:
    print("Too few samples per class in training set; skipping GridSearch for LR and using default pipeline parameters")
    pipeline_lr.fit(X_train, y_train)
    best_lr = pipeline_lr

# Evaluate both on test set and pick the best by f1
y_pred_nb = best_nb.predict(X_test)
f1_nb = f1_score(y_test, y_pred_nb)

y_pred_lr = best_lr.predict(X_test)
f1_lr = f1_score(y_test, y_pred_lr)

print(f"NB F1 on test: {f1_nb:.4f}")
print(f"LR F1 on test: {f1_lr:.4f}")

if f1_lr > f1_nb:
    best = best_lr
    print("Selected LogisticRegression pipeline")
else:
    best = best_nb
    print("Selected MultinomialNB pipeline")

# Final evaluation
y_pred = best.predict(X_test)
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:\n", accuracy_score(y_test, y_pred))

# Save best pipeline for use by the API
pickle.dump(best, open("spam_pipeline.pkl", "wb"))
# Also overwrite spam_model.pkl for backward compatibility
pickle.dump(best, open("spam_model.pkl", "wb"))

print("Best pipeline trained and saved as 'spam_pipeline.pkl'")
