import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, BaggingClassifier
)
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack


#Chargement du corpus
fichier = "corpus_cleaned.xlsx"
df = pd.read_excel(fichier)

questions = df["question"].astype(str)
context_left = df["previous_context"].astype(str)
context_right = df["next_context"].astype(str)

# Nettoyage + harmonisation des labels
df["Intention"] = df["Intention"].astype(str).str.lower().str.strip()
df["Intention"] = df["Intention"].replace({
    "question canonique": 1,
    "canonique": 1,
    "non-question": 0,
    "non question": 0,
    "non": 0,
    "0": 0
})
labels = df["Intention"].astype(int)


#Split du jeu de donnée
X_train_left, X_test_left, \
X_train_q, X_test_q, \
X_train_right, X_test_right, \
y_train, y_test = train_test_split(
    context_left, questions, context_right, labels,
    test_size=0.2, random_state=42, stratify=labels
)


#Vectorisation
vectorizer_left = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
vectorizer_q = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
vectorizer_right = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)

#partie train
X_train_left_vec = vectorizer_left.fit_transform(X_train_left)
X_train_right_vec = vectorizer_right.fit_transform(X_train_right)
X_train_q_vec = vectorizer_q.fit_transform(X_train_q)

#partie test
X_test_left_vec = vectorizer_left.transform(X_test_left)
X_test_right_vec = vectorizer_right.transform(X_test_right)
X_test_q_vec = vectorizer_q.transform(X_test_q)

#Concaténation de train et test sous la forme : [context_left , question , context_right]
X_train_vec = hstack([X_train_left_vec,X_train_q_vec,X_train_right_vec])
X_test_vec  = hstack([X_test_left_vec,X_test_q_vec,X_test_right_vec])
#notes: hstack = horizontal stack, pour creer une grande matrice

#print(X_train_vec.shape)
#print(X_test_vec.shape)

#MODELS
models_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=500),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"]
        }
    },
    "Linear SVM": {
        "model": LinearSVC(max_iter=5000),
        "params": {
            "C": [0.01, 0.1, 1, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    },
    "Multinomial NB": {
        "model": MultinomialNB(),
        "params": {
            "alpha": [0.5, 1.0, 1.5]
        }
    },
    "AdaBoostClassifier": {
        "model": AdaBoostClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1]
        }
    },
    "ExtraTreesClassifier": {
        "model": ExtraTreesClassifier(),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20]
        }
    },
    "BaggingClassifier": {
        "model": BaggingClassifier(estimator=SVC()),
        "params": {
            "n_estimators": [5, 10, 20]
        }
    }
}


results = []

# Boucle sur chaque modèle
for name, mp in models_params.items():
    print(f"\n===== {name} =====")
    grid = GridSearchCV(mp["model"], mp["params"], cv=3, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_train_vec, y_train)

    best_model = grid.best_estimator_
    print("Meilleurs paramètres :", grid.best_params_)

    y_pred = best_model.predict(X_test_vec)

    # Stockage résultats
    results.append({
        "Model": name,
        "Best Params": grid.best_params_,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    })

# Tableau final
df_results = pd.DataFrame(results)
print("\n===== TABLEAU RÉCAPITULATIF =====\n")
print(df_results)

# Sauvegarde
df_results.to_excel("results_with_contexts_gridsearch.xlsx", index=False)


