import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score,f1_score,precision_score,recall_score)
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
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
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=300),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    "Multinomial NB": MultinomialNB(),
    "AdaBoostClassifier": AdaBoostClassifier(n_estimators=100),
    "ExtraTreesClassifier" : ExtraTreesClassifier(n_estimators=100),
    "BaggingClassifier": BaggingClassifier(estimator=SVC(), n_estimators=10)
}


results = []

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    cv_scores = cross_val_score(model,X_train_vec, y_train, cv=5)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "CV (mean)": np.mean(cv_scores),
        "CV (std)": np.std(cv_scores)
    })

# Affiche le tableau final + Sauvegarde
df_results = pd.DataFrame(results) # Convertir en DataFrame

print("\n===== TABLEAU RÉCAPITULATIF =====\n")
print(df_results)

with open("results_with_contexts.txt", "w", encoding="utf-8") as f:
    f.write(df_results.to_string(index=False))

