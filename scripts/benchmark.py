import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,f1_score,precision_score,recall_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#Chargement du corpus + split du jeu de donnée
fichier = "corpus_cleaned.xlsx"
df = pd.read_excel(fichier)

X = df["question"].astype(str)
y = df["Intention"].astype(str)

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

y = df["Intention"].astype(int)

print("Valeurs uniques dans y :", y.unique())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Vectorisation
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=300),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "Multinomial NB": MultinomialNB(),
    "AdaBoostClassifier": AdaBoostClassifier(n_estimators=100),
    "ExtraTreesClassifier" : ExtraTreesClassifier(n_estimators=100),
    "BaggingClassifier": BaggingClassifier(estimator=SVC(), n_estimators=10)
}


results = []

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    })

# Affiche le tableau final + Sauvegarde
df_results = pd.DataFrame(results) # Convertir en DataFrame

print("\n===== TABLEAU RÉCAPITULATIF =====\n")
print(df_results)

with open("benchmark_results.txt", "w", encoding="utf-8") as f:
    f.write(df_results.to_string(index=False))
