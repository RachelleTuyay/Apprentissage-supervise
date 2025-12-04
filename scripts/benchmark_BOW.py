import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

# Chargement du corpus + split du jeu de donnée
fichier = "../corpus/corpus_cleaned.xlsx"
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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorisation en Bag of Words (BOW)
vectorizer = CountVectorizer(
    ngram_range=(1, 2),  # unigrams et bigrams
    min_df=1,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Définition des modèles
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=300),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    "Multinomial NB": MultinomialNB(),
    "AdaBoostClassifier": AdaBoostClassifier(n_estimators=100),
    "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=100),
    "BaggingClassifier": BaggingClassifier(estimator=SVC(), n_estimators=10)
}

# Evaluation
results = []

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "CV (mean)": np.mean(cv_scores),
        "Ecart-type": np.std(cv_scores)
    })

# Affichage + sauvegarde des résultats
df_results = pd.DataFrame(results)

print("\n===== TABLEAU RÉCAPITULATIF =====\n")
print(df_results)

# Créer dossier si nécessaire
os.makedirs("results", exist_ok=True)

with open("../results/benchmark_BOW_results.txt", "w", encoding="utf-8") as f:
    f.write(df_results.to_string(index=False))
