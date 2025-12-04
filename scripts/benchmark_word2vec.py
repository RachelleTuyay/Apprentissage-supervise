import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Chargement du corpus
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

# Prétraitement pour Word2Vec (tokenization simple)
X_train_tokens = [simple_preprocess(text) for text in X_train]
X_test_tokens  = [simple_preprocess(text) for text in X_test]

# Entraînement du modèle Word2Vec
w2v_model = Word2Vec(
    sentences=X_train_tokens,
    vector_size=100,  # dimension des vecteurs
    window=5,
    min_count=1,
    workers=4,
    sg=1  # skip-gram
)

# Fonction pour transformer une phrase en vecteur en moyennant les mots
def sentence_to_vec(tokens, model, vector_size):
    vec = np.zeros(vector_size)
    count = 0
    for token in tokens:
        if token in model.wv:
            vec += model.wv[token]
            count += 1
    if count > 0:
        vec /= count
    return vec

# Transformation des datasets en vecteurs
X_train_vec = np.array([sentence_to_vec(tokens, w2v_model, 100) for tokens in X_train_tokens])
X_test_vec  = np.array([sentence_to_vec(tokens, w2v_model, 100) for tokens in X_test_tokens])

# Définition des modèles
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=300),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    "Multinomial NB": MultinomialNB(),  # attention : MultinomialNB n'aime pas les vecteurs continus
    "AdaBoostClassifier": AdaBoostClassifier(n_estimators=100),
    "ExtraTreesClassifier" : ExtraTreesClassifier(n_estimators=100),
    "BaggingClassifier": BaggingClassifier(estimator=SVC(), n_estimators=10)
}

# Evaluation
results = []

for name, model in models.items():
    # Skip MultinomialNB car il ne supporte pas les vecteurs continus
    if name == "Multinomial NB":
        continue

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

# Affichage + sauvegarde
df_results = pd.DataFrame(results)

print("\n===== TABLEAU RÉCAPITULATIF =====\n")
print(df_results)

# Créer dossier si nécessaire
os.makedirs("results", exist_ok=True)

with open("../results/benchmark_Word2Vec_results.txt", "w", encoding="utf-8") as f:
    f.write(df_results.to_string(index=False))
