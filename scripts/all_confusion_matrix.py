import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from plotnine import *


# Chargement du corpus
fichier = "../corpus/corpus_cleaned.xlsx"
df = pd.read_excel(fichier)

X = df["question"].astype(str)
y = df["Intention"].astype(str)

df["Intention"] = df["Intention"].str.lower().str.strip()
df["Intention"] = df["Intention"].replace({
    "question canonique": 1,
    "canonique": 1,
    "non-question": 0,
    "non question": 0,
    "non": 0,
    "0": 0
})

y = df["Intention"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# Modèles
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

# Liste de couleurs (pour chaque modèle)
colors = [
    ("#fef0d9", "#b30000"),
    ("#edf8fb", "#006d2c"),
    ("#fff7fb", "#7a0177"),
    ("#f7fcf5", "#00441b"),
    ("#f7f4f9", "#54278f"),
    ("#fff5f0", "#cb181d"),
    ("#f7fcfd", "#0868ac"),
    ("#fff5eb", "#d94801"),
    ("#f7f7f7", "#252525")
]

# Préparer le DataFrame global pour toutes les matrices
all_cm_df = pd.DataFrame()

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm, columns=["Pred_0", "Pred_1"], index=["Réel_0", "Réel_1"])
    cm_df = cm_df.reset_index().melt(id_vars="index")
    cm_df.columns = ["Réel", "Pred", "Count"]
    cm_df["Model"] = name
    cm_df["low_color"] = colors[i % len(colors)][0]
    cm_df["high_color"] = colors[i % len(colors)][1]

    all_cm_df = pd.concat([all_cm_df, cm_df], axis=0)

# ============================
# Plot avec facets (une matrice par modèle)
# ============================
# On utilise fill=Count pour heatmap et facet_wrap pour créer un sous-graphique par modèle
plot = (
    ggplot(all_cm_df, aes("Pred", "Réel", fill="Count"))
    + geom_tile()
    + geom_text(aes(label="Count"), color="black", size=10)
    + facet_wrap("~Model")
    + scale_fill_gradient(low="#f0f0f0", high="#2b8cbe")  # couleur par défaut (appliquée uniformément)
    + theme_minimal()
    + ggtitle("Matrices de confusion pour tous les modèles")
)

# Créer dossier si nécessaire
os.makedirs("matrix", exist_ok=True)

plot.save("matrix/all_models_confusion_matrix.png", dpi=150)
print("Figure enregistrée : matrix/all_models_confusion_matrix.png")
