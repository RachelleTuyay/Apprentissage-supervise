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

# Nettoyage et harmonisation
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
print("Valeurs uniques dans y :", y.unique())

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

# Définition des modèles
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

# Liste de couleurs (low → high) pour chaque modèle
colors = [
    ("#fee5d9", "#de2d26"),  # rouge vif
    ("#eff3ff", "#08519c"),  # bleu foncé
    ("#f7fbf0", "#31a354"),  # vert vif
    ("#fff5eb", "#e6550d"),  # orange vif
    ("#f7f7f7", "#636363"),  # gris foncé
    ("#fce6f2", "#ce1256"),  # rose foncé
    ("#e5f5f9", "#2ca25f"),  # vert-bleu
    ("#fff7f3", "#fb6a4a"),  # saumon
    ("#f7f4f9", "#6a51a3")   # violet foncé
]


# Création du dossier de sortie si inexistant
os.makedirs("matrix", exist_ok=True)

# Fonction pour tracer la matrice
def plot_confusion_matrix(cm, model_name, low_color, high_color):
    cm_df = pd.DataFrame(cm, columns=["Pred_0", "Pred_1"], index=["Réel_0", "Réel_1"])
    cm_df = cm_df.reset_index().melt(id_vars="index")
    cm_df.columns = ["Réel", "Pred", "Count"]

    plot = (
        ggplot(cm_df, aes("Pred", "Réel", fill="Count"))
        + geom_tile()
        + geom_text(aes(label="Count"), color="black", size=12)
        + scale_fill_gradient(low=low_color, high=high_color)
        + ggtitle(f"Matrice de confusion : {model_name}")
        + theme_minimal()
    )

    plot.save(f"matrix/{model_name}_confusion_matrix.png", dpi=150)
    print(f"Matrice enregistrée : {model_name}_confusion_matrix.png")

# Boucle sur les modèles
for i, (name, model) in enumerate(models.items()):
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    cm = confusion_matrix(y_test, y_pred)
    low_color, high_color = colors[i % len(colors)]  # attribue une couleur différente
    plot_confusion_matrix(cm, name, low_color, high_color)
