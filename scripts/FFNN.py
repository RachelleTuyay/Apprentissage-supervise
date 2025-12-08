import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from plotnine import *
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
import os

# Chargement du corpus
fichier = "../corpus/corpus_cleaned.xlsx"
df = pd.read_excel(fichier)

# Textes et labels
X = df["question"].astype(str)
y = df["Intention"].astype(str)

# Nettoyage des labels
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

#Split du dataset
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

# Conversion en Tensors
X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_vec.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)


#Création d'un modèle MLP "nn.Sequential()"
input_size = X_train_tensor.shape[1]

def build_model(): #source : https://www.datacamp.com/tutorial/feed-forward-neural-networks-explained
    return nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

model = build_model()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#cross validation PyToch (en k-folds)
def pytorch_cross_val(X, y, k=5, epochs=50):
    """Retourne accuracy moyen sur k folds en PyTorch."""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n🔹 Fold {fold+1}/{k}")

        # Split
        X_train_f = X[train_idx]
        y_train_f = y[train_idx]
        X_val_f   = X[val_idx]
        y_val_f   = y[val_idx]

        # Création modèle neuf
        model_cv = build_model()
        optimizer_cv = optim.Adam(model_cv.parameters(), lr=0.001)

        # Training
        for epoch in range(epochs):
            model_cv.train()
            optimizer_cv.zero_grad()
            outputs = model_cv(X_train_f)
            loss = criterion(outputs, y_train_f)
            loss.backward()
            optimizer_cv.step()

        # Validation
        model_cv.eval()
        with torch.no_grad():
            val_logits = model_cv(X_val_f)
            val_pred = (torch.sigmoid(val_logits) >= 0.5).int()
            acc = accuracy_score(y_val_f.numpy(), val_pred.numpy())
            accuracies.append(acc)

        print(f"   Fold Accuracy = {acc:.4f}")

    print("\n✅ Cross-validation Accuracy mean =", np.mean(accuracies))
    return np.array(accuracies)


cv_scores = pytorch_cross_val(X_train_tensor, y_train_tensor, k=5, epochs=50)



#Entrainement du modèle
EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

print("\nTrain set distribution :")
print(y_train.value_counts())
print("\nTest set distribution :")
print(y_test.value_counts())


#Evaluation
os.makedirs("evaluation_outputs", exist_ok=True)

model.eval()
with torch.no_grad():
    y_pred_logits = model(X_test_tensor)
    y_pred_prob = torch.sigmoid(y_pred_logits).numpy().flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)



accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_prob)

results_df = pd.DataFrame([{
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "CV-mean": cv_scores.mean(),
    "CV-std": cv_scores.std(),
    "AUC-ROC": auc
}])

print(results_df)
with open("results/metrics_results_FFNN.txt", "w", encoding="utf-8") as f:
    f.write(results_df.to_string(index=False))


#Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, columns=["Prédit:0", "Prédit:1"])
cm_df["Réel"] = ["0", "1"]
cm_df = cm_df.melt(id_vars="Réel", var_name="Prédit", value_name="Count")

plot_cm = (
    ggplot(cm_df, aes(x="Prédit", y="Réel", fill="Count"))
    + geom_tile()
    + geom_text(aes(label="Count"), color="white", size=12)
    + scale_fill_gradient(low="#6baed6", high="#08306b")
    + labs(title="Matrice de confusion (FFN)")
    + theme_minimal()
)
plot_cm.save("results/confusion_matrix_FFNN.png", dpi=300)
print("Matrice de confusion sauvegardée")



#Courbe AUC-ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})

plot_roc = (
    ggplot(roc_df, aes(x="FPR", y="TPR"))
    + geom_line(size=1.5)
    + geom_abline(linetype="dashed")
    + labs(title=f"Courbe ROC (AUC = {auc:.4f})",
           x="Taux de faux positifs (FPR)",
           y="Taux de vrais positifs (TPR)")
    + theme_minimal()
)
plot_roc.save("results/ROCcurve_FFNN.png", dpi=300)
print("Courbe ROC sauvegardée")
