import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement du corpus

df = pd.read_excel("corpus_cleaned.xlsx")

df["input_text"] = ( df["question"].astype(str))
df["label"] = df["Intention"].astype("category").cat.codes

labelmap = dict(enumerate(df["Intention"].astype("category").cat.categories))
print("Map label :", labelmap)

dataset = Dataset.from_pandas(df[["input_text","label"]])

# Tokenisation

tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_cased")

def tokenize(batch):
    return tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

# Chargement du modèle

model = AutoModelForSequenceClassification.from_pretrained(
    "flaubert/flaubert_base_cased",
    num_labels=df["label"].nunique()
)

# Configuration de l'entrainement

args = TrainingArguments(
    output_dir="results",
    eval_strategy="epoch", #eval_strategy ou evaluation_strategy selon la version de transformers
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=20,
)

# Métriques

def eval_metric(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels,preds,average="weighted")
    f1 = f1_score(labels,preds,average="weighted")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1" : f1
    }

# Fonction pour la matrice de confusion

def plot_confusion_matrix(cm, labelmap):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True,fmt='d',cmap="Blues", xticklabels=labelmap.values(), yticklabels=labelmap.values())
    plt.xlabel("Prédictions")
    plt.ylabel("Véritable Valeurs")
    plt.title("Matrice de Confusion")
    plt.show()

# Entrainement

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=eval_metric
)

trainer.train()

# Evaluation

metrics = trainer.evaluate()
print("Résultats de l'évaluation FlauBERT:", metrics)

# Matrice de confusion

predictions = trainer.predict(dataset["test"])
labels = predictions.label_ids
preds = np.argmax(predictions.predictions, axis=1)

cm = confusion_matrix(labels, preds)
plot_confusion_matrix(cm, labelmap)
