import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_excel("../corpus/corpus_cleaned.xlsx")

df["input_text"] = ( df["question"].astype(str))

df["label"] = df["Intention"].astype("category").cat.codes
labelmap = dict(enumerate(df["Intention"].astype("category").cat.categories))
print("Map label :", labelmap)

dataset = Dataset.from_pandas(df[["input_text","label"]])

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize(batch):
    return tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=df["label"].nunique()
)

args = TrainingArguments(
    output_dir="results",
    eval_strategy="epoch", #evaluation_strategy ou eval_strategy selon la version de transformers
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_steps=20,
)

def eval_metric(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels,preds),
        "f1" : f1_score(labels,preds, average="weighted")
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=eval_metric
)

trainer.train()
metrics = trainer.evaluate()
print("Résultats de l'évaluation BERT:", metrics)
