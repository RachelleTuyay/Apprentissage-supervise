import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

# Chargement des données

df = pd.read_excel("../corpus/corpus_cleaned.xlsx")
texts = df["question"].astype(str).values
labels = df["Intention"].astype(str).values

# Encodage binaire : 1 = question canonique, 0 = non question
y = (labels == "question canonique").astype(int)

# Configuration du Modèle et des Données


max_words = 5000 
maxlen = 40
embedding_dim = 128  
lstm_units = 64 
batch_size = 16
epochs = 50

# Early stopping

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# Évaluation initiale (pour l'affichage des métriques détaillées)

# Train / test split initial
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42, # Utilisé pour les métriques détaillées
    stratify=y
)

# Tokenisation + séquences (fit sur X_train)
tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<UNK>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = keras.preprocessing.sequence.pad_sequences(
    X_train_seq, maxlen=maxlen, padding="post", truncating="post"
)
X_test_pad = keras.preprocessing.sequence.pad_sequences(
    X_test_seq, maxlen=maxlen, padding="post", truncating="post"
)

vocab_size = len(tokenizer.word_index) + 1 

# Modèle CONV1D 

model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
    layers.Conv1D(filters=128, kernel_size=5, activation='relu'), # Ajout de Conv1D
    layers.GlobalMaxPooling1D(), 
    layers.Dropout(0.5),                 
    layers.Dense(32, activation="relu"), 
    layers.Dense(1, activation="sigmoid")
])


# Ajustement du Taux d'Apprentissage pour plus de stabilité
optimizer = keras.optimizers.Adam(learning_rate=0.0005) 

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Entraînement

print("Entraînement Initial (random_state=42)")
history = model.fit(
    X_train_pad,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Évaluation

test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=0)
print("Évaluation initiale (random_state=42)")
print("Test loss:", test_loss)
print("Test accuracy (Keras):", test_acc)

# Calcul de la moyenne et de l'écart-type de l'accuracy

num_runs = 10
accuracies = []

print(f"Exécution de l'évaluation {num_runs} fois pour la moyenne et l'écart-type")

for i in range(num_runs):
    print(f"Lancement de l'exécution {i+1}/{num_runs}...")
    
    # 1. Nouvelle division train/test avec un random_state différent
    X_train_run, X_test_run, y_train_run, y_test_run = train_test_split(
        texts,
        y,
        test_size=0.2,
        random_state=i, 
        stratify=y
    )

    # 2. Tokenisation + séquences
    tokenizer_run = keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<UNK>")
    tokenizer_run.fit_on_texts(X_train_run)

    X_train_seq_run = tokenizer_run.texts_to_sequences(X_train_run)
    X_test_seq_run = tokenizer_run.texts_to_sequences(X_test_run)

    X_train_pad_run = keras.preprocessing.sequence.pad_sequences(
        X_train_seq_run, maxlen=maxlen, padding="post", truncating="post"
    )
    X_test_pad_run = keras.preprocessing.sequence.pad_sequences(
        X_test_seq_run, maxlen=maxlen, padding="post", truncating="post"
    )

    vocab_size_run = len(tokenizer_run.word_index) + 1 
    
    # 3. Réinitialisation du modèle (Conv1D)
    model_run = keras.Sequential([
        layers.Embedding(input_dim=vocab_size_run, output_dim=embedding_dim, input_length=maxlen),
        layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.5),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    optimizer_run = keras.optimizers.Adam(learning_rate=0.0005) 
    
    model_run.compile(
        optimizer=optimizer_run,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Entraînement
    model_run.fit(
        X_train_pad_run,
        y_train_run,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    # 5. Évaluation et stockage de l'accuracy
    _, test_acc_run = model_run.evaluate(X_test_pad_run, y_test_run, verbose=0)
    accuracies.append(test_acc_run)
    print(f"Résultat Exécution {i+1}: Accuracy = {test_acc_run:.4f}")

# 6. Calcul de la moyenne et de l'écart-type
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print(f"Accuracy Moyenne ({num_runs} exécutions): {mean_acc:.4f}")
print(f"Écart-type de l'Accuracy ({num_runs} exécutions): {std_acc:.4f}")

# Métriques détaillées (basées sur l'évaluation initiale)

# Probabilités sur le test initial (random_state=42)
y_prob = model.predict(X_test_pad)[:, 0]

# Binarisation
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# Métriques
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("Métriques détaillées (basées sur random_state=42)")
print("Accuracy (sklearn):", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("ROC AUC:", roc_auc)
print("PR AUC:", pr_auc)
print("Confusion matrix:\n", cm)
print("\nClassification report:\n",
      classification_report(y_test, y_pred, target_names=["non question", "question canonique"], zero_division=0))