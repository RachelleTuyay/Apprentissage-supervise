# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


#Chargement du corpus
fichier = "../corpus/corpus_cleaned.xlsx"
df = pd.read_excel(fichier)

# Textes et labels
X = df["question"].astype(str)
y = df["Intention"].astype(str)

# Nettoyage et harmonisation des labels
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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Vectorisation TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)



# Conversion en Tensors PyTorch
X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_vec.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)




#Création d'un modèle MLP en utilisant nn.Sequential() :
input_size = X_train_tensor.shape[1]

model = nn.Sequential(
    nn.Linear(input_size, 128), #input layer en hidden layer
    nn.ReLU(),  #fonction ReLU
    nn.Linear(128, 64), #hidden layer 1 en hidden layer 2
    nn.ReLU(),
    nn.Linear(64, 1)  #hidden layer 2 en output
)

# Loss and Optimizer :
criterion = nn.BCEWithLogitsLoss()  # combine sigmoid + binary cross-entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)


#Entraînement :
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



#Évaluation :
model.eval()
with torch.no_grad():
    y_pred_logits = model(X_test_tensor)
    y_pred_prob = torch.sigmoid(y_pred_logits)
    y_pred = (y_pred_prob >= 0.5).int()  # Seuil 0.5

accuracy = accuracy_score(y_test_tensor, y_pred)
print(f"\nAccuracy sur le test set: {accuracy:.4f}")
