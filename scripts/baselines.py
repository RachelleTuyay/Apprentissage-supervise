import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lazypredict.Supervised import LazyClassifier
from nltk.corpus import stopwords
import nltk

#stopwords français avec nltk
nltk.download("stopwords")
french_stopwords = stopwords.words("french")

#Chargement du corpus
fichier = "corpus_cleaned.xlsx"
df = pl.read_excel(fichier)
df_pd = df.to_pandas()

#X et Y
X_text = df_pd["question"].astype(str)
y = df_pd["Intention"].astype(str)

# Vérifier la distribution initiale des classes
#print("Distribution initiale des classes :")
#print(y.value_counts(), "\n")

#Vectorisation
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words=french_stopwords
)
X = vectorizer.fit_transform(X_text)

#Split train/test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y  
)

# Vérifier la distribution après split
#print("Classes dans le jeu d'entraînement :")
#print(y_train.value_counts(), "\n")
#print("Classes dans le jeu de test :")
#print(y_test.value_counts(), "\n")

#Conversion en DF Pandas
X_train_df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
X_test_df = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_df, X_test_df, y_train, y_test)

#Sauvegarde des résultats dans un fichier txt
models.to_string("baselines_results.txt", index=True)
print("\nRésultats enregistrés dans 'baselines_results.txt'")
