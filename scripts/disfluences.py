import polars as pl
import re

# Charger le corpus filtré
fichier = "corpus_balanced.csv"
df = pl.read_csv(fichier)

#Supprimer les disfluances du types : hm , euh, hésitations, répétitions, etc....
df= df.with_columns(
    pl.col("question_originale").str.replace(r"(euh|hm*)", "")  #\b\w+-(?=\s|\w) ? pour répét
)

print(df.head())

# Sauvegarder le corpus
df.write_excel("corpus_balanced_cleaned.xlsx")
print("Corpus nettoyé enregistré.")

