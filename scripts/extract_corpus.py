import polars as pl

# Charger le corpus
fichier = "corpus_augmente_v2.xlsx"
df = pl.read_excel(fichier)

# Garder seulement les colonnes souhaitées : question + label et les contextes
colonnes_a_garder = ["question", "Intention","previous_context", "next_context"]
df_filtre = df.select(colonnes_a_garder)

print(df_filtre.head())

#Sauvegarder le corpus filtré dans un nouveau fichier
df_filtre.write_excel("corpus_filtred.xlsx")
print("Corpus filtré enregistré sous 'corpus_filtred.xlsx'")
