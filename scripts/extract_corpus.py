import polars as pl

# Charger le corpus
fichier = "corpus_simplifie_131025.xlsx"
df = pl.read_excel(fichier)

print("Colonnes disponibles :", df.columns)

# Garder seulement les colonnes souhaitées
colonnes_a_garder = ["question", "type de question (simplifié)"]
df_filtre = df.select(colonnes_a_garder)

print(df_filtre.head())

#Sauvegarder le corpus filtré dans un nouveau fichier
df_filtre.write_excel("corpus_filtre.xlsx")
print("Corpus filtré enregistré sous 'corpus_filtre.xlsx'")
