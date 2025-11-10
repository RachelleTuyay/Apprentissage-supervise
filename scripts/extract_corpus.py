import polars as pl

# Charger le corpus
fichier = "corpus_simplifie_201025.xlsx"
df = pl.read_excel(fichier)

# Garder seulement les colonnes souhaitées
colonnes_a_garder = ["question", "Intention"]
df_filtre = df.select(colonnes_a_garder)

print(df_filtre.head())

#Sauvegarder le corpus filtré dans un nouveau fichier
df_filtre.write_excel("corpus_filtre.xlsx")
print("Corpus filtré enregistré sous 'corpus_filtre.xlsx'")
