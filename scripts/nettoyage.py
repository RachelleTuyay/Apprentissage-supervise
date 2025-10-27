import polars as pl

# Charger le corpus filtré
fichier = "corpus_filtre.xlsx"
df = pl.read_excel(fichier)

# Nettoyer la colonne "question" en enlevant les balises #spk1: et #spk2:
df = df.with_columns(
    pl.col("question")
    .str.replace_all(r"#spk1:", "")  # supprime #spk1:
    .str.replace_all(r"#spk2:", "")  # supprime #spk2:
    .alias("question")
)

#Supprimer les espaces en trop autour du texte
df = df.with_columns(
    pl.col("question").str.strip_chars().alias("question")
)

print(df.head())

# Sauvegarder le corpus
df.write_excel("corpus_nettoye.xlsx")
print("Corpus nettoyé enregistré sous 'corpus_nettoye.xlsx'")
