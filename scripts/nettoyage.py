import polars as pl

# Charger le corpus filtré
fichier = "corpus_filtred.xlsx"
df = pl.read_excel(fichier)

# Nettoyer la colonne "question" en enlevant les balises #spk1: et #spk2:
df = df.with_columns(
    pl.col("question")
    .str.replace_all(r"#spk1:", "")  # supprime #spk1:
    .str.replace_all(r"#spk2:", "")  # supprime #spk2:
    .str.replace_all(r"#spk3:", "")  # supprime #spk3:
    .str.replace_all(r"#spk5:", "")  # supprime #spk5:
    .alias("question")
)

# Supprimer les disfluences du type : hm, euh, hésitations, répétitions, etc.
df = df.with_columns(
    pl.col("question")
    .str.replace(r"\b(euh|hm+)\b", "", literal=False)  # euh, hm...
)

#Supprimer les espaces en trop autour du texte
df = df.filter(
    pl.col("Intention").is_not_null()
    & (pl.col("Intention").str.strip_chars() != "")
)
print(df.head())

# Sauvegarder le corpus
df.write_excel("corpus_cleaned.xlsx")
print("Corpus nettoyé enregistré.")
