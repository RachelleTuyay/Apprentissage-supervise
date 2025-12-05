# Apprentissage supervise PROJET

Ce projet vise à comparer les performances de plusieurs classifieurs à partir d’un corpus préalablement annoté, avant d’en analyser et d’en interpréter les résultats.

## Corpus
Le corpus est une transcription de l'oral d'entretiens entre chercheurs et habitants d'Orléans. Il a été annoté à l’aide d’un système binaire d’étiquettes identifiant les questions comme « question canonique » ou « non-question ».

Une question canonique correspond à une demande d'information concrète.
Une non-question, en revanche, ne demande pas réellement d'information. Elle permet plutôt de demander une précision, de clarifier un propos ou bien de réagir à ce qui a été dit.


## Classifieurs / Modèles utilisés
### Apprentissage supervisé de surface : 
- Logistic Regression
- Linear SVM
- Gradient Boosting
- XGBoost
- Multinomial NB (Naives Bayes)
- AdaBoostClassifier
- ExtraTreesClassifier
- BaggingClassifier

Résultats :

| Models | Accuracy | Precision | Recall | F-score | CV (mean)| Ecart-type (std) |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.916667 | 0.920875 | 0.916667 | 0.916458 | 0.827083 | 0.037034 |
| SVM | 0.933333 | 0.920875 | 0.933333 | 0.933036 | 0.864583 | 0.028717
| Random Forest | 0.858333 | 0.866581 | 0.858333 | 0.857532 | 0.793750 | 0.024116 |
| Gradient Boosting | 0.816667 | 0.819865 | 0.816667 | 0.816207 | 0.783333 | 0.056443 |
| XGBoost | 0.816667 | 0.816667 | 0.816667 | 0.816667 | 0.756250 | 0.015590 | 
| Multinomial NB | 0.908333 | 0.922535  | 0.908333 | 0.907557 | 0.829167 | 0.025173 | 
| AdaBoost | 0.766667 | 0.787081 | 0.766667| 0.762443 | 0.702083 | 0.073243 |
| ExtraTreesClassifier |0.900000 | 0.904040 | 0.900000 | 0.899749 | 0.797917 | 0.039308 |
| BaggingClassifier | 0.933333 | 0.937710 | 0.933333 | 0.933166 | 0.822917 | 0.023754 |


### Apprentissage supervisé profonde :
- BERT
- CamemBERT ? Beaucoup trop lourd , alternative :
- FlauBERT
- RNN
- FFNN (Feed foward neural network)

