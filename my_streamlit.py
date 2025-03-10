import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import random  # Ajout de l'import random

# 🔹 Chargement des données avec cache pour éviter le rechargement inutile
@st.cache_data
def load_data():
    return pd.read_csv("/Users/akira/Streamlit/train.csv")  # Assure-toi que le fichier existe

df = load_data()

# 🔥 Interface Streamlit
st.title("Projet de classification binaire - Titanic")
st.sidebar.title("Sommaire")
pages = ["Exploration", "DataVizualization", "Modélisation"]
page = st.sidebar.radio("Aller vers", pages)

# 🌍 Page Exploration
if page == pages[0]:
    st.write("### Introduction")
    st.write("✅ Fichier CSV chargé avec succès !")
    st.dataframe(df.head(10))  # Afficher les 10 premières lignes
    st.write("Dimensions du dataframe :", df.shape)
    st.dataframe(df.describe())  # Statistiques descriptives
    if st.checkbox("Afficher les valeurs manquantes"):
        st.dataframe(df.isna().sum())

# 📊 Page Data Visualization
elif page == pages[1]:
    st.write("### Visualisation des données")

    # Distribution de la variable cible "Survived"
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', data=df, ax=ax)
    plt.title("Distribution de la variable 'Survived'")
    st.pyplot(fig)

    # Répartition du genre des passagers
    fig, ax = plt.subplots()
    sns.countplot(x='Sex', data=df, ax=ax)
    plt.title("Répartition du genre des passagers")
    st.pyplot(fig)

    # Répartition des classes des passagers
    fig, ax = plt.subplots()
    sns.countplot(x='Pclass', data=df, ax=ax)
    plt.title("Répartition des classes des passagers")
    st.pyplot(fig)

    # Distribution de l'âge des passagers
    fig, ax = plt.subplots()
    sns.histplot(x='Age', data=df, kde=True, ax=ax)
    plt.title("Distribution de l'âge des passagers")
    st.pyplot(fig)

    # Countplot de la variable cible en fonction du genre
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', hue='Sex', data=df, ax=ax)
    plt.title("Survie selon le genre")
    st.pyplot(fig)

    # Relation entre classe et survie
    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    fig.fig.suptitle("Survie selon la classe")  # Correction ici
    st.pyplot(fig.fig)

    # Relation entre âge et survie
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    fig.fig.suptitle("Relation entre âge et survie selon la classe")  # Correction ici
    st.pyplot(fig.fig)

# 🤖 Page Modélisation
elif page == pages[2]:
    st.write("### Modélisation")

    # Prétraitement des données
    df_model = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Définir la variable cible et les variables explicatives
    y = df_model['Survived']
    X_cat = df_model[['Pclass', 'Sex', 'Embarked']].copy()
    X_num = df_model[['Age', 'Fare', 'SibSp', 'Parch']].copy()

    # Remplacer les valeurs manquantes
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())

    # Encoder les variables catégorielles
    X_cat_encoded = pd.get_dummies(X_cat, columns=X_cat.columns)

    # Concaténer les variables explicatives encodées et numériques
    X = pd.concat([X_cat_encoded, X_num], axis=1)

    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Standardiser les variables numériques
    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    # Sélection du modèle via une selectbox
    choix = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)

    # Fonction de prédiction
    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'SVC':
            clf = SVC()
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf

    # Entraînement du modèle
    clf = prediction(option)

    # Choix d'affichage des résultats
    display = st.radio('Que souhaitez-vous afficher ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        accuracy = clf.score(X_test, y_test)
        st.write(f"🔹 Précision du modèle : {accuracy:.2%}")
    elif display == 'Confusion matrix':
        cm = confusion_matrix(y_test, clf.predict(X_test))
        st.dataframe(cm)

def generate_random_value(x): 
    return random.uniform(0, x)  # Correction ici

# Générer et afficher des valeurs aléatoires
a = generate_random_value(10) 
b = generate_random_value(20) 
st.write(a) 
st.write(b)