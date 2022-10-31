# Import des differents modules necessaires
import numpy as  np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from streamlit.elements.button import ButtonSerde
import streamlit as st

# Lecture du fichier
data = pd.read_csv('iris.csv')

# entrainement par les donnees
knn = KNeighborsClassifier(n_neighbors=5)
X = data.drop("species", axis=1)
y = data["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
sns.pairplot(data, hue = 'species')
sns.set_theme(style="ticks")
knn.fit(X_train, y_train)

# Titres de mon fichier
st.title('Quel iris suis-je ?')
st.subheader('Choisissez vos paramètres !')

# Sliders min - max
longueur_sepale = st.slider("Choisissez la longueur du sépale", 4.0, 8.0)
largeur_sepale = st.slider("Choisissez la largeur du sépale", 2.0, 5.0)
longueur_petale = st.slider("Choisissez la longueur du pétale", 1.0, 8.0)
largeur_petale = st.slider("Choisissez la largeur du pétale", 0.0, 3.0)

# Espece de l'utilisateur
data_utilisateur = [longueur_sepale, largeur_sepale, longueur_petale, largeur_petale]
espece_utilisateur = knn.predict([data_utilisateur])

# choix utilisateur avec utilisation de dictionnaire et graphique
point_en_cours = {
        'sepal_length': [longueur_sepale],
        "sepal_width" : [largeur_sepale],
        "petal_length" : [longueur_petale],
        "petal_width": [largeur_petale],
        "species": "votre espèce est ici"
        }


df_choix_utilisateur = pd.DataFrame(point_en_cours)
df_final = pd.concat([data, df_choix_utilisateur], axis=0)

# Affichage du bouton Validay
if st.button("It's validay ?"):
    st.success("Awesome, you are an amazing : " + espece_utilisateur[0])
    st.pyplot(sns.pairplot(df_final, x_vars=["petal_length"], y_vars=["petal_width"], hue="species", markers=["o", "s", "D", "p"]))
    st.pyplot(sns.pairplot(df_final, x_vars=["sepal_length"], y_vars=["sepal_width"], hue="species", markers=["o", "s", "D", "p"]))
    st.write('Well done ma boi')
    st.write('Arriverderci')
