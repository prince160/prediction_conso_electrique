import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Chargement des données
df = pd.read_csv('C:/Users/pamat/Documents/Doc France/B3  Info Keyce/Projet/Kit game/kit/donnees2.0.csv', sep=';')

# Supprimer les colonnes ne contenant que des NaN
df.dropna(axis=1, how="all", inplace=True)

# Supprimer les lignes avec des NaN
df.dropna(inplace=True)

# Prétraiter les données
scaler = MinMaxScaler()
df[['Heure', 'Jour', 'Mois', 'Année', 'consommation', 'Point_de_rosée', 'Température']] = scaler.fit_transform(df[['Heure', 'Jour', 'Mois', 'Année', 'consommation', 'Point_de_rosée', 'Température']])

# Préparer les données d'entraînement et de test
sequence_length = 6 # Longueur de la séquence pour prédire la consommation
test_size = 0.2 # Proportion de données à utiliser pour le test

X = []
Y = []

for i in range(len(df) - sequence_length):
    X.append(df.iloc[i:i+sequence_length][['Heure', 'Jour', 'Mois', 'Année', 'consommation', 'Point_de_rosée', 'Température']].values)
    Y.append(df.iloc[i+sequence_length]['consommation'])

X = np.array(X)
Y = np.array(Y)

# Séparer les données en ensembles d'entraînement et de test
train_size = int(len(X) * (1 - test_size))
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

# Créer le modèle de réseau de neurones
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Entraîner le modèle
model.fit(X_train, Y_train, epochs=5, batch_size=16, verbose=1)

# Faire les prédictions pour les données de test
predictions = model.predict(X_test)

# Ajouter une dimension à la prédiction
predictions = predictions.reshape(predictions.shape[0], 1)

# Dénormaliser les prédictions
consumption_min = scaler.data_min_[4]
consumption_max = scaler.data_max_[4]
predictions_denormalized = predictions * (consumption_max - consumption_min) + consumption_min

# Calculer la MAE
mae = mean_absolute_error(Y_test, predictions_denormalized)

# Calculer la fiabilité (R²)
SS_res = np.sum((Y_test - predictions.squeeze())**2)
SS_tot = np.sum((Y_test - np.mean(Y_test))**2)
R_squared = 1 - (SS_res / SS_tot)

##NOMBRE DE PREDICTION DE SEQUENCES DE 3H##
nbr = 1

# Afficher les résultats
print("Prévisions :")
for i in range(nbr):
    print("Temps", i+1, ":", predictions_denormalized[i][0])
print("MAE :", mae)
print("Fiabilité (R²) :", R_squared)

# Créer le graphique
plt.scatter(Y_test, X_test[:, -1, -1], label="Données réelles")
plt.scatter(predictions, X_test[:, -1, -1], label="Prévisions")
plt.xlabel("Consommation")
plt.ylabel("Température")
plt.legend()
plt.show()