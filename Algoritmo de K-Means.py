import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo Excel y convertirlo en DataFrame
df1 = pd.read_excel('Base de datos 2.xlsx', engine='openpyxl')

# Extraer las columnas necesarias
X = df1[[ 'Numero de cluster', 'densidad', 'camaras encendidas','numero de hits']].values

K = 3  # Número de clusters

# Calcular los valores máximos y mínimos para inicializar los centroides
min_vals = np.min(X, axis=0)
max_vals = np.max(X, axis=0)

np.random.seed(1492)

# Inicializar los centroides aleatoriamente
mu = np.array([np.random.uniform(min_vals, max_vals) for _ in range(K)])

# Definir la matriz R para la asignación de clusters
R = np.zeros((X.shape[0], K))

# Algoritmo de K-Medias
J_plot = []
J_prev = 0

for i in range(10):  # Iteraciones
    J = 0
    
    # Actualización de R
    for n in range(X.shape[0]):
        x = X[n, :]
        distances = np.linalg.norm(x - mu, axis=1)**2
        k_r = np.argmin(distances)
        R[n, :] = 0
        R[n, k_r] = 1
    
    # Calcular la función objetivo J
    J = np.sum(R * np.linalg.norm(X[:, np.newaxis] - mu, axis=2)**2)
    J_plot.append(J)
    
    # Actualización de centroides
    for k in range(K):
        if np.sum(R[:, k]) > 0:
            mu[k] = np.sum(X * R[:, k][:, np.newaxis], axis=0) / np.sum(R[:, k])
    
    if i > 0 and abs(J - J_prev) < 0.2:
        break
    J_prev = J

# Asignar etiquetas de cluster a los datos
labels = np.argmax(R, axis=1)

# Colores específicos para cada cluster
colors = ['purple', 'red', 'green']

# Graficar Número de clusters vs Densidad
plt.figure()
for k in range(K):
    plt.scatter(df1['Numero de cluster'][labels == k], df1['densidad'][labels == k], 
                color=colors[k], label=f'Cluster {k}', alpha=0.7)
plt.xlabel('Número de cluster', fontsize=20)
plt.ylabel('Densidad', fontsize=20)

plt.legend(title='Cluster', title_fontsize='20', fontsize='18')
plt.show()


