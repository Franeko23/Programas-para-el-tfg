import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo Excel y convertirlo en DataFrame
df1 = pd.read_excel('Base de datos 3.xlsx', engine='openpyxl')

# Extraer las columnas necesarias
X = df1[['tamaño', 'Numero de cluster', 'densidad', 'camaras encendidas', 'numero de hits']].values

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

# Elegir el clúster a excluir (por ejemplo, el clúster 0)
cluster_to_exclude = 0

# Filtrar los datos del grupo 0
excluded_hits = df1['numero de hits'][labels == cluster_to_exclude]

# Filtrar los datos completos
all_hits = df1['numero de hits']

# Aplicar el filtro adicional:
# Filtrar datos del grupo 0 donde todas las columnas de la 8 a la 12 tienen un valor >= 5
all_values_5_or_more = (df1.iloc[:, 7:12] >= 5).all(axis=1)
filtered_excluded_hits = excluded_hits[all_values_5_or_more[labels == cluster_to_exclude]]

# Graficar el histograma con el formato deseado
plt.figure(figsize=(12, 8))

# Histograma para datos excluidos (Grupo etiquetado como 0)
n_excluded, bins_excluded = np.histogram(excluded_hits, bins='auto')
n_excluded_half, bins_excluded_half = np.histogram(excluded_hits, bins=len(bins_excluded)//2)

# Graficar la parte superior del histograma para datos excluidos (color negro)
plt.step(0.5 * (bins_excluded_half[:-1] + bins_excluded_half[1:]), n_excluded_half, where='mid', color='black', label='Grupo etiquetado como 0')

# Histograma para datos completos (incluyendo todos los datos)
n_total, bins_total = np.histogram(all_hits, bins='auto')
n_total_half, bins_total_half = np.histogram(all_hits, bins=len(bins_total)//2)

# Graficar la parte superior del histograma para datos completos (color rojo)
plt.step(0.5 * (bins_total_half[:-1] + bins_total_half[1:]), n_total_half, where='mid', color='red', label='Datos completos')

# Histograma para datos del grupo 0 que pasan el filtro adicional
n_filtered, bins_filtered = np.histogram(filtered_excluded_hits, bins='auto')
n_filtered_half, bins_filtered_half = np.histogram(filtered_excluded_hits, bins=len(bins_filtered)//2)

# Graficar la parte superior del histograma para datos filtrados (color verde)
plt.step(0.5 * (bins_filtered_half[:-1] + bins_filtered_half[1:]), n_filtered_half, where='mid', color='green', label='Grupo etiquetado como 0 con condición de penetrabilidad')

# Configurar los ticks del eje x para que vayan de 100 en 100 desde 0 hasta 1100, y estén centrados
plt.xticks(np.arange(0, 1101, 100))

# Establecer los límites del eje x
plt.xlim(0, 1100)

# Configurar el eje y para que esté en escala logarítmica
plt.yscale('log')

# Título y etiquetas de los ejes con tamaño de fuente aumentado
plt.title('Distribución del número de Hits (escala logarítmica)', fontsize=25)
plt.xlabel('Número de Hits', fontsize=20)
plt.ylabel('Frecuencia', fontsize=20)

# Configurar el tamaño de fuente de los ticks
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Añadir la leyenda con tamaño de fuente aumentado
plt.legend(fontsize=18)

# Mostrar el gráfico
plt.show()
