import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Ruta a la carpeta donde están los archivos CSV
carpeta = r'C:\Users\Francisco\Desktop\TFG\Datos_50GeV\DatosEl_50GeV'

# Crear una matriz de ceros de dimensiones 48x97x97
matrix = np.zeros((48, 97, 97))

# Leer los archivos CSV y actualizar la matriz
hits = 0
for i in range(1, 49):
    nombre_archivo = f'file{i}.csv'
    ruta_archivo = os.path.join(carpeta, nombre_archivo)
    try:
        with open(ruta_archivo, 'r') as file:
            for j, line in enumerate(file):
                if j == 309:
                    pares_coordenadas = line.strip().split(';')
                    for par in pares_coordenadas:
                        if ',' not in par:
                            continue  # Saltar si no hay comas en la línea
                        coordenada_x, coordenada_y = map(int, par.split(','))
                        matrix[i-1, coordenada_x, coordenada_y] = 1
                        hits += 1
    except FileNotFoundError:
        print(f'El archivo {nombre_archivo} no fue encontrado.')
    except Exception as e:
        print(f'Error al procesar el archivo {nombre_archivo}: {str(e)}')

# Funciones para encontrar clusters y sus centroides
def find_neighbors(matrix, row, col):
    neighbors = []
    directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    for dir in directions:
        newRow = row + dir[0]
        newCol = col + dir[1]
        if 0 <= newRow < len(matrix) and 0 <= newCol < len(matrix[0]) and matrix[newRow][newCol] == 1:
            neighbors.append((newRow, newCol))
    return neighbors

def find_clusters(matrix):
    clusters = []
    visited = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if not visited[i][j] and matrix[i][j] == 1:
                cluster = [(i, j)]
                visited[i][j] = True
                k = 0
                while k < len(cluster):
                    row, col = cluster[k]
                    neighbors = find_neighbors(matrix, row, col)
                    for neighbor in neighbors:
                        if not visited[neighbor[0]][neighbor[1]]:
                            cluster.append(neighbor)
                            visited[neighbor[0]][neighbor[1]] = True
                    k += 1
                if len(cluster) > 0:
                    clusters.append(cluster)
    return clusters

def find_centroids(clusters):
    mu = []
    for i in range(len(clusters)):
        centroid = np.zeros(2)
        for j in range(len(clusters[i])):
            centroid += np.array(clusters[i][j])
        mu.append(centroid / len(clusters[i]))
    return mu

allmu = []
allpoints = []
for i in range(len(matrix)):
    clusters = find_clusters(matrix[i])
    if clusters:
        mu = find_centroids(clusters)
        mu = np.array(mu)
        position_of_camara = np.full((mu.shape[0], 1), i)
        mu = np.hstack((position_of_camara, mu))
        allmu.append(mu)
        
        for cluster in clusters:
            for point in cluster:
                allpoints.append([i, point[0], point[1]])

centroids = np.empty((0, 3))
for i in range(len(allmu)):
    centroids = np.vstack((centroids, np.array(allmu[i])))

x = centroids[:, 2]  
y = centroids[:, 0] 
z = centroids[:, 1] 

allpoints = np.array(allpoints)
points_x = allpoints[:, 2]
points_y = allpoints[:, 0]
points_z = allpoints[:, 1]

# Gráfica 3D inicial de los centroides y los puntos originales
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Puntos originales
ax.scatter(points_x, points_y, points_z, c='b', marker='o', alpha=0.5, label='Puntos Originales')

# Centroides
ax.scatter(x, y, z, c='r', marker='x', s=100, label='Centroides')

ax.set_xlabel('Eje X', fontsize=16)  # Aumentado el tamaño de fuente
ax.set_ylabel('Cámara del detector', fontsize=16)  # Aumentado el tamaño de fuente
ax.set_zlabel('Eje Y', fontsize=16)  # Aumentado el tamaño de fuente

# Ajustar el tamaño de los números de los ejes
ax.tick_params(axis='both', which='major', labelsize=14)

ax.set_xlim(40, 90)
ax.set_zlim(40, 80)
ax.legend()
plt.show()

# Comparación y actualización de centroides
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

for j, punto in enumerate(allmu):
    if j < len(allmu) - 1:
        puntos_siguientes = allmu[j + 1]
        vector_distancias = []
        for siguiente_punto in puntos_siguientes:
            distancia = euclidean_distance(punto[0], siguiente_punto)
            vector_distancias.append(distancia)
        min_valor = min(vector_distancias)
        posicion = vector_distancias.index(min_valor)
        allmu[j + 1] = [puntos_siguientes[posicion]]

# Actualización de centroids para visualización final en 3D
centroids = np.vstack(allmu)
x = centroids[:, 2]  
y = centroids[:, 0] 
z = centroids[:, 1] 

# Filtrar los puntos originales que pertenecen a los centroides existentes
allpoints = np.array(allpoints)
filtered_points = []

for point in allpoints:
    camera_id = point[0]
    centroid_camera = centroids[:, 0]
    if camera_id in centroid_camera:
        filtered_points.append(point)

filtered_points = np.array(filtered_points)
points_x = filtered_points[:, 2]
points_y = filtered_points[:, 0]
points_z = filtered_points[:, 1]

# Gráfica 3D con los puntos originales filtrados y los centroides
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Puntos originales filtrados
ax.scatter(points_x, points_y, points_z, c='b', marker='o', alpha=0.5, label='Puntos Originales')

# Centroides
ax.scatter(x, y, z, c='r', marker='x', s=100, label='Centroides filtrados')

ax.set_xlabel('Eje X', fontsize=16)  # Aumentado el tamaño de fuente
ax.set_ylabel('Cámara del detector', fontsize=16)  # Aumentado el tamaño de fuente
ax.set_zlabel('Eje Y', fontsize=16)  # Aumentado el tamaño de fuente

# Ajustar el tamaño de los números de los ejes
ax.tick_params(axis='both', which='major', labelsize=14)

# Establecer límites en los ejes X y Z
ax.set_xlim(40, 90)
ax.set_zlim(40, 80)

ax.legend()
plt.show()

y = centroids[:, 0]  # Coordenada Y (variable independiente)
z = centroids[:, 1]  # Coordenada Z (variable dependiente)

# Calcular el valor medio de x a partir de los centroides
mean_x = np.mean(centroids[:, 2])

# Reorganizar los datos para la regresión
X = y.reshape(-1, 1)  # La coordenada Y como características
y_target = z  # La coordenada Z como objetivo

# Crear el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y_target)

# Obtener los coeficientes de la regresión
a = model.coef_[0]  # Pendiente
c = model.intercept_  # Intersección

# Crear una malla de puntos para visualizar la línea ajustada
y_range = np.linspace(y.min(), y.max(), 100)
z_grid = a * y_range + c

# Visualizar el resultado en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Puntos originales filtrados
ax.scatter(mean_x * np.ones_like(y), y, z, c='b', marker='o', alpha=0.5, label='centroides filtrados')

# Línea ajustada
ax.plot(mean_x * np.ones_like(y_range), y_range, z_grid, c='r', label=f'Línea Ajustada\nPendiente (a): {a:.2f}\nIntersección (c): {c:.2f}', linewidth=2)

ax.set_xlabel('Eje X', fontsize=16)  # Aumentado el tamaño de fuente
ax.set_ylabel('Cámara del detector', fontsize=16)  # Aumentado el tamaño de fuente
ax.set_zlabel('Eje Y', fontsize=16)  # Aumentado el tamaño de fuente

# Ajustar el tamaño de los números de los ejes
ax.tick_params(axis='both', which='major', labelsize=14)


ax.set_xlim(40, 90)
ax.set_zlim(40, 80)

ax.legend()
plt.show()
