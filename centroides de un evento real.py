import numpy as np

import os

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import Axes3D

# Ruta a la carpeta donde están los archivos CSV

carpeta = r'C:\Users\Francisco\Desktop\TFG\Datos_50GeV\DatosEl_50GeV'

# Crear una matriz de ceros de dimensiones 48x97x97

matrix = np.zeros((15, 97, 97))

# Leer los archivos CSV y actualizar la matriz

for i in range(1, 49):
    
    nombre_archivo = f'file{i}.csv'
    
    ruta_archivo = os.path.join(carpeta, nombre_archivo)
    
    try:
        
        with open(ruta_archivo, 'r') as file:
            
            for j, line in enumerate(file):
                
                if j == 60397 :
                    
                    visitado = set()
                    
                    pares_coordenadas = line.strip().split(';')
                    
                    for par in pares_coordenadas:
                        
                        if ',' not in par:
                            
                            continue  # Saltar si no hay comas en la línea
                            
                        coordenada_x, coordenada_y = map(int, par.split(','))
                        
                        matrix[i-1, coordenada_x, coordenada_y] = 1
                        
    except FileNotFoundError:
        
        print(f'El archivo {nombre_archivo} no fue encontrado.')
        
    except Exception as e:
        
        print(f'Error al procesar el archivo {nombre_archivo}: {str(e)}')
        
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
    
    number_of_cluster = 0
    
    visited = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

    for i in range(len(matrix)):
        
        for j in range(len(matrix[0])):
            
            if not visited[i][j] and matrix[i][j] == 1:
                
                cluster = [(i, j)]
                
                visited[i][j] = True
                
                number_of_cluster = number_of_cluster + 1

                # Búsqueda de vecindario usando DFS (Depth-First Search).
                k = 0
                
                while k < len(cluster):
                    
                    row, col = cluster[k]
                    
                    neighbors = find_neighbors(matrix, row, col)
                    
                    for neighbor in neighbors:
                        
                        if not visited[neighbor[0]][neighbor[1]]:
                            
                            cluster.append(neighbor)
                            
                            visited[neighbor[0]][neighbor[1]] = True
                    k += 1

                # Si el tamaño del cluster es mayor o igual a 1, se agrega a la lista de clusters.
                if len(cluster) > 0:
                    
                    clusters.append(cluster)

    return clusters

def find_centroids(clusters):
    
    mu = []
        
    for i in range(len(clusters)):
            
        centroid = np.zeros(2)
        
        cont = 0
            
        for j in range(len(clusters[i])):
            
          centroid = centroid + np.array(clusters[i][j])
          
          cont = cont + 1
          
        lista = centroid/ cont
      
        mu.append([lista[1], lista[0]])
        
    mu = np.array(mu)

    return mu

allmu = []

allclusters = []

for i in range(len(matrix)):
    
    clusters = find_clusters(matrix[i])

    mu = find_centroids(clusters)
    
    plt.figure()


    for j, cluster in enumerate(clusters):
        x_values = [point[1] for point in cluster]
        y_values = [point[0] for point in cluster]
        plt.scatter(x_values, y_values, label=f'Cluster {j+1}')

    # Graficar los centroides como cruces verdes
    plt.scatter(mu[:, 0], mu[:, 1], s=100, marker='x', color='green', label='Centroides')

    plt.gca().invert_yaxis()

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlabel('Position x')
    plt.ylabel('Position y')
    plt.title('Clusters y centroides')
    plt.show()
    
    position_of_camara = np.full((mu.shape[0], 1), i)
    
    mu = np.hstack((position_of_camara, mu))
    
    allmu.append(mu)


    
centroids = np.empty((1,3))
    
for i in range(len(allmu)):
    
    temporal = np.array(allmu[i])
    
    centroids = np.vstack((centroids,temporal))
    
centroids = centroids[1:]

x = centroids[:, 1]  

y = centroids[:, 2] 
 
z = centroids[:, 0]  

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')


ax.scatter(x, y, z, c='r', marker='o')


ax.set_xlabel('X')

ax.set_ylabel('Y')

ax.set_zlabel('Z')


plt.show()