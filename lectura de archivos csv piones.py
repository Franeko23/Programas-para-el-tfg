import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

# Definición de la función para encontrar vecinos

def find_neighbors(coordenada_x, coordenada_y, pares_coordenadas):
    
    neighbors = []
    
    directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    
    for dir in directions:
        
        new_x = coordenada_x + dir[0]
        
        new_y = coordenada_y + dir[1]
        
        par_buscar = f'{new_x},{new_y}'
        
        if par_buscar in pares_coordenadas:
            
            neighbors.append((new_x, new_y))
            
    return neighbors

carpeta = r'C:\Users\Francisco\Desktop\DatosPi_50GeV'

# Iterar sobre los archivos del 1 al 48

clusters = []

for i in range(1, 49):
    
    nombre_archivo = f'camara{i}.csv'
    
    ruta_archivo = os.path.join(carpeta, nombre_archivo)
    
    number_of_cluster = 0
    
    try:
        
        with open(ruta_archivo, 'r') as file:
            
            for line in file:
                
                if line.strip():  # Verificar que la línea no esté vacía
                
                    visitado = set()
                    
                    pares_coordenadas = line.strip().split(';')
                    
                    for par in pares_coordenadas:
                        
                        if ',' not in par:
                            
                            continue  # Saltar si no hay comas en la línea
                        
                        coordenada_x, coordenada_y = map(int, par.split(','))
                        
                        # Verificar si el par de coordenadas ya ha sido visitado
                        
                        if par not in visitado:
                            
                            visitado.add(par)
                            
                            cluster = [(coordenada_x, coordenada_y)]
                            
                            # Búsqueda de vecindario usando DFS (Depth-First Search)
                            k = 0
                            
                            while k < len(cluster):
                                
                                x, y = cluster[k]
                                
                                neighbors = find_neighbors(x, y, pares_coordenadas)
                                
                                for neighbor in neighbors:
                                    
                                    if neighbor not in cluster:
                                        
                                        cluster.append(neighbor)
                                        
                                        visitado.add(f'{neighbor[0]},{neighbor[1]}')
                                k += 1
                                
                            if len(cluster) > 0:
                                
                                number_of_cluster += 1  # Incrementar contador de clusters
            
            # Después de procesar todas las líneas del archivo, agregar el número de clusters a la lista
            
            clusters.append(number_of_cluster)
    
    except FileNotFoundError:
        
        print(f'El archivo {nombre_archivo} no fue encontrado.')
    
    except Exception as e:
        
        print(f'Error al procesar el archivo {nombre_archivo}: {str(e)}')

# Graficar los resultados
camaras = list(range(1, 49))

plt.bar(camaras, clusters, color='blue', edgecolor='black')

plt.xlabel('Cámaras')

plt.ylabel('Número de clusters')

plt.title('Número de clusters por cámara')

plt.xticks(camaras)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

