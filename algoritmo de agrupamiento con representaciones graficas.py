import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

# Definición de la funcion para encontrar vecinos cuando procesamos una linea del csv

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

# En estas primeras lineas tratamos de identificar la carpeta donde estan los csv
# y definimos la variable de tramaño por camara. Es una lista que se va rellenando
# para cada camara con el tamaño medio del cluster por camara

carpeta = r'C:\Users\Francisco\Desktop\TFG\Datos_50GeV\DatosEl_50GeV'

size_per_camera = []

hits_per_camera = []

vector = [0] * 96

number_of_clusters = []

repeticion = []

# Abrimos los csv para ir leyendo mas tarde linea a linea y en cada csv
#definios las variables de number of hits total y number of cluster per
#camera que contienen la informacion al final del algoritmo del numero de
#hits por camara y el numero de cluster, que nos servira para calcular mas
#adelante el tamaño medio por camara.

for i in range(1, 49):
    
    nombre_archivo = f'file{i}.csv'
    
    ruta_archivo = os.path.join(carpeta, nombre_archivo)
    
    number_of_cluster_per_camera = 0
    
    number_of_hits_total = 0
    
    try:
        
        with open(ruta_archivo, 'r') as file:
            
            for j, line in enumerate(file):
                
                if line.strip():  # Verificar que la línea no esté vacía
                
                    visitado = set()
            
                    number_of_cluster = 0
                    
                    pares_coordenadas = line.strip().split(';')
                    
                    # Inicializar para calcular el número de pares y el número total
                    
                    number_of_par = 0                                       
                    
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
                                
                                number_of_par += len(cluster)  # Sumar al número total de pares

                    number_of_cluster_per_camera = number_of_cluster_per_camera + number_of_cluster
                    
                    number_of_hits_total = number_of_hits_total + number_of_par
                    
                    while len(number_of_clusters) <= j:
                                            
                        number_of_clusters.append(0)
                    
                    number_of_clusters[j] += number_of_cluster   
                
                    clusters = np.array(number_of_clusters)
                
                    if number_of_cluster > 0:
                    
                       average_value = number_of_par / number_of_cluster
                       
                       average_value = round(average_value)
                        
                       # Asegurar que average_value esté dentro del rango del vector
                       
                       if 0 <= average_value < len(vector):
                           
                            vector[average_value-1] += 1
                
            size_per_camera.append(number_of_hits_total / number_of_cluster_per_camera)

            hits_per_camera.append(number_of_hits_total) 

            repeticion.append(int(number_of_hits_total / number_of_cluster_per_camera))                                                   
                                          
    except FileNotFoundError:
        
        print(f'El archivo {nombre_archivo} no fue encontrado.')
        
    except Exception as e:
        
        print(f'Error al procesar el archivo {nombre_archivo}: {str(e)}')
        
# Representacion del vector size_per_camera como un histograma para ver cual es el tamaño
# medio del cluster por camara. Con el valor encima de la barra.

camaras = list(range(1, 49))

plt.figure(figsize=(12, 6))

bars = plt.bar(camaras, size_per_camera, color='blue', edgecolor='black')

plt.xlabel('Cámaras')

plt.ylabel('Tamaño medio del cluster')

plt.title('tamaño medio de clusters por camara')

plt.xticks(camaras)

plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    
    yval = bar.get_height()
    
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')  # Ajuste de la posición vertical
    
plt.tight_layout()

plt.show()

# Aqui comienza la representacion de otra figura en este caso representaremos 
# El numero de hits por camara

plt.figure(figsize=(12, 6))

plt.bar(camaras, hits_per_camera, color='blue', edgecolor='black')

plt.xlabel('Cámaras')

plt.ylabel('Numero de hits')

plt.title('Numero de hits por cada camara')

plt.xticks(camaras)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# Comienzo de una nueva figura:
# En este caso se calcula el numero de cluster con tamaño medio 1,2,3,4 .....
# hasta llegar a 96, puesto que a partir de ahi no hay cluster mucho mas grandes.
#evidentemente para crear este histograma necesitamos redondear el tamaño medio
#por eso nos salen valores enteros. Ademas, los cluster son para todas las lineas
#de todos los csv.

indices = range(1, len(vector) + 1)

plt.figure(figsize=(10, 6))

bars = plt.bar(indices, vector, color='skyblue', edgecolor='black')

# Añadir etiquetas y títulos

plt.title('Histograma de la frecuencia de valores de tamaños medio de cluster')

plt.xlabel('Valor del tamaño medio')

plt.ylabel('frecuencia')

plt.xticks(indices[::2])   # Establecer los ticks del eje x como los índices

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Añadir etiquetas con los valores encima de cada barra

for i, bar in enumerate(bars):
    
    yval = bar.get_height()
    
    if i < 5 or i >= len(bars) - 10:
        
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()

plt.show()

# En esta figura pasamos a vector una lista de la misma longitud del numero de eventos
# de la particula que estamos estudiando. En cada elemento, podremos observar un numero
# que representa el numero de cluster por evento. Cada elemento representa un evento
# evidentemente en fila. A continuacion se muestra el algoritmo para plotear el histograma
#de dicho vector.

plt.figure(figsize=(12, 6))

# Crear el histograma
counts, bins, _ = plt.hist(clusters, bins=np.arange(0, 180, 2), color='skyblue', edgecolor='black')

# Añadir etiquetas y título
plt.title('Histograma número de clusters' )
plt.xlabel('Número de Clusters')
plt.ylabel('Frecuencia con la que se repiten el numero de cluster')

# Ajustar ticks del eje x para mostrar todos los números
plt.xticks(np.arange(0, 200, 4))  # Ajusta el paso del tick según tus necesidades

# Ajustar límites del eje y
plt.ylim(0, max(counts) + 10)

# Mostrar el histograma
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# En este otro plot, se puede observar un histograma con el tamaño medio del cluster 
# Es decir, representara tengo un tamaño medio de 1 en tantas camaras, tengo un tamaño
# 4 en tantas camaras y asi.

plt.figure(figsize=(12, 6))

# Calcular las posiciones centrales de cada bin

bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Crear el histograma

counts, bins, _ = plt.hist(repeticion, bins = 10, color='skyblue', edgecolor='black')

# Añadir etiquetas y título

plt.title('Histograma tantas camaras con tamaño medio de cluster para electrones' )

plt.xlabel('Tamaño medio cluster')

plt.ylabel('Nº de camaras')

plt.xticks(bin_centers, labels=np.arange(1, len(bin_centers) + 1))

# Mostrar el histograma

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

plt.show()

