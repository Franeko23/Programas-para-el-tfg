import numpy as np
import os
import matplotlib.pyplot as plt

# Ruta a la carpeta donde están los archivos CSV
carpeta = r'C:\Users\Francisco\Desktop\TFG\Datos_50GeV\DatosEl_50GeV'

# Crear una matriz de ceros de dimensiones 48x97x97 para cada evento
matrix_47 = np.zeros((48, 97, 97))
matrix_144 = np.zeros((48, 97, 97))

# Leer los archivos CSV y actualizar las matrices para los eventos 47 y 144
for i in range(1, 49):
    nombre_archivo = f'file{i}.csv'
    ruta_archivo = os.path.join(carpeta, nombre_archivo)
    
    try:
        with open(ruta_archivo, 'r') as file:
            for j, line in enumerate(file):
                if j == 47:  # Evento 47
                    pares_coordenadas = line.strip().split(';')
                    for par in pares_coordenadas:
                        if ',' not in par:
                            continue  # Saltar si no hay comas en la línea
                        coordenada_x, coordenada_y = map(int, par.split(','))
                        matrix_47[i-1, coordenada_x, coordenada_y] = 1
                
                elif j == 144:  # Evento 144
                    pares_coordenadas = line.strip().split(';')
                    for par in pares_coordenadas:
                        if ',' not in par:
                            continue  # Saltar si no hay comas en la línea
                        coordenada_x, coordenada_y = map(int, par.split(','))
                        matrix_144[i-1, coordenada_x, coordenada_y] = 1

    except FileNotFoundError:
        print(f'El archivo {nombre_archivo} no fue encontrado.')
    except Exception as e:
        print(f'Error al procesar el archivo {nombre_archivo}: {str(e)}')

# Crear la figura y dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))  # Dos subplots en una fila

# Graficar los puntos del evento 47 en azul (electrón)
num_matrices, rows, cols = matrix_47.shape
for i in range(num_matrices):
    x_vals, y_vals = np.where(matrix_47[i] == 1)
    y_vals = np.full_like(x_vals, i)
    ax1.scatter(y_vals, x_vals, marker='o', color='blue')

# Graficar los puntos del evento 144 en rojo (pión)
num_matrices, rows, cols = matrix_144.shape
for i in range(num_matrices):
    x_vals, y_vals = np.where(matrix_144[i] == 1)
    y_vals = np.full_like(x_vals, i)
    ax2.scatter(y_vals, x_vals, marker='o', color='red')

# Ajustar límites de los ejes y configurar el eje X hacia abajo
ax1.set_xlim(0, 48)  # Limitar el eje X de 0 a 48 para el evento 47
ax1.set_ylim(0, 96)  # Limitar el eje Y de 0 a 96 para el evento 47
ax2.set_xlim(0, 48)  # Limitar el eje X de 0 a 48 para el evento 144
ax2.set_ylim(0, 96)  # Limitar el eje Y de 0 a 96 para el evento 144

# Etiquetas y título de los gráficos
ax1.set_xlabel('Cámara del detector', fontsize=20)
ax1.set_ylabel('Eje Y [cm]', fontsize=20)
ax1.xaxis.tick_top()  # Mover etiquetas del eje X arriba
ax1.xaxis.set_label_position("top")  # Etiqueta del eje X arriba
ax1.set_title('Electrón', fontsize=20)

ax2.set_xlabel('Cámara del detector', fontsize=20)
ax2.set_ylabel('Eje Y [cm]', fontsize=20)
ax2.xaxis.tick_top()  # Mover etiquetas del eje X arriba
ax2.xaxis.set_label_position("top")  # Etiqueta del eje X arriba
ax2.set_title('Pión ', fontsize=20)

# Mostrar ambos gráficos en una sola imagen
plt.show()
