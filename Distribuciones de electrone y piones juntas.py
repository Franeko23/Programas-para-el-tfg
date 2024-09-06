import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

# Cargar los datos desde los archivos de Excel

df = pd.read_excel('Base de datos.xlsx', engine='openpyxl')

df1 = pd.read_excel('Base de datos piones.xlsx', engine='openpyxl')

# Crear el histograma para df con la mitad de las barras y color negro

n, bins, patches = plt.hist(df['numero de hits'], bins='auto', edgecolor='black', alpha=0)  # Histograma oculto para calcular bins

half_bins = len(bins) // 2  # Reducir a la mitad el número de barras

n, bins, patches = plt.hist(df['numero de hits'], bins=half_bins, edgecolor='black', color='black', alpha=0)  # Histograma oculto

# Graficar la parte superior del histograma para df (color negro)

plt.step(0.5 * (bins[:-1] + bins[1:]), n, where='mid', color='black', label='Run de electrones')

# Crear el histograma para df1 con la mitad de las barras y color rojo

n1, bins1, patches1 = plt.hist(df1['numero de hits'], bins='auto', edgecolor='red', color='red', alpha=0)  # Histograma oculto

half_bins2 = len(bins1) // 2 

n1, bins1, patches1 = plt.hist(df1['numero de hits'], bins=half_bins2, edgecolor='red', color='red', alpha=0)  # Histograma oculto

# Graficar la parte superior del histograma para df1 (color rojo)

plt.step(0.5 * (bins1[:-1] + bins1[1:]), n1, where='mid', color='red', label='Run de piones')

# Configurar los ticks del eje x para que vayan de 100 en 100 desde 0 hasta 1000, y estén centrados

plt.xticks(np.arange(0, 1101, 100))

# Establecer los límites del eje x

plt.xlim(0, 1100)

# Configurar el eje y para que esté en escala logarítmica
plt.yscale('log')

# Título y etiquetas de los ejes con tamaño de fuente aumentado
plt.xlabel('Número de Hits', fontsize=20)
plt.ylabel('Frecuencia', fontsize=20)

# Configurar el tamaño de fuente de los ticks
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Añadir la leyenda con tamaño de fuente aumentado
plt.legend(fontsize=15)

# Mostrar el gráfico
plt.show()
