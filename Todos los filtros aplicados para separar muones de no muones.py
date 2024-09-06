import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Cargar los datos desde el archivo de Excel
df = pd.read_excel('Base de datos 2.xlsx', engine='openpyxl')

# Filtrar los datos donde la condición 'camaras 1 y 2 encendidas' es True
df_true = df[df['camaras 1 y 2 encendidas'] == True]

# Aplicar el primer filtro:
# Filtrar datos donde todas las columnas de la 8 a la 12 tienen un valor >= 5
all_values_5_or_more = (df_true.iloc[:, 7:12] >= 5).all(axis=1)
filtered_df_all_values_5 = df_true[all_values_5_or_more]

# Aplicar el segundo filtro:
# Filtrar datos donde la columna 'densidad' es menor que 5
final_filtered_df = filtered_df_all_values_5[filtered_df_all_values_5['densidad'] < 5]

# Contar el número de filas en el DataFrame final filtrado
num_final_filtered = final_filtered_df.shape[0]
print(f'Número de filas en el último grupo filtrado: {num_final_filtered}')

# Crear el histograma para todos los datos con la mitad de las barras y color negro
n, bins = np.histogram(df['numero de hits'], bins='auto')  # Cálculo de bins
half_bins = len(bins) // 2  # Reducir a la mitad el número de barras
n, bins = np.histogram(df['numero de hits'], bins=half_bins)  # Histograma calculado con la mitad de los bins

# Graficar solo la parte superior del histograma para todos los datos (color negro)
plt.step(0.5 * (bins[:-1] + bins[1:]), n, where='mid', color='black', label='Todos los datos')

# Crear el histograma para los datos filtrados donde 'camaras 1 y 2 encendidas' es True, con la mitad de las barras y color rojo
n_true, bins_true = np.histogram(df_true['numero de hits'], bins=half_bins)  # Histograma calculado con la mitad de los bins

# Graficar solo la parte superior del histograma para los datos filtrados (color rojo)
plt.step(0.5 * (bins_true[:-1] + bins_true[1:]), n_true, where='mid', color='red', label='Cámaras 1 y 2 encendidas')

# Crear el histograma para los datos filtrados donde todas las columnas 8-12 >= 5, con la mitad de las barras y color azul
n_all_values_5, bins_all_values_5 = np.histogram(filtered_df_all_values_5['numero de hits'], bins=half_bins)  # Histograma calculado con la mitad de los bins

# Graficar solo la parte superior del histograma para el nuevo grupo filtrado (color azul)
plt.step(0.5 * (bins_all_values_5[:-1] + bins_all_values_5[1:]), n_all_values_5, where='mid', color='blue', label='Condición de penetrabilidad')

# Crear el histograma para el último grupo filtrado donde 'densidad' < 5, con la mitad de las barras y color verde
n_final_filtered, bins_final_filtered = np.histogram(final_filtered_df['numero de hits'], bins=half_bins)  # Histograma calculado con la mitad de los bins

# Graficar solo la parte superior del histograma para el último grupo filtrado (color verde)
plt.step(0.5 * (bins_final_filtered[:-1] + bins_final_filtered[1:]), n_final_filtered, where='mid', color='green', label='Condición de penetrabilidad y densidad < 5')

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
plt.legend(fontsize=12)

# Mostrar el gráfico
plt.show()
