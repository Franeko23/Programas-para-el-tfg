import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo Excel y convertirlo en DataFrame
df = pd.read_excel('Base de datos piones.xlsx', engine='openpyxl')

plt.figure(figsize=(10, 6))

# Graficar los puntos con transparencia
plt.scatter(df['numero de hits'], df['Numero de cluster'], color='blue', alpha=0.2)


plt.xlabel('Número de Hits', fontsize=20)
plt.ylabel('Número de Cluster', fontsize=20)

# Ajustar el tamaño de los números de los ejes
plt.tick_params(axis='both', which='major', labelsize=15)

plt.grid(True)

# Mostrar el gráfico
plt.show()

