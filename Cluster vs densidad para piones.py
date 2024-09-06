import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo Excel y convertirlo en DataFrame
df = pd.read_excel('Base de datos piones.xlsx', engine='openpyxl')

# Crear el gráfico de dispersión
plt.scatter(df['Numero de cluster'], df['densidad'], color='blue', alpha=0.2)

# Aumentar el tamaño del título y etiquetas de los ejes
plt.title('Número de Cluster vs Densidad', fontsize=25)
plt.xlabel('Número de Cluster', fontsize=20)
plt.ylabel('Densidad', fontsize=20)

# Mostrar el gráfico
plt.show()
