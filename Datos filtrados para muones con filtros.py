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

# Guardar el DataFrame final filtrado en un archivo Excel
final_filtered_df.to_excel('Datos_Finales_Filtrados.xlsx', index=False)