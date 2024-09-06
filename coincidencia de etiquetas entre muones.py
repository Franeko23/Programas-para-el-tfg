import pandas as pd

# Cargar los dos archivos Excel
df_finales = pd.read_excel('Datos_Finales_Filtrados.xlsx', engine='openpyxl')
df_excluidos = pd.read_excel('Datos_Excluidos_Cluster_0.xlsx', engine='openpyxl')

# Asegurarse de que la columna 'posiciones' esté presente en ambos DataFrames
if 'posiciones' in df_finales.columns and 'posiciones' in df_excluidos.columns:
    # Obtener las posiciones de ambos DataFrames
    posiciones_finales = df_finales['posiciones']
    posiciones_excluidos = df_excluidos['posiciones']
    
    # Encontrar las coincidencias entre las posiciones
    coincidencias = posiciones_finales.isin(posiciones_excluidos)
    
    # Contar el número de coincidencias
    num_coincidencias = coincidencias.sum()
    
    print(f"El número de posiciones que coinciden entre ambos archivos es: {num_coincidencias}")
else:
    print("La columna 'posiciones' no está presente en ambos archivos.")
