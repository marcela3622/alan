import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar los datos desde el archivo JSON
df = pd.read_json('med_MOCK_DATA.json')

# Limpiar los nombres de las columnas
df.columns = df.columns.str.strip()

# Verificar los nombres de las columnas
print("\n" + "="*50)
print(" Columnas disponibles en el DataFrame ")
print("="*50)
print(", ".join(df.columns))

# Mostrar los primeros registros del DataFrame para verificar
print("\nPrimeros registros del DataFrame:")
print("="*50)
print(df.head())

# Manejar valores faltantes (eliminar filas con NaN o rellenarlas con la media)
df = df.dropna()  # Eliminar filas con NaN

# Codificar la columna 'genero' como numérica (Masculino=1, Femenino=0)
df['genero'] = df['genero'].map({'Masculino': 1, 'Femenino': 0})

# Separar las variables predictoras (X) y la variable objetivo (y)
X = df[['edad', 'genero', 'IMC', 'Fumador']]  # Variables predictoras
y = df['Gastos_Medicos']  # Variable objetivo

# Dividir los datos en conjunto de entrenamiento (80%) y conjunto de prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Predecir los valores de los gastos médicos en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el rendimiento del modelo utilizando el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print("\n" + "="*50)
print(f" Error Cuadrático Medio (MSE): {mse:.2f}")
print("="*50)

# Imprimir las predicciones para los primeros 100 registros del conjunto de prueba en formato tabular
print("\nPredicciones para los primeros 100 registros del conjunto de prueba:")
print("="*50)

for i, pred in enumerate(y_pred[:100], 1):
    print(f"{i:>3}: {pred:.2f}")
    if i % 10 == 0:
        print("-"*50)  # Separador cada 10 registros
