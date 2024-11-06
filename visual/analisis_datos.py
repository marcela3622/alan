import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar los datos desde el archivo JSON
df = pd.read_json('C:\\visual\\med_MOCK_DATA.json')

# Limpiar los nombres de las columnas
df.columns = df.columns.str.strip()

# Verificar los nombres de las columnas
print("Columnas disponibles:", df.columns)

# Mostrar los primeros registros del DataFrame para verificar
print(df.head())

# Manejar valores faltantes (eliminar filas con NaN o rellenarlas con la media)
df = df.dropna()  # Eliminar filas con NaN (si decides eliminar las filas con datos faltantes)

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
print(f"Error cuadrático medio (MSE): {mse}")

# Imprimir las predicciones para los primeros 100 registros del conjunto de prueba
print("Predicciones para los primeros 100 registros del conjunto de prueba:")
for pred in y_pred[:100]:  # Mostrar los primeros 20 registros
    print(pred)