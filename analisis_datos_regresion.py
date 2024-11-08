import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# Cargar los datos desde el archivo JSON
df = pd.read_json('med_MOCK_DATA.json')

# Limpiar los nombres de las columnas
df.columns = df.columns.str.strip()

# Manejar valores faltantes (eliminar filas con NaN)
df = df.dropna() 

# Codificar la columna 'genero' como numérica
df['genero'] = df['genero'].map({'Masculino': 1, 'Femenino': 0})

# Separar las variables predictoras (X) y la variable objetivo (y)
X = df[['edad', 'genero', 'IMC', 'Fumador']]
y = df['Gastos_Medicos']

# Normalizar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Seleccionar las mejores características (por ejemplo, las 3 mejores)
selector = SelectKBest(score_func=f_regression, k=3)
X_selected = selector.fit_transform(X, y)

# Dividir los datos en conjunto de entrenamiento (80%) y conjunto de prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Inicializar el modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Predecir los valores de los gastos médicos en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el rendimiento del modelo utilizando el error cuadrático medio (MSE) y el R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print(f" Error Cuadrático Medio (MSE): {mse:.2f}")
print(f" Coeficiente de Determinación (R2 Score): {r2:.2f}")
print("="*50)

# Imprimir las predicciones para los primeros 100 registros del conjunto de prueba en formato tabular
print("\nPredicciones para los primeros 100 registros del conjunto de prueba:")
print("="*50)
for i, pred in enumerate(y_pred[:100], 1):
    print(f"{i:>3}: {pred:.2f}")
    if i % 10 == 0:
        print("-" * 50)
