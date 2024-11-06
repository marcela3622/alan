import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Carga de los datos desde el archivo JSON
df = pd.read_json('personales_MOCK_DATA.json')

# Limpieza de los nombres de las columnas
df.columns = df.columns.str.strip()

# variables categóricas con LabelEncoder
label_encoders = {}
for column in ['Genero', 'Rango_ de_edad', 'Estado_Civil', 'Ocupación', 'Nivel_educativo']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Guardar el codificador por si lo necesitas más adelante

# variables predictoras (X) y la variable objetivo (y)
X = df[['Genero', 'Rango_ de_edad', 'Estado_Civil', 'Ocupación']]
y = df['Nivel_educativo']  # Variable objetivo para clasificación

#  datos en conjunto de entrenamiento (80%) y conjunto de prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializa el modelo de clasificación
modelo = LogisticRegression(max_iter=1000)

# Entrena el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Predice las clases en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evalua el rendimiento del modelo utilizando la precisión y un reporte de clasificación
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy}")
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Imprime las predicciones para los primeros 100 registros del conjunto de prueba
print("Predicciones para los primeros 100 registros del conjunto de prueba:")
for pred in y_pred[:100]:
    print(pred)
