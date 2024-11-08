from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Carga de los datos
df = pd.read_json('personales_MOCK_DATA.json')
df.columns = df.columns.str.strip()

# Variables categóricas con LabelEncoder
label_encoders = {}
for column in ['Genero', 'Rango_ de_edad', 'Estado_Civil', 'Ocupación', 'Nivel_educativo']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Variables predictoras (X) y variable objetivo (y)
X = df[['Genero', 'Rango_ de_edad', 'Estado_Civil', 'Ocupación']]
y = df['Nivel_educativo']

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Selección de características
selector = RFE(LogisticRegression(max_iter=1000), n_features_to_select=3)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Ajuste de hiperparámetros
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_selected, y_train)

# Evaluar el mejor modelo encontrado
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del Modelo Ajustado: {accuracy:.2%}")
print(classification_report(y_test, y_pred))
