from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Cargar dataset y entrenar modelo
iris = load_iris()
X, y = iris.data, iris.target

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Guardar el modelo entrenado
joblib.dump(clf, "model.joblib")
print("Modelo entrenado y guardado exitosamente como model.joblib")