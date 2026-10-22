from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Esta es la variable 'app' que Uvicorn está buscando
app = FastAPI(title="ML Prediction API")

# Cargar el modelo guardado
model = joblib.load("model.joblib")

class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "API de Predicción de ML funcionando correctamente"}

@app.post("/predict")
def predict(data: Features):
    input_data = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])
    prediction = model.predict(input_data)
    classes = ["setosa", "versicolor", "virginica"]
    return {"class_id": int(prediction[0]), "class_name": classes[prediction[0]]}