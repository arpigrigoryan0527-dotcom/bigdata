from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

app = FastAPI()

data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

class WineInput(BaseModel):
    alcohol: float = Field(..., gt=0)
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float


@app.get("/")
def home():
    return {"message": "Wine ML API running"}


@app.post("/predict")
def predict(input_data: WineInput):
    X_input = np.array([[ 
        input_data.alcohol,
        input_data.malic_acid,
        input_data.ash,
        input_data.alcalinity_of_ash,
        input_data.magnesium,
        input_data.total_phenols,
        input_data.flavanoids,
        input_data.nonflavanoid_phenols,
        input_data.proanthocyanins,
        input_data.color_intensity,
        input_data.hue,
        input_data.od280_od315,
        input_data.proline
    ]])

    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input).max()

    return {
        "prediction": data.target_names[pred],
        "confidence": float(prob)
    }

@app.get("/info")
def info():
    return {
        "model": "RandomForestClassifier",
        "features": list(data.feature_names),
        "classes": list(data.target_names)
    }

@app.get("/metrics")
def metrics():
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted"))
    }


@app.get("/sample")
def sample():
    return {
        "example": dict(zip(data.feature_names, X[0]))
    }
