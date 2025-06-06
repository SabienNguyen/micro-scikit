from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram


# Load environment variables
load_dotenv()

model_path = "models/model.pkl"
vectorizer_path = "models/vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("Vectorizer IDF fitted:", hasattr(vectorizer, "idf_"))

app = FastAPI()
Instrumentator().instrument(app).expose(app)

class Message(BaseModel):
    text: str

prediction_counter = Counter("model_predictions_total", "Total number of model predictions", ["label"])
prediction_latency = Histogram("model_prediction_latency_seconds", "Latency for model predictions")

@app.post("/predict")
@prediction_latency.time()
def predict(data: Message):
    X = vectorizer.transform([data.text])
    y_pred = model.predict(X)
    label = "spam" if y_pred[0] == 1 else "ham"
    prediction_counter.labels(label=label).inc()
    return {"prediction": label}
