from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

model_dir = os.getenv("MODEL_DIR")



model_path = os.path.join(model_dir, "model.pkl")
vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("Vectorizer IDF fitted:", hasattr(vectorizer, "idf_"))

app = FastAPI()

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict(data: Message):
    X = vectorizer.transform([data.text])
    y_pred = model.predict(X)
    label = "spam" if y_pred[0] == 1 else "ham"
    return {"prediction": label}
