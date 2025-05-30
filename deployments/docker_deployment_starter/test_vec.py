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