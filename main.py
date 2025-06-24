from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from forecast_model import load_and_forecast

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. Set specific origin in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict(index: str = "nasdaq", days: int = 365):
    return load_and_forecast(index=index, days=days)
