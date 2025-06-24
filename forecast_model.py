from prophet import Prophet
import pandas as pd
from datetime import datetime

# --- Global caches ---
MODEL_CACHE = {}
DATA_CACHE = {}

# --- File paths ---
file_map = {
    "nasdaq": "data/nasdaq.csv",
    "snp": "data/snp.csv",
    "dow": "data/downjones.csv",
    "russell": "data/rut.csv"
}

# --- Preload and train models once ---
for index, path in file_map.items():
    try:
        df = pd.read_csv(path)
        df.rename(columns={"Date": "ds", "Close/Last": "y"}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"], errors='coerce')
        df["y"] = df["y"].replace('[\$,]', '', regex=True).astype(float)
        df = df.dropna(subset=["ds", "y"]).drop_duplicates(subset="ds").sort_values("ds")

        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(df)

        MODEL_CACHE[index] = model
        DATA_CACHE[index] = df
        print(f"✅ Loaded and trained model for {index}")
    except Exception as e:
        print(f"❌ Failed to load model for {index}: {e}")

# --- Fast forecasting using cached models ---
def load_and_forecast(index: str = "nasdaq", days: int = 365):
    if index not in MODEL_CACHE:
        return {"error": "Invalid index"}

    model = MODEL_CACHE[index]

    # Predict future values
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # Smooth forecast
    future_range = forecast[['ds', 'yhat']].tail(days).copy()
    future_range['yhat_smooth'] = future_range['yhat'].rolling(window=7, min_periods=1).mean()

    result = [
        {"ds": row["ds"].strftime("%Y-%m-%d"), "yhat": round(row["yhat_smooth"], 2)}
        for _, row in future_range.iterrows()
    ]
    return {"forecast": result}
