from statsforecast import StatsForecast
from statsforecast.models import AutoETS
import pandas as pd
import os

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

# --- Load data and train models on import ---
for index, path in file_map.items():
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        # Load and clean data
        df = pd.read_csv(path)
        df.rename(columns={"Date": "ds", "Close/Last": "y"}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["y"] = df["y"].replace('[\$,]', '', regex=True).astype(float)
        df = df.dropna(subset=["ds", "y"]).drop_duplicates(subset="ds").sort_values("ds")

        # Add required column for StatsForecast
        df["unique_id"] = index

        # Save data for later reference
        DATA_CACHE[index] = df

        # Train model
        sf = StatsForecast(
            models=[AutoETS(season_length=252, model="ZZZ")],  # 252 = trading days/year
            freq="D",
            n_jobs=1
        )
        sf.fit(df[["unique_id", "ds", "y"]])
        MODEL_CACHE[index] = sf

        print(f"✅ Loaded and trained StatsForecast model for {index}")
    except Exception as e:
        print(f"❌ Failed to load model for {index}: {e}")

# --- Forecast function ---
def load_and_forecast(index: str = "nasdaq", days: int = 365):
    if index not in MODEL_CACHE:
        return {"error": "Invalid index"}

    try:
        sf = MODEL_CACHE[index]
        df = DATA_CACHE[index]

        # Predict future values
        forecast_df = sf.predict(h=days)
        forecast = forecast_df[forecast_df["unique_id"] == index].copy()

        # Smooth predictions using a 7-day rolling average
        forecast["yhat_smooth"] = forecast["yhat"].rolling(window=7, min_periods=1).mean()

        # Format output
        result = [
            {
                "ds": (df["ds"].max() + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d"),
                "yhat": round(row["yhat_smooth"], 2)
            }
            for i, row in forecast.iterrows()
        ]

        return {"forecast": result}

    except Exception as e:
        return {"error": f"Forecasting failed: {str(e)}"}
