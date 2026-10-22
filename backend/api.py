from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json, pandas as pd

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SLUGS = {
    "Linear Regression": "linear_regression",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
}
MODELS = {name: joblib.load(f"artifacts/{slug}.pkl") for name, slug in SLUGS.items()}
METRICS = json.load(open("artifacts/metrics.json"))

FEATURES = ["car_km_week", "flights_year", "electricity_kwh_month",
            "diet", "waste_kg_week", "shopping_usd_month"]
RANGES = {"car_km_week": (0, 600), "flights_year": (0, 15), "electricity_kwh_month": (50, 800),
          "diet": (0, 3), "waste_kg_week": (1, 25), "shopping_usd_month": (20, 1500)}


class Inputs(BaseModel):
    car_km_week: float
    flights_year: float
    electricity_kwh_month: float
    diet: float
    waste_kg_week: float
    shopping_usd_month: float


class SensReq(BaseModel):
    inputs: Inputs
    feature: str


def get_model(name: str):
    if name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {name}")
    return MODELS[name]


def predict_rows(model, rows):
    return model.predict(pd.DataFrame(rows)[FEATURES])


@app.post("/predict")
def predict(inp: Inputs, model: str = "Gradient Boosting"):
    m = get_model(model)
    d = inp.model_dump()
    # Row 0 is the real input; the rest set one factor at a time to its minimum
    rows = [d] + [{**d, f: RANGES[f][0]} for f in FEATURES]
    preds = predict_rows(m, rows)
    total = float(preds[0])
    breakdown = {f: max(total - float(preds[i + 1]), 0) for i, f in enumerate(FEATURES)}
    return {"total": total, "breakdown": breakdown}


@app.post("/predict_all")
def predict_all(inp: Inputs):
    d = inp.model_dump()
    return {name: float(predict_rows(m, [d])[0]) for name, m in MODELS.items()}


@app.post("/sensitivity")
def sensitivity(req: SensReq, model: str = "Gradient Boosting"):
    m = get_model(model)
    d = req.inputs.model_dump()
    lo, hi = RANGES[req.feature]
    steps = 30
    xs = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
    preds = predict_rows(m, [{**d, req.feature: v} for v in xs])
    return [{"x": round(v, 2), "y": float(p)} for v, p in zip(xs, preds)]


@app.get("/metrics")
def metrics():
    return METRICS