import numpy as np, pandas as pd, os

rng = np.random.default_rng(42)
n = 5000

df = pd.DataFrame({
    "car_km_week": rng.uniform(0, 600, n),
    "flights_year": rng.integers(0, 16, n),
    "electricity_kwh_month": rng.uniform(50, 800, n),
    "diet": rng.integers(0, 4, n),            # 0 vegan, 1 vegetarian, 2 mixed, 3 meat-heavy
    "waste_kg_week": rng.uniform(1, 25, n),
    "shopping_usd_month": rng.uniform(20, 1500, n),
})

# Emission factors are approximate, based on EPA / DEFRA style values
df["total"] = (
    df.car_km_week * 52 * 0.12
    + df.flights_year * 200
    + df.electricity_kwh_month * 12 * 0.25
    + df.diet * 500 + 300
    + df.waste_kg_week * 52 * 0.3
    + df.shopping_usd_month * 12 * 0.1
    + df.flights_year * df.diet * 25          # interaction term
    + rng.normal(0, 150, n)                   # noise
)

os.makedirs("data", exist_ok=True)
df.to_csv("data/carbon_data.csv", index=False)
print("Saved", df.shape)