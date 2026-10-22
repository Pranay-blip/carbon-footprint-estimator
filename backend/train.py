import pandas as pd, numpy as np, joblib, json, os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

df = pd.read_csv("data/carbon_data.csv")
X, y = df.drop(columns="total"), df["total"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
SLUGS = {
    "Linear Regression": "linear_regression",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
}

os.makedirs("artifacts", exist_ok=True)
results, details = [], {}

for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict(X_te)
    results.append({"model": name,
                    "r2": round(r2_score(y_te, p), 4),
                    "mae": round(mean_absolute_error(y_te, p), 1)})
    print(results[-1])

    imp = permutation_importance(m, X_te, y_te, n_repeats=10, random_state=42)
    details[name] = {
        "importance": sorted(
            [{"feature": f, "importance": round(float(v), 4)} for f, v in zip(X.columns, imp.importances_mean)],
            key=lambda d: d["importance"], reverse=True),
        "scatter": [{"actual": float(a), "predicted": float(b)} for a, b in list(zip(y_te, p))[:250]],
    }
    joblib.dump(m, f"artifacts/{SLUGS[name]}.pkl", compress=3)

best_name = max(results, key=lambda r: r["r2"])["model"]
print("Best:", best_name)

json.dump({"best": best_name, "models": results, "details": details,
           "n_train": len(X_tr), "n_test": len(X_te)},
          open("artifacts/metrics.json", "w"))