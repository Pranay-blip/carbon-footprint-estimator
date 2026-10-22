import { useEffect, useState } from "react";
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, LineChart, Line, ScatterChart, Scatter, ReferenceLine,
} from "recharts";
import "./App.css";

const API = "http://localhost:8000";
const DIET = ["Vegan", "Vegetarian", "Mixed", "Meat-heavy"];
const COLORS = ["#2e7d32", "#0288d1", "#f9a825", "#8e24aa", "#e64a19", "#546e7a"];

const SLIDERS = [
  { key: "car_km_week", label: "Car travel (km/week)", min: 0, max: 600, step: 10 },
  { key: "flights_year", label: "Flights per year", min: 0, max: 15, step: 1 },
  { key: "electricity_kwh_month", label: "Electricity (kWh/month)", min: 50, max: 800, step: 10 },
  { key: "diet", label: "Diet", min: 0, max: 3, step: 1 },
  { key: "waste_kg_week", label: "Waste (kg/week)", min: 1, max: 25, step: 1 },
  { key: "shopping_usd_month", label: "Shopping ($/month)", min: 20, max: 1500, step: 20 },
];
const LABELS = Object.fromEntries(SLIDERS.map((s) => [s.key, s.label]));

const DEFAULTS = { car_km_week: 150, flights_year: 2, electricity_kwh_month: 250,
                   diet: 2, waste_kg_week: 8, shopping_usd_month: 300 };
const PRESETS = {
  Student: { car_km_week: 20, flights_year: 1, electricity_kwh_month: 120, diet: 1, waste_kg_week: 5, shopping_usd_month: 100 },
  Commuter: { car_km_week: 400, flights_year: 3, electricity_kwh_month: 300, diet: 2, waste_kg_week: 10, shopping_usd_month: 400 },
  "Frequent flyer": { car_km_week: 200, flights_year: 14, electricity_kwh_month: 500, diet: 3, waste_kg_week: 15, shopping_usd_month: 900 },
};

const MODEL_INFO = {
  "Linear Regression": "Fits one weighted sum of the inputs. Fast and easy to interpret, but it cannot capture interactions like flights × diet.",
  "Random Forest": "Averages 200 decision trees. Handles non-linear effects, so its curves look step-like.",
  "Gradient Boosting": "Builds trees one after another, each correcting the previous errors. Usually the most accurate.",
};

const GLOBAL_AVG = 4700;
const TARGET_2030 = 2300;

const post = (path, body, model) =>
  fetch(`${API}${path}?model=${encodeURIComponent(model)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => {
    if (!r.ok) throw new Error("API error");
    return r.json();
  });

export default function App() {
  const [inputs, setInputs] = useState(DEFAULTS);
  const [model, setModel] = useState("Gradient Boosting");
  const [goal, setGoal] = useState(TARGET_2030);
  const [result, setResult] = useState(null);
  const [allPreds, setAllPreds] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [sensFeature, setSensFeature] = useState("flights_year");
  const [sens, setSens] = useState([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch(API + "/metrics").then((r) => r.json()).then(setMetrics).catch(() => setError(true));
  }, []);

  useEffect(() => {
    let live = true;
    const t = setTimeout(() => {
      Promise.all([
        post("/predict", inputs, model),
        post("/sensitivity", { inputs, feature: sensFeature }, model),
        post("/predict_all", inputs, model),
      ])
        .then(([r, s, a]) => {
          if (!live) return;
          setResult(r); setSens(s); setAllPreds(a);
        })
        .catch(() => live && setError(true));
    }, 150);
    return () => { live = false; clearTimeout(t); };
  }, [inputs, sensFeature, model]);

  if (error) return <div className="msg">Cannot reach the API. Is uvicorn running on port 8000?</div>;
  if (!result || !metrics || !allPreds) return <div className="msg">Loading...</div>;

  const total = Math.round(result.total);
  const current = metrics.models.find((m) => m.model === model);
  const details = metrics.details[model];

  const breakdown = Object.entries(result.breakdown).map(([k, v]) => ({ name: LABELS[k], value: Math.round(v) }));
  const importance = details.importance.map((d) => ({ name: LABELS[d.feature], importance: d.importance }));
  const compare = Object.entries(allPreds).map(([name, kg]) => ({ name, kg: Math.round(kg) }));
  const benchmark = [
    { name: "You", kg: total },
    { name: "Your goal", kg: goal },
    { name: "Global avg", kg: GLOBAL_AVG },
    { name: "2030 target", kg: TARGET_2030 },
  ];
  const status = total <= TARGET_2030 ? "good" : total <= GLOBAL_AVG ? "mid" : "bad";

  // Goal tracker
  const gap = total - goal;
  const reached = gap <= 0;
  const progress = reached ? 100 : Math.max(Math.round((goal / total) * 100), 3);
  const topFactor = Object.entries(result.breakdown).sort((a, b) => b[1] - a[1])[0];

  return (
    <div className="app">
      <h1>🌍 Personal Carbon Footprint Estimator</h1>

      <div className="layout">
        <aside className="card">
          <h2>Model</h2>
          <div className="models">
            {metrics.models.map((m) => (
              <button key={m.model} className={model === m.model ? "active" : ""}
                onClick={() => setModel(m.model)}>
                {m.model}
              </button>
            ))}
          </div>
          <p className="hint">{MODEL_INFO[model]}</p>
          <p className="hint">R² <b>{current.r2}</b> · MAE <b>{current.mae} kg</b></p>

          <h2>Adjust your lifestyle</h2>
          <div className="presets">
            {Object.entries(PRESETS).map(([n, p]) => (
              <button key={n} onClick={() => setInputs(p)}>{n}</button>
            ))}
            <button onClick={() => setInputs(DEFAULTS)}>Reset</button>
          </div>
          {SLIDERS.map((s) => (
            <div key={s.key} className="slider">
              <label>
                {s.label}: <b>{s.key === "diet" ? DIET[inputs.diet] : inputs[s.key]}</b>
              </label>
              <input type="range" min={s.min} max={s.max} step={s.step} value={inputs[s.key]}
                onChange={(e) => setInputs({ ...inputs, [s.key]: Number(e.target.value) })} />
            </div>
          ))}
        </aside>

        <main>
          <div className={`card result ${status}`}>
            <div className="big">{total.toLocaleString()} <small>kg CO₂e / year</small></div>
            <div>{(total / 1000).toFixed(1)} tonnes · {Math.round((total / GLOBAL_AVG) * 100)}% of global average</div>
            <div className="small">Predicted by {model}</div>
          </div>

          <div className="card">
            <h3>🎯 Goal tracker</h3>
            <label>Target: <b>{goal.toLocaleString()}</b> kg CO₂e / year</label>
            <input type="range" min={1000} max={8000} step={100} value={goal}
              style={{ width: "100%", accentColor: "#2e7d32" }}
              onChange={(e) => setGoal(Number(e.target.value))} />
            <div className="presets goalbtns">
              <button onClick={() => setGoal(TARGET_2030)}>2030 target (2,300)</button>
              <button onClick={() => setGoal(GLOBAL_AVG)}>Global average (4,700)</button>
            </div>
            <div className="bar">
              <div className={`fill ${reached ? "ok" : "no"}`} style={{ width: `${progress}%` }} />
            </div>
            <p>
              {reached
                ? `✅ Goal reached! You are ${(-gap).toLocaleString()} kg under your target.`
                : `You need to cut ${gap.toLocaleString()} kg (${Math.round((gap / total) * 100)}% of your footprint) to hit your goal. Your biggest contributor is ${LABELS[topFactor[0]]}.`}
            </p>
          </div>

          <div className="grid">
            <div className="card">
              <h3>Contribution by factor</h3>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={breakdown} dataKey="value" nameKey="name" innerRadius={50} outerRadius={90} label>
                    {breakdown.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <h3>You vs goal and benchmarks</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={benchmark}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" /><YAxis /><Tooltip />
                  <Bar dataKey="kg">
                    {benchmark.map((b, i) => (
                      <Cell key={i} fill={["#2e7d32", "#0288d1", "#9e9e9e", "#9e9e9e"][i]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="card wide">
              <h3>Same inputs, three models</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={compare}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" /><YAxis domain={["auto", "auto"]} /><Tooltip />
                  <Bar dataKey="kg">
                    {compare.map((c) => (
                      <Cell key={c.name} fill={c.name === model ? "#2e7d32" : "#b0bec5"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="card wide">
              <h3>What-if: sweep one factor ({model})</h3>
              <select value={sensFeature} onChange={(e) => setSensFeature(e.target.value)}>
                {SLIDERS.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
              </select>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={sens}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="x" />
                  <YAxis domain={[(min) => Math.min(min, goal), (max) => Math.max(max, goal)]} />
                  <Tooltip />
                  <ReferenceLine y={goal} stroke="#c62828" strokeDasharray="5 5"
                    label={{ value: "Goal", fill: "#c62828", position: "insideTopRight" }} />
                  <Line type="monotone" dataKey="y" stroke="#0288d1" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </main>
      </div>

      <h2>How the model works</h2>
      <div className="pipeline">
        {["Synthetic data (5,000 rows)", "Train/test split 80/20", "Train 3 models", `Selected: ${model}`, "Live prediction"]
          .map((s, i, a) => <span key={s}><div className="step">{s}</div>{i < a.length - 1 && "→"}</span>)}
      </div>

      <div className="grid">
        <div className="card">
          <h3>Model comparison</h3>
          <table>
            <thead><tr><th>Model</th><th>R²</th><th>MAE (kg)</th></tr></thead>
            <tbody>
              {metrics.models.map((m) => (
                <tr key={m.model} className={m.model === model ? "best" : ""}
                  style={{ cursor: "pointer" }} onClick={() => setModel(m.model)}>
                  <td>{m.model}{m.model === metrics.best && " ★"}</td><td>{m.r2}</td><td>{m.mae}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="hint">★ best R². Click a row to switch model.</p>
        </div>

        <div className="card">
          <h3>Feature importance ({model})</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={importance} layout="vertical" margin={{ left: 40 }}>
              <XAxis type="number" /><YAxis type="category" dataKey="name" width={150} /><Tooltip />
              <Bar dataKey="importance" fill="#8e24aa" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card wide">
          <h3>Actual vs predicted, test set ({model})</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid />
              <XAxis type="number" dataKey="actual" name="Actual" domain={["auto", "auto"]} />
              <YAxis type="number" dataKey="predicted" name="Predicted" domain={["auto", "auto"]} />
              <Tooltip /><Scatter data={details.scatter} fill="#e64a19" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}