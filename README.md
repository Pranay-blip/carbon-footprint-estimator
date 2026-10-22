# Personal Carbon Footprint Estimator

ML mini project: predicts annual CO₂ emissions from lifestyle inputs, with an interactive React dashboard (sliders, model switcher, goal tracker, what-if charts).

**Stack:** Python, scikit-learn, FastAPI, React (Vite), Recharts

## Setup

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python train.py
uvicorn api:app --reload
```

### Frontend (second terminal)
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173