# PackAI — AI Packaging Automation Platform

Hybrid rule-based + ML packaging optimizer for Indian ecommerce warehouses.
Cuts shipping costs by up to 35% using volumetric pricing optimization.

---

## What's included

```
packai/
├── backend/                  ← FastAPI backend
│   ├── app/
│   │   ├── api/              ← All route handlers
│   │   ├── models/           ← SQLAlchemy ORM models (8 tables)
│   │   ├── schemas/          ← Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── decision_service.py   ← Main hybrid optimizer
│   │   │   ├── packing_service.py    ← FFD bin packing algorithm
│   │   │   ├── cost_service.py       ← Volumetric pricing engine
│   │   │   ├── ml_service.py         ← Multi-model ML engine
│   │   └── core/             ← Config, DB, JWT security
│   ├── main.py               ← FastAPI app entry point
│   ├── requirements.txt
│   └── Dockerfile
│
├── ml_engine/                ← Machine learning pipeline
│   ├── train_models.py       ← Trains 5 models + ensemble
│   ├── predict.py            ← Standalone prediction module
│   ├── dataset/              ← Auto-generated if no CSV
│   └── models/               ← Saved .pkl files after training
│
└── frontend/
    └── packai.html           ← Complete single-file frontend
```

---

## Setup — Backend

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Set up PostgreSQL
```bash
# Create database
createdb packaidb
createuser packai
psql -c "ALTER USER packai WITH PASSWORD 'packai';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE packaidb TO packai;"
```

Or update DATABASE_URL in `.env`:
```
DATABASE_URL=postgresql://your_user:your_pass@localhost:5432/packaidb
SECRET_KEY=your-secret-key-here
```

### 3. Train ML models first
```bash
cd ml_engine
python train_models.py
# Trains: Random Forest, Gradient Boosting, Extra Trees, SVM, Ensemble
# Saves to: ml_engine/models/
```

### 4. Run the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

API docs at: http://localhost:8000/docs

---

## Setup — Frontend

The frontend is a single HTML file — no build step needed.

```bash
# Open directly in browser
open frontend/packai.html

# Or serve with Python
cd frontend
python -m http.server 3000
# Visit: http://localhost:3000/packai.html
```

The frontend works in demo mode even without the backend running.
Connect it to the backend by updating the `API` variable in the HTML:
```js
const API = 'http://localhost:8000';
```

---

## Docker (optional)

```bash
# From backend directory
docker build -t packai-backend .
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/packaidb \
  -e SECRET_KEY=your-secret \
  packai-backend
```

---

## API Endpoints

| Method | Endpoint              | Description                    |
|--------|-----------------------|--------------------------------|
| GET    | /health               | Health check + loaded models   |
| POST   | /auth/register        | Register new user              |
| POST   | /auth/login           | Login, get JWT token           |
| POST   | /products             | Add a product                  |
| GET    | /products             | List your products             |
| POST   | /orders               | Create an order                |
| GET    | /orders               | List your orders               |
| POST   | /optimize-packaging   | Run hybrid decision engine     |
| GET    | /inventory            | List box inventory             |
| POST   | /inventory            | Add box type                   |
| PUT    | /inventory/{type}/restock | Restock a box type         |
| GET    | /analytics/summary    | Full analytics data            |

---

## ML Models Trained

| Model             | Description                            |
|-------------------|----------------------------------------|
| Random Forest     | 200 trees, high accuracy, fast         |
| Gradient Boosting | 200 estimators, lr=0.1, strong         |
| Extra Trees       | 200 trees, randomized splits           |
| SVM               | RBF kernel, C=10, good on boundaries   |
| Voting Ensemble   | Soft-vote of RF + GB + ET (best)       |

Feature engineering: volume, dimensional weight, aspect ratio, density, LW ratio.

---

## Free Deployment

### Backend → Render.com (free)
1. Push to GitHub
2. Connect repo to render.com
3. Set environment variables
4. Deploy

### Frontend → Netlify (free)
1. Drag and drop `packai.html` to netlify.com/drop
2. Done — live URL in seconds

---

## Target customers (India)

- D2C brand warehouses shipping 500–5000 orders/month
- 3PL providers handling multiple brands
- Small ecommerce businesses on Shiprocket / Delhivery

## Pricing suggestion (from earlier analysis)
- Starter: ₹2,000/month (up to 500 orders)
- Growth: ₹5,000/month (up to 2,000 orders)
- Pro: ₹10,000/month (unlimited)
