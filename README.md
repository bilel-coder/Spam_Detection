# 🛡️ Spam Detection — ML Deployment Project

> A production-grade spam classifier built with scikit-learn + FastAPI + Docker.  
> Module: Déploiement de Modèles de ML — ECE 2025-2026

---

## 📁 Project Structure

```
spam-detection/
├── data/
│   ├── raw/                    ← place spam.csv here
│   └── processed/              ← auto-generated after preprocessing
├── models/
│   ├── artifacts/
│   │   └── spam_pipeline.joblib  ← saved best model
│   └── metrics/
│       └── metrics.json          ← comparison results
├── src/
│   ├── spamdet/
│   │   ├── __init__.py
│   │   ├── config.py           ← all paths & hyperparameters
│   │   ├── data.py             ← data loading
│   │   ├── preprocessing.py    ← text cleaning & feature engineering
│   │   ├── train.py            ← model comparison & selection
│   │   ├── inference.py        ← prediction engine
│   │   └── schemas.py          ← Pydantic request/response models
│   └── api/
│       ├── main.py             ← FastAPI app entry point
│       └── routes.py           ← API endpoints
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── tests/
│   ├── conftest.py
│   ├── test_inference.py
│   └── test_api.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/<your-username>/spam-detection.git
cd spam-detection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the dataset

Download `spam.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it in:

```
data/raw/spam.csv
```

### 3. Preprocess data

```bash
cd src
python -m spamdet.preprocessing
```

### 4. Train & compare models

```bash
python -m spamdet.train
```

This trains **5 models**, compares them, and saves the best one to `models/artifacts/spam_pipeline.joblib`.

### 5. Run the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open: http://localhost:8000  
Swagger docs: http://localhost:8000/docs

---

## 🐳 Docker

### Build

```bash
docker build -t spam-detection .
```

### Run

```bash
docker run -p 8000:8000 spam-detection
```

---

## 📡 API Endpoints

| Method | Endpoint              | Description                     |
|--------|-----------------------|---------------------------------|
| GET    | `/api/v1/health`      | Health check                    |
| POST   | `/api/v1/predict`     | Classify a single message       |
| POST   | `/api/v1/predict/batch` | Classify up to 100 messages   |
| GET    | `/api/v1/model/info`  | Model metrics & comparison      |

### Example — single prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have won a FREE iPhone!"}'
```

Response:
```json
{
  "label": "spam",
  "label_id": 1,
  "confidence": 0.9812,
  "spam_proba": 0.9812,
  "is_spam": true
}
```

### Example — batch prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Free prize! Call now!", "See you at 5pm"]}'
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 🤖 Models Compared

| Model               | Notes                                  |
|---------------------|----------------------------------------|
| Logistic Regression | Fast, strong baseline                  |
| Naive Bayes         | Classic spam filter, very fast         |
| Linear SVM (SGD)    | Robust to high-dimensional TF-IDF      |
| Random Forest       | Ensemble, captures non-linear patterns |
| Gradient Boosting   | Often best accuracy, slower to train   |

The model with the **highest F1-macro** on the held-out test set is automatically selected.

---

## ☁️ Deployment (HuggingFace Spaces)

1. Create a Space on [huggingface.co/spaces](https://huggingface.co/spaces) — select **Docker** SDK
2. Push your repository:
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/spam-detection
   git push space main
   ```
3. Your API will be live at `https://<username>-spam-detection.hf.space`

---

## 👥 Team

| Name | Role |
|------|------|
| ...  | ...  |

---

## 📄 License

MIT — for academic use.
