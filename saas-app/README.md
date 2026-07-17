# 🍏 FreshScan AI — SaaS Anti-Gaspillage

Application SaaS complète de détection de qualité des fruits et légumes par IA.

## 🏗️ Architecture

```
saas-app/
├── backend/
│   ├── main.py               ← API FastAPI + inférence TFLite
│   ├── knowledge_base.json   ← Recettes & astuces pour 14 fruits
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── index.css          ← Design system Tailwind
    │   ├── components/
    │   │   └── Navbar.jsx
    │   └── pages/
    │       ├── LandingPage.jsx     ← Page d'accueil SaaS
    │       ├── AnalyzePage.jsx     ← Application de détection
    │       └── PerformancePage.jsx ← Page portfolio / R&D
    ├── tailwind.config.js
    ├── postcss.config.js
    └── package.json
```

## 🚀 Démarrage

### Backend (FastAPI)

```powershell
# Depuis la racine du projet
cd saas-app/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

L'API sera disponible sur **http://localhost:8000**  
Documentation Swagger : **http://localhost:8000/docs**

### Frontend (React + Vite)

```powershell
cd saas-app/frontend
npm install
npm run dev
```

L'interface sera disponible sur **http://localhost:5173**

## 📡 Endpoints API

| Méthode | Route | Description |
|:--------|:------|:------------|
| `GET`  | `/`              | Health check |
| `GET`  | `/api/classes`   | Liste des 28 classes |
| `POST` | `/api/predict`   | Analyser une image |
| `GET`  | `/api/plots`     | Liste des graphiques disponibles |
| `GET`  | `/plots/{file}`  | Afficher un graphique |

## 🤖 Modèle

- **Modèle :** MobileNetV2 (Transfer Learning + Fine-Tuning)
- **Fichier :** `best_model/fruit_quality_mobilenet_fp32.tflite`
- **Input :** `160×160` pixels, prétraitement MobileNetV2 (-1 à 1)
- **Output :** Vecteur de 28 probabilités (softmax)
- **Précision :** 98% sur 5 863 images de validation
