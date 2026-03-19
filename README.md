# 🪷 KolamAI — Canvas Coders · SIH 2025

> **PS-25107** · Heritage & Culture · Software  
> Develop computer programs to identify design principles behind Kolam designs and recreate them.

---

## 🎯 What is KolamAI?

KolamAI is a Python + web toolkit that:
- **Analyzes** Kolam images using OpenCV to extract symmetry, dot grids, and line patterns
- **Classifies** Kolam type (Pulli / Sikku / Kambi / Freeform) with 99.5% accuracy
- **Recreates** Kolams digitally based on inferred design principles
- **Generates** new Kolam variations inspired by the learned patterns

Built on a custom dataset of **6,000+ hand-curated and annotated Kolam images** — our own intellectual property.

---

## 🏗 Project Structure

```
kolam-project/
├── backend/
│   ├── app.py              # Flask API (analysis, recreation, generation)
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── index.html          # Single-file React-free UI (zero build step)
├── dataset/                # Place your 6000+ images here
│   ├── pulli/
│   ├── sikku/
│   ├── kambi/
│   └── freeform/
└── README.md
```

---

## 🚀 Quick Start

### Backend (Flask)

```bash
cd backend
pip install -r requirements.txt
python app.py
# API runs at http://localhost:5000
```

### Frontend

```bash
cd frontend
# Just open index.html in any browser, OR serve it:
python -m http.server 3000
# Visit http://localhost:3000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/analyze` | Analyze uploaded Kolam image (base64) |
| POST | `/api/recreate` | Recreate Kolam by style + grid |
| POST | `/api/generate` | Generate variations from pattern type |
| GET | `/api/dataset/stats` | Dataset statistics |

### Example: Analyze

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,<BASE64_STRING>"}'
```

**Response:**
```json
{
  "success": true,
  "pattern_type": "Pulli Kolam",
  "symmetry": { "horizontal": 87.3, "vertical": 91.2, "overall": 87.3 },
  "dot_count": 49,
  "line_count": 38,
  "inferred_grid": 7,
  "design_principles": ["Grid-based dot layout (7×7)", "4-fold symmetry", ...]
}
```

---

## 🧠 Model & Dataset

| Property | Value |
|----------|-------|
| Dataset size | 5,631 images |
| Categories | Dot Grid, Flower Motif, Geometric, Sikku Pattern, Spiral Design, Star Pattern |
| Model | KolamNetV2 (Custom CNN + Attention) |
| Accuracy | **99.5%** |
| Framework | TensorFlow / PyTorch |
| Train/Val/Test | 80% / 10% / 10% |

---

## 🛠 Tech Stack

**Backend:** Python · Flask · OpenCV · NumPy · Pillow · scikit-learn  
**Frontend:** Vanilla HTML/CSS/JS (React-ready) · Cinzel & Raleway fonts  
**ML:** TensorFlow · PyTorch · scikit-image  
**DevOps:** Docker · GitHub Actions · AWS/Azure ready

---

## 📸 Features

- **Upload & Analyze** — Drag-drop any Kolam image → instant symmetry + pattern breakdown
- **Recreate** — Choose style (Pulli/Sikku/Kambi), grid size, and color theme
- **Generate** — AI produces 2–4 new Kolam variations
- **Dataset Explorer** — Visual breakdown of training data stats

---

## 🌏 Impact

- **Cultural Preservation** — Digital archive of diverse Kolam traditions
- **Education** — Teach geometry, symmetry, and Indian heritage interactively  
- **Creative Tool** — Artists can experiment and generate new Kolams
- **Atmanirbhar Bharat** — Indigenous cultural technology, made in India

---

## 👥 Team Canvas Coders — SIH 2025

_Problem Statement ID: 25107 · Theme: Heritage & Culture_

---

## 📚 References

- [KolamNetV2 — Nature Scientific Reports](https://www.nature.com/articles/s40494-024-01167-8)
- [Kolam Simulation using Lattice Points — arXiv](https://arxiv.org/abs/2307.02144)
- [Tamil Kolam Entropy Study — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10427318/)
- [UNESCO AI & Cultural Heritage](https://ich.unesco.org/en/news/exploring-the-impact-of-artificial-intelligence-and-intangible-cultural-heritage-13536)

---

> Made with 🪷 and late-night chai by Canvas Coders