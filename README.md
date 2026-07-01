<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:006e2f,100:22c55e&height=220&section=header&text=Agrivion%20AI&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=AI-Powered%20Smart%20Agriculture%20Platform&descAlignY=55&descSize=18" alt="Agrivion AI Banner" width="100%">
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Inter&size=18&duration=3000&pause=1000&color=22C55E&center=true&vCenter=true&width=700&lines=Crop+Disease+Detection+with+AI+%F0%9F%94%AC;Smart+Irrigation+%26+IoT+Monitoring+%F0%9F%92%A7;Yield+Prediction+%26+Crop+Recommendation+%F0%9F%8C%BE;Multilingual+AI+Farming+Assistant+%F0%9F%A4%96" alt="Typing SVG" />
</p> — built with Streamlit, Plotly, and LLM-driven advisory intelligence.

Agrivion AI is a single-file, production-styled Streamlit dashboard that brings together crop disease detection, smart irrigation control, weather analytics, yield prediction, IoT sensor monitoring, and a multilingual AI farming assistant into one unified interface.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.3x-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Screenshots](#-screenshots)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the App](#running-the-app)
- [ML Models](#-ml-models)
- [Pages Guide](#-pages-guide)
- [Configuration](#-configuration)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🌾 Overview

Agrivion AI is designed to give farmers, agronomists, and farm managers a single pane of glass for precision agriculture. It combines:

- **Computer vision** for crop disease detection from leaf/crop images
- **Machine learning** for crop recommendation and yield forecasting
- **LLM-powered chat** (via Groq) for natural-language farming advice in 10+ languages
- **Live weather integration** (via OpenWeather) with actionable advisories
- **Simulated IoT telemetry** for soil moisture, temperature, humidity, pH, and more
- **Interactive farm mapping** with sector-level health and risk visualization

The app is built entirely in Streamlit with a custom design system (CSS tokens, cards, KPI tiles, gradients, and animations) for a polished, dashboard-grade UI — no external frontend framework required.

> **Note:** When API keys or trained model files are not available, the app gracefully falls back to realistic mock data/predictions so the full UI remains explorable out of the box.

---

## ✨ Features

| Category | Capabilities |
|---|---|
| 🔬 **Disease Detection** | Upload leaf/crop images → CNN-based (or mock) disease classification, confidence scores, treatment protocols, spread forecasts, and a sector-level disease risk heatmap |
| 🌱 **Crop Recommendation** | ML-based crop suggestion from N-P-K, temperature, humidity, pH, and rainfall inputs, with top-5 alternatives and a fertilizer plan |
| 💧 **Smart Irrigation** | Live tank levels, soil moisture history, pump/valve toggles, AI-optimized irrigation scheduling, and a 7-day schedule table |
| 🌤️ **Weather Analytics** | Current conditions, 7-day forecast, rainfall projections, 30-day temperature trends, and weather-to-farming advisories |
| 📈 **Yield Prediction** | Configurable crop/soil/area/season inputs, AI confidence score, yield projection curves, feature importance, and year-over-year comparison |
| 📡 **IoT Sensors** | Simulated live sensor grid (moisture, temperature, humidity, pH, EC, CO₂, light, groundwater) with sector filtering and 24-hour trend charts |
| 🗺️ **Live Farm Map** | Visual sector map with health status, sensor overlays, live asset tracking (drones, tractors, bots), and a status legend |
| 🚜 **Farm Management** | Multi-farm registry, farm health scores, new-farm onboarding form, and cross-farm analytics |
| 📋 **Reports & Analytics** | Exportable **PDF** and **CSV** reports, yield/revenue trends, water usage analytics, and disease detection vs. treatment charts |
| 🔔 **Alert Center** | Configurable alerts for disease, moisture, pests, rainfall, and sensor failures |
| 🤖 **AI Chat Assistant** | Groq-powered (Llama3) conversational assistant — "AgriBot" — with quick-question shortcuts and 10+ language support (with a rule-based fallback when no API key is set) |
| ⚙️ **Settings** | API key management, model status dashboard, alert preferences, and farmer profile settings |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| App Framework | [Streamlit](https://streamlit.io/) |
| Visualization | [Plotly](https://plotly.com/python/) (`graph_objects`, `express`) |
| Image Processing | [Pillow (PIL)](https://python-pillow.org/) |
| Data Handling | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| PDF Generation | [fpdf2](https://pyfpdf.github.io/fpdf2/) |
| ML / Disease Detection | TensorFlow / Keras (`.h5` model) — optional |
| ML / Crop Recommendation | scikit-learn (pickled model + scaler) — optional |
| LLM Chat | [Groq API](https://groq.com/) (Llama 3 8B) |
| Weather Data | [OpenWeatherMap API](https://openweathermap.org/api) |
| Styling | Custom CSS design system (Inter font, CSS variables, gradients, animations) |

---

## 📸 Screenshots

> Add screenshots or a demo GIF here once available.

```
docs/
 ├── screenshot-dashboard.png
 ├── screenshot-disease-detection.png
 ├── screenshot-irrigation.png
 └── screenshot-chat.png
```

---

## 📂 Project Structure

```
agrivion-ai/
├── app.py                      # Main Streamlit application (single-file)
├── .env                        # API keys & config (not committed)
├── requirements.txt            # Python dependencies
├── models/
│   ├── AgriVisionAI_Final.h5   # Disease detection CNN model (optional)
│   ├── disease_labels.json     # Class labels for disease model
│   ├── crop_model.pkl          # Crop recommendation model (optional)
│   ├── scaler.pkl              # Feature scaler for crop model (optional)
│   └── label_encoder.pkl       # Label encoder (optional)
├── docs/                        # Screenshots / documentation assets
└── README.md
```

> All optional model files fall back to deterministic/mock predictions if missing, so the app runs fully without them.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- pip (or a virtual environment tool of your choice)
- (Optional) A [Groq API key](https://console.groq.com/) for live AI chat
- (Optional) An [OpenWeatherMap API key](https://openweathermap.org/api) for live weather data

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/agrivion-ai.git
cd agrivion-ai

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`** (recommended):

```txt
streamlit>=1.30
pandas
numpy
plotly
Pillow
requests
python-dotenv
fpdf2
tensorflow        # optional, only needed for real disease model
scikit-learn       # optional, only needed for real crop model
joblib             # optional, used to load scaler
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
MODEL_PATH=models/AgriVisionAI_Final.h5
CROP_MODEL_PATH=models/crop_model.pkl
SCALER_PATH=models/scaler.pkl
LABEL_ENCODER_PATH=models/label_encoder.pkl
```

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | No | Enables live LLM chat via Groq. Without it, AgriBot uses a rule-based fallback responder. |
| `OPENWEATHER_API_KEY` | No | Enables live weather data. Without it, realistic mock weather data is used. |
| `MODEL_PATH` | No | Path to trained disease-detection Keras model. Falls back to mock predictions if missing. |
| `CROP_MODEL_PATH` | No | Path to trained crop-recommendation model (pickled). Falls back to rule-based logic if missing. |
| `SCALER_PATH` | No | Path to the feature scaler used with the crop model. |
| `LABEL_ENCODER_PATH` | No | Path to the label encoder for crop classes (if applicable). |

> You can also manage/update API keys directly from the **Settings → API Keys** tab inside the running app.

### Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` by default.

---

## 🧠 ML Models

Agrivion AI is designed to work with (but does **not require**) two ML models:

1. **Disease Detection Model** — a Keras/TensorFlow `.h5` CNN classifier trained on labeled crop-disease image datasets (e.g., PlantVillage-style datasets). Paired with a `disease_labels.json` file mapping output indices to class names.
2. **Crop Recommendation Model** — a scikit-learn classifier trained on soil/climate parameters (Nitrogen, Phosphorus, Potassium, temperature, humidity, pH, rainfall) to recommend the most suitable crop, paired with a feature scaler.

If either model is absent or fails to load, the app automatically uses **deterministic mock logic** so every page remains fully functional and demoable.

You can upload new model files directly from **Settings → AI Models**.

---

## 🧭 Pages Guide

| # | Page | Description |
|---|---|---|
| 1 | 📊 Dashboard | Farm-wide KPIs, performance trends, crop health donuts, AI insights, live weather, alerts, and autonomous fleet status |
| 2 | 🌿 Crop Monitoring | Growth-stage tracking table, 90-day growth trend chart, per-crop image analysis, and nutrient status |
| 3 | 🔬 AI Disease Detection | Image upload & analysis, detection history, treatment protocols, spread forecast, and disease risk heatmap |
| 4 | 💧 Smart Irrigation | Tank level, soil moisture history, pump/valve controls, AI irrigation insights, 7-day schedule, and sensor grid |
| 5 | 🌤️ Weather Analytics | Live conditions, 7-day forecast, rainfall chart, 30-day temperature trend, and farm advisories |
| 6 | 📈 Yield Prediction | Configurable inputs, AI confidence score, projection curve, feature importance, and year-on-year comparison |
| 7 | 🌱 Crop Recommendation | Soil/climate parameter inputs, top crop match, top-5 alternatives, growing tips, and fertilizer plan |
| 8 | 📡 IoT Sensors | Live sensor tiles across sectors, sector filtering, and 24-hour sensor trend chart |
| 9 | 🚜 Farm Management | Farm registry, new-farm form, and cross-farm yield/water analytics |
| 10 | 🗺️ Live Farm Map | Visual sector map, sensor/disease overlays, asset tracker, and legend |
| 11 | 📋 Reports & Analytics | Exportable PDF/CSV reports, yield/revenue, water usage, and disease analytics tabs |
| 12 | 🔔 Alert Center | Real-time alerts for disease, moisture, weather, and equipment |
| 13 | 🤖 AI Chat Assistant | Multilingual LLM chat (Groq/Llama3) with quick-question shortcuts |
| 14 | ⚙️ Settings | API keys, model status, alert preferences, and farmer profile |

---

## ⚙️ Configuration

- **Theming**: All colors, spacing, and shadows are controlled via CSS variables (`:root`) at the top of the stylesheet — easy to rebrand.
- **Mock Data**: Weather, sensor, and disease data are seeded deterministically so demos are consistent across runs.
- **Session State**: Chat history, selected language, and active page are persisted via `st.session_state`.

---

## 🗺️ Roadmap

- [ ] Real GPS-based interactive map (Leaflet/Mapbox) integration
- [ ] Multi-user auth & role-based access (Admin / Agronomist / Farmer)
- [ ] Database-backed sensor telemetry (replace mock data with live IoT feed)
- [ ] Model retraining pipeline & MLOps integration
- [ ] Mobile-responsive layout improvements
- [ ] Push notifications for critical alerts (SMS/WhatsApp/Email)
- [ ] Offline-first support for low-connectivity rural areas

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please open an issue first for major changes to discuss what you'd like to add.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the rapid app framework
- [Plotly](https://plotly.com/) for interactive charting
- [Groq](https://groq.com/) for high-speed LLM inference
- [OpenWeatherMap](https://openweathermap.org/) for weather data
- The open-source agricultural ML community (e.g., PlantVillage dataset contributors)

---

<p align="center">🌱 <strong>Agrivion AI</strong> — Smart Agriculture Platform &nbsp;·&nbsp; Built with ❤️ for farmers</p>
