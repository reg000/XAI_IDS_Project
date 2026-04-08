# XAI-SHAP Forensic Terminal 🛡️
**Live Network Intrusion Detection System using Explainable AI (XAI)**

This project is a Master's level implementation of a real-time, AI-driven Security Operations Center (SOC) dashboard. It bridges the gap between "Black Box" machine learning and human-readable cybersecurity analysis by intercepting live network traffic, classifying threats using a CatBoost model, and mathematically explaining the AI's reasoning using SHAP (SHapley Additive exPlanations).

## 🚀 Key Features

* **Real-Time Packet Sniffing:** Utilizes `Scapy` to intercept and extract live network flow features.
* **Optimized AI Engine:** Implements `CatBoost`, overcoming Random Forest's limitations with imbalanced network data to accurately classify complex attacks.
* **Live SHAP Explanations:** Instantly calculates the mathematical impact of individual packet features (e.g., *Forward Packet Length*, *SYN Flags*) driving the AI's predictions.
* **Mathematical Expert System:** Replaces generic ML alerts with an intelligent, KNN-style weighted scoring matrix that deduces the exact attack vector (e.g., *Data Exfiltration*, *Port Scanning*, *DoS*) based on SHAP tensors.
* **Interactive Forensic Dashboard:** A Flask-served, Tailwind-styled UI featuring live sparklines, SHAP waterfall charts, and normalized dual-radar analysis for instant SOC triage.

## 🛠️ Tech Stack

**Backend & Data Pipeline**
- Python 3.10+
- Scapy (Network Packet Sniffing)
- Pandas & NumPy (Data Processing)
- Flask (Microservice API Backend)

**Machine Learning & XAI**
- CatBoost (Classification Engine)
- Scikit-Learn (Benchmarking & Scalers)
- SHAP (Explainable AI Mathematics)
- Joblib (Model Persistence)

**Frontend Dashboard**
- HTML5, JS, CSS (Vanilla architecture for maximum portability)
- TailwindCSS (Styling)
- Chart.js (Data Visualization)

## ⚙️ Installation & Usage

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the Forensic Terminal (UI):**
    ```bash
    python dashboard/app.py
    ```
3. **Launch the Network Sniffer/Simulator:(On pre specified PCAP)**
    ```bash
    python test_pcap.py
    ```