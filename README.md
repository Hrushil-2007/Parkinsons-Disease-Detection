# 🧠 Parkinson's Disease Voice Biomarker Detection System

A complete clinical decision-support and interactive predictive system for detecting **Parkinson's Disease (PD)** using biomedical acoustic measurements of sustained phonation.

---

## 📌 Overview

This project transforms the Jupyter / Google Colab machine learning script into a production-grade interactive web application. It uses a **Support Vector Machine (Linear SVM)** trained on patient vocal biomarkers (StandardScaler + SVM with calibrated probability estimation) to accurately differentiate between healthy individuals and patients diagnosed with Parkinson's disease.

### Key Highlights:
- **Accuracy**: ~89.7% on test evaluation.
- **Biomarkers**: Analyzes **22 acoustic features** across vocal pitch, jitter, shimmer, harmonic noise ratios, and nonlinear dynamical complexity.
- **Interactive UI (Streamlit)**:
  - **Single Patient Diagnosis**: Interactive entry with 1-click test presets (Healthy, Parkinson's, Notebook sample vector, Random dataset patient).
  - **Visual Risk Meter & Biomarker Profiling**: Gauge chart, confidence percentage, and deviation comparison against healthy cohort baselines.
  - **Batch Screening**: Upload a CSV of multiple patient acoustic records and download automated diagnostic screening reports.
  - **Model Clinical Performance**: Confusion matrix, classification report, and class distribution analytics.
  - **Acoustic Biomarker Encyclopedia**: Complete clinical dictionary explaining all 22 acoustic features and their connection to neurodegenerative vocal impairment.

---

## 🚀 Quick Start (Running the Application)

### Option 1: Double-Click Launcher (Windows)
Double-click the [`run_app.bat`](file:///C:/Users/Hrushil/.gemini/antigravity/scratch/parkinsons-disease-detection/run_app.bat) file in this directory.

### Option 2: Command Line (PowerShell)
From this project directory:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
The application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🗂️ Project Structure

```
parkinsons-disease-detection/
├── app.py                     # Interactive Streamlit Web Application
├── train.py                   # Model training, validation & artifact export
├── parkinsons.csv             # UCI Parkinson's Voice Biomarker Dataset
├── requirements.txt           # Python dependency specifications
├── run_app.bat                # 1-Click Windows Batch Launcher
├── models/
│   ├── parkinsons_model.pkl   # Trained Support Vector Machine model (SVM)
│   ├── scaler.pkl             # Fitted StandardScaler for 22 vocal features
│   ├── metrics.json           # Train/test accuracy, confusion matrix, means
│   ├── feature_metadata.json  # Feature descriptions, units, and categories
│   └── sample_profiles.json   # 1-click test patient sample vectors
└── .venv/                     # Dedicated Python virtual environment
```

---

## 🧬 Vocal Acoustic Biomarkers Analyzed

1. **Fundamental Frequency ($F_0$)**:
   - `MDVP:Fo(Hz)`: Average vocal fundamental frequency
   - `MDVP:Fhi(Hz)`: Maximum vocal fundamental frequency
   - `MDVP:Flo(Hz)`: Minimum vocal fundamental frequency

2. **Vocal Jitter (Frequency Variation)**:
   - `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`

3. **Vocal Shimmer (Amplitude / Loudness Variation)**:
   - `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`

4. **Harmonics & Noise**:
   - `HNR`: Harmonics-to-Noise Ratio (decreased in Parkinson's dysphonia)
   - `NHR`: Noise-to-Harmonics Ratio (elevated in dysphonic speech)

5. **Nonlinear Dynamical Complexity**:
   - `RPDE`: Recurrence Period Density Entropy
   - `DFA`: Detrended Fluctuation Analysis
   - `spread1`, `spread2`: Nonlinear pitch variation measures
   - `D2`: Correlation dimension
   - `PPE`: Pitch Period Entropy (elevated pitch instability)

---

## 👨‍🔬 Re-training the Model

If you ever update `parkinsons.csv` or want to re-run the training pipeline:
```powershell
.\.venv\Scripts\python.exe train.py
```
This automatically updates `models/` with new weights, metrics, and reference cohort averages.
