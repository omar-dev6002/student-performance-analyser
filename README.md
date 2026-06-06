# 🎓 Student Performance Analyser

A data analysis and prediction project to identify factors affecting student academic performance using Python and Machine Learning.

---

## 📌 Project Goals

- Analyse key features affecting student grades (study hours, attendance, parental education, etc.)
- Visualise performance trends using Matplotlib & Seaborn
- Build a predictive ML model using Scikit-learn
- Deploy as an interactive Streamlit app

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data wrangling | Python · Pandas · NumPy |
| Visualisation | Matplotlib · Seaborn |
| Machine Learning | Scikit-learn |
| Deployment *(planned)* | Streamlit |

---

## 📁 Project Structure

```
student-performance-analyser/
│
├── data/
│   ├── raw/                  # Original Kaggle dataset (not committed)
│   └── processed/            # Cleaned & encoded dataset
│
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory Data Analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Load & inspect raw data
│   ├── preprocessing.py      # Cleaning, encoding, feature engineering
│   └── model.py              # Train, evaluate, save ML models
│
├── outputs/
│   ├── figures/              # Saved plots
│   └── models/               # Saved .pkl model files
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

### 1. Clone the repo
```bash
git clone https://github.com/omar-dev6002/student-performance-analyser.git
cd student-performance-analyser
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
**Option A — Kaggle CLI (recommended)**
```bash
# Set up your Kaggle API key first: https://www.kaggle.com/docs/api
kaggle datasets download -d whenamancodes/student-performance -p data/raw --unzip
```

**Option B — Manual**
1. Go to https://www.kaggle.com/datasets/whenamancodes/student-performance
2. Download and unzip into `data/raw/`
3. Rename if needed so the file is `data/raw/student-mat.csv`

### 5. Launch Jupyter
```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## 📊 Dataset

[Student Performance Dataset](https://www.kaggle.com/datasets/whenamancodes/student-performance) — UCI Machine Learning Repository  
395 students · 33 features · Math course (`student-mat.csv`)

Key features: `studytime`, `failures`, `absences`, `Medu`, `Fedu`, `internet`, `higher`, `G1`, `G2`, `G3` (final grade, target)

---

## 🗺️ Roadmap

- [x] Project scaffold & folder structure
- [ ] EDA notebook (`01_eda.ipynb`)
- [ ] Preprocessing pipeline (`src/preprocessing.py`)
- [ ] ML model training & evaluation (`src/model.py`)
- [ ] Streamlit deployment

---

## 👤 Author

**Omar Farooq Anis**  
B.Tech CSE (AI & Data Science) · Jamia Millia Islamia · Batch 2029  
[GitHub](https://github.com/omar-dev6002) · [LinkedIn](https://linkedin.com/in/omar-farooq-anis)
