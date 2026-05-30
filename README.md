# 💊 Can This Mix? — Drug Interaction Predictor

> **Machine Learning - Final Project**  
> Devin Farrell  · Navizar Maestiano Lubis · Zacky Satria Nugraha

A binary classification system that predicts whether two drugs will have an **adverse interaction**, using Morgan Fingerprints (ECFP4) as chemical features and a Random Forest classifier trained on ~380,000 drug pairs.

---

## 🎯 How It Works

1. Each drug's SMILES string is converted into a **2,048-bit Morgan Fingerprint** (ECFP4, radius=2)
2. The app concatenates the fingerprints in both directions (Forward and Backward) and evaluates both 4,096-dimensional vectors to ensure no asymmetric interactions are missed.
3. A **Random Forest** (200 trees) classifies the pair as *safe* or *adverse interaction*
4. Results are delivered in **under 100 ms** via an offline SMILES lookup dictionary

---

## 📂 Project Structure

```
can-this-mix-ML/
│
├── app.py                          # Streamlit deployment app
├── preprocessing.py                # Fetches SMILES from PubChem & builds dataset
├── eda_drug_interactions.ipynb     # Exploratory Data Analysis
├── model_training.ipynb            # Model training, evaluation & plot generation
│
├── db_drug_interactions.csv   # Core dataset 
├── drug_interactions_with_smiles.csv   # preprocessed dataset (output of preprocessing)
├── best_model.pkl                      # ⚠️ Not included — generate via notebook / inside google drive
├── drug_smiles_dictionary.pkl          # ⚠️ Not included — generate via notebook  / inside google drive
```

> **Note:** `best_model.pkl`, `drug_smiles_dictionary.pkl` `and other csv file are excluded from this repo due to file size. Run `model_training.ipynb` to regenerate them or you can download them in the google drive provided in the setup and installation. you can skip the whole model_training.ipynb,eda_drug_interactions.ipynb and preprocessing.py and immediately run the app.py by downloading the whole google drive and placing to the same folder this means you can skip step 3 and step 4 under.

---

## ⚙️ Setup & Installation

### Requirements
- Python 3.9+
- Git

*Google drive link*
```bash
[📥 Download from Google Drive](https://drive.google.com/drive/folders/10dWcM5xXWYQL8BGpbLCUBwJuUkEp2z-U?usp=sharing)
```

### Step 1 — Clone the Repository

```bash
git clone https://github.com/devin210606-arch/can-this-mix-ML.git
cd can-this-mix-ML
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present or doesnt work, try installing manually:

```bash
pip install streamlit numpy pandas scikit-learn matplotlib seaborn joblib rdkit
```
One heads up: rdkit can be tricky to install via plain pip on some machines.the fix is:

```bash
pip install rdkit-pypi
```

### Step 3 — Verify the Dataset (optional)

Make sure `drug_interactions_with_smiles.csv` is present in the root directory. This is the preprocessed dataset output. If it is missing, run `preprocessing.py` first:

```bash
python preprocessing.py
```

> ⚠️ `preprocessing.py` fetches SMILES structures from the PubChem API and may take a long time depending on your connection. you can Skip this step if the CSV is already present by downloading immediately from the google drive.

### Step 4 — Generate the Model Files (optional)

Open and run all cells in `model_training.ipynb` from top to bottom. This will:

- Generate Morgan Fingerprints for all drug pairs
- Train Logistic Regression (baseline) and Random Forest models
- Evaluate and compare model performance
- Save `best_model.pkl` and `drug_smiles_dictionary.pkl` to the root directory
- Save all 5 evaluation plots

> ⚠️ Training may take **5–15 minutes** depending on your hardware. The notebook is configured to manage memory and parallelism safely. you can skip this step by downloading the pkl file available on the google drive link

### Step 5 — Run the App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

please submit your feedback under the page after trying the app 
thank you!!
or you can also see our application without download by following this link 

```bash
https://huggingface.co/spaces/Turu67/can-this-mix
```
once again thanks for trying our app and giving your feedback we appreciate your feedback!
---

## 📊 Model Performance

| Model | Recall | Precision | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Random Forest 🏆 | **0.7153** | **0.8636** | **0.7825** | **0.8843** |
| Logistic Regression | 0.5552 | 0.6450 | 0.5967 | 0.6701 |

> **Priority metric: Recall** — in a medical context, missing a dangerous interaction (false negative) carries a significantly higher real-world cost than a false alarm.

---

## 🛠️ Key Engineering Decisions

| Problem | Solution |
|---|---|
| No negative samples in raw data | Generated synthetic negatives via Open-World Assumption (13.25% known pair coverage) |
| MemoryError on 4,096-dim matrix | Downcast NumPy arrays to `np.int8` — 8× memory reduction |
| SVM computationally infeasible | Dropped SVM (`O(N³)` complexity on 190k samples); pivoted to Random Forest |
| Nested parallelism crashing RAM | Limited CV to `n_jobs=2`, estimator to `n_jobs=4` |
| PubChem API latency (1–3s) | Pre-exported offline SMILES dictionary loaded via `@st.cache_resource` |
| Inference API mismatch | Standardized both training and inference to `AllChem.GetMorganFingerprintAsBitVect` |
| Asymmetric predictions |Implemented Bidirectional Inference (evaluating both A+B and B+A configurations and taking the highest risk probability) to eliminate Train-Serve skew and maximize Recall. |

---

## 👥 Team

| Name | Student ID |
|---|---|
| Devin Farrell | 2802525792 |
| Navizar Maestiano Lubis | 2802546765 |
| Zacky Satria Nugraha | 2802550270 |

---

> *This tool is for educational and research purposes only. Always consult a licensed healthcare professional before making any medical decisions.*
