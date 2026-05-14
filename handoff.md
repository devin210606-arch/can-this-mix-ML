# 💊 Can This Mix? — Machine Learning Project Handoff

> ⚠️ **IMPORTANT: EXTERNAL DATA LINKS** > Due to GitHub's 100MB file size limit, the trained models and raw datasets are hosted on Google Drive. 
> **Download here:** https://drive.google.com/drive/folders/10dWcM5xXWYQL8BGpbLCUBwJuUkEp2z-U?usp=sharing
> *Please place `best_model.pkl`, `drug_smiles_dictionary.pkl`, and `db_drug_interactions.csv` in the root directory before running the app.*

## 1. Project Overview
A Drug-Drug Interaction (DDI) prediction system using a Random Forest classifier trained on 380,000+ interactions. The system converts drug names to SMILES and then into 2048-bit Morgan Fingerprints to predict adverse interaction probability.

## 2. Technical Stack
- **Language:** Python 3.13
- **ML Framework:** Scikit-Learn
- **Cheminformatics:** RDKit
- **Frontend:** Streamlit
- **Optimization:** Intel Core Ultra (OpenMP/n_jobs) & Local Dictionary caching

## 3. Key Features
- **Zero-Leakage Training:** Implemented `GroupShuffleSplit` and `GroupKFold` to ensure the model generalizes to new, unseen drugs (Cold-Start scenario).
- **High Performance:** Offline SMILES dictionary reduces prediction latency to < 200ms.
- **Open-World Assumption:** Synthetic negative class generation for robust binary classification.

## 4. Performance Metrics (Final Results)
- **Mean ROC-AUC:** ~0.89 - 0.92 (Drug-Aware Split)
- **Primary Goal:** Recall-optimized to prioritize patient safety over precision.

## 5. How to Run
1. Install dependencies: `pip install -r requirements.txt` (or manually install rdkit, streamlit, scikit-learn).
2. Ensure the `.pkl` and `.csv` files from Google Drive are in the folder.
3. Run the command: `streamlit run app.py`