import streamlit as st
import numpy as np
import joblib
import time
import csv
import os
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Can This Mix? | Drug Interaction Checker",
    page_icon="💊",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main { background-color: #f4f6fb; }

    .card-danger {
        padding: 22px 24px;
        border-radius: 14px;
        background: #fff5f5;
        border-left: 6px solid #e53e3e;
        margin-bottom: 12px;
        color: #1a202c !important;
    }
    .card-danger h3, .card-danger p, .card-danger strong { color: #1a202c !important; }

    .card-safe {
        padding: 22px 24px;
        border-radius: 14px;
        background: #f0fff4;
        border-left: 6px solid #38a169;
        margin-bottom: 12px;
        color: #1a202c !important;
    }
    .card-safe h3, .card-safe p, .card-safe strong { color: #1a202c !important; }

    .card-info {
        padding: 16px 20px;
        border-radius: 10px;
        background: #ebf4ff;
        border-left: 5px solid #4299e1;
        font-size: 0.9rem;
        margin-top: 10px;
        color: #1a202c !important;
    }
    .card-info strong, .card-info em { color: #1a202c !important; }
    </style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
FP_RADIUS     = 2
FP_NBITS      = 2048
FEEDBACK_FILE = "user_feedback.csv"

# ── LOAD ASSETS (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model     = joblib.load('best_model.pkl')
    drug_dict = joblib.load('drug_smiles_dictionary.pkl')
    return model, drug_dict

model, drug_dict = load_assets()
drug_names = sorted(drug_dict.keys())

# ── CORE FUNCTIONS ─────────────────────────────────────────────────────────────
def get_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS),
        dtype=np.int8
    )

def predict_interaction(drug_a: str, drug_b: str):
    """
    Symmetry fix: sort the pair alphabetically before fingerprint
    concatenation so (Drug A, Drug B) == (Drug B, Drug A) always.
    """
    name1, name2 = sorted([drug_a, drug_b])
    fp1 = get_fingerprint(drug_dict[name1])
    fp2 = get_fingerprint(drug_dict[name2])

    if fp1 is None or fp2 is None:
        return None, None

    X     = np.hstack([fp1, fp2]).reshape(1, -1)
    proba = model.predict_proba(X)[0]          # single pass through all 200 trees
    prediction  = int(np.argmax(proba))        # 0 or 1 — derived, not a second call
    probability = float(proba[1])
    return prediction, probability

def save_feedback(drug1, drug2, prediction_label, usefulness, ease, comment):
    """Append one feedback row to CSV. Creates the file with a header if missing."""
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'timestamp', 'drug_1', 'drug_2', 'prediction',
                'usefulness_1_to_5', 'ease_of_use_1_to_5', 'comment'
            ])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            drug1, drug2, prediction_label,
            usefulness, ease, comment.strip()
        ])

# ── HEADER ─────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=64)
with col_title:
    st.markdown("## 💊 Can This Mix?")
    st.markdown(
        "**AI-powered drug interaction checker** — enter two medications to see "
        "if they may cause a dangerous reaction when taken together."
    )
st.divider()

# ── MAIN LAYOUT ────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([4, 6], gap="large")

with col_input:
    st.subheader("🔍 Check Two Medications")
    st.caption("Start typing in either box to search by drug name.")

    drug1 = st.selectbox("First medication:", drug_names, index=0)
    drug2 = st.selectbox("Second medication:", drug_names,
                         index=min(1, len(drug_names) - 1))

    st.write("")
    check_btn = st.button("Check for Interaction", type="primary",
                          use_container_width=True)

    st.divider()
    st.caption(f"🗄️ **{len(drug_names):,} drugs** in the database")
    st.caption("⚡ Results delivered in under 100 ms")
    st.caption("⚠️ For informational use only — always consult a healthcare professional.")

# ── RESULT PANEL ───────────────────────────────────────────────────────────────
with col_result:

    if check_btn:

        if drug1 == drug2:
            st.warning("Please choose two *different* medications.")

        else:
            t0 = time.time()
            prediction, probability = predict_interaction(drug1, drug2)
            latency_ms = (time.time() - t0) * 1000

            if prediction is None:
                st.error(
                    "⚠️ We couldn't read the chemical structure for one or both "
                    "drugs. Please try a different combination."
                )
            else:
                # Persist result for the feedback widget below
                st.session_state['last_result'] = {
                    'drug1': drug1, 'drug2': drug2,
                    'prediction': prediction, 'probability': probability
                }
                st.session_state['feedback_submitted'] = False

                st.subheader("📋 Result")

                # ── INTERACTION DETECTED ──────────────────────────────────────
                if prediction == 1:
                    risk_label = "High Risk" if probability >= 0.80 else "Moderate Risk"
                    bar_color  = "🔴"        if probability >= 0.80 else "🟠"

                    st.markdown(f"""
<div class="card-danger">
<h3>{bar_color} Potential Interaction Detected — {risk_label}</h3>
<p>
Our model flagged <strong>{drug1}</strong> and <strong>{drug2}</strong> as a
combination that may cause an <strong>adverse drug reaction</strong>.
</p>
<p>Interaction likelihood: <strong>{probability:.0%}</strong></p>
</div>
""", unsafe_allow_html=True)

                    st.progress(probability,
                                text=f"Interaction likelihood: {probability:.0%}")

                    st.markdown("""
<div class="card-info">
⚠️ <strong>What to do:</strong> Do not take these medications together without
first speaking to your doctor or pharmacist. They can advise on a safe
alternative or adjust your dosage.
</div>
""", unsafe_allow_html=True)

                # ── NO INTERACTION ────────────────────────────────────────────
                else:
                    safety_conf = 1.0 - probability

                    st.markdown(f"""
<div class="card-safe">
<h3>🟢 No Known Interaction Found</h3>
<p>
<strong>{drug1}</strong> and <strong>{drug2}</strong> were <strong>not flagged</strong>
as a dangerous combination in our training database.
</p>
<p>Safety confidence: <strong>{safety_conf:.0%}</strong></p>
</div>
""", unsafe_allow_html=True)

                    st.progress(safety_conf,
                                text=f"Safety confidence: {safety_conf:.0%}")

                    st.markdown("""
<div class="card-info">
ℹ️ <strong>Please note:</strong> A "no interaction" result does not guarantee
these drugs are safe for <em>you</em> specifically. Individual health factors,
age, weight, and other medications all matter. Always confirm with a healthcare
professional before making any changes.
</div>
""", unsafe_allow_html=True)

                st.caption(f"⚡ Analysis completed in `{latency_ms:.1f} ms`")

    elif 'last_result' not in st.session_state:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("👈 Select two medications on the left, then click **Check for Interaction**.")

# ── FEEDBACK SECTION ───────────────────────────────────────────────────────────
if 'last_result' in st.session_state:
    st.divider()
    result = st.session_state['last_result']

    if not st.session_state.get('feedback_submitted', False):

        st.subheader("📝 Rate This Result")
        st.caption(
            "Your feedback is used to evaluate and improve this system — "
            "it takes less than a minute and helps the research team."
        )

        USEFULNESS_LABELS = {
            1: "1 — Not useful at all",
            2: "2 — Slightly useful",
            3: "3 — Somewhat useful",
            4: "4 — Very useful",
            5: "5 — Extremely useful",
        }
        EASE_LABELS = {
            1: "1 — Very difficult",
            2: "2 — Difficult",
            3: "3 — Neutral",
            4: "4 — Easy",
            5: "5 — Very easy",
        }

        fb_col1, fb_col2 = st.columns(2)

        with fb_col1:
            usefulness = st.radio(
                "How **useful** was this result?",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: USEFULNESS_LABELS[x],
                index=2,
                key="fb_usefulness"
            )

        with fb_col2:
            ease = st.radio(
                "How **easy** was the app to use?",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: EASE_LABELS[x],
                index=2,
                key="fb_ease"
            )

        comment = st.text_area(
            "Any comments or suggestions? *(optional)*",
            placeholder="e.g. The result was clear and I knew what to do next…",
            height=90,
            key="fb_comment"
        )

        if st.button("Submit Feedback", type="secondary"):
            prediction_label = (
                "Interaction Detected" if result['prediction'] == 1
                else "No Interaction"
            )
            save_feedback(
                result['drug1'], result['drug2'],
                prediction_label,
                usefulness, ease,
                st.session_state.get('fb_comment', '')
            )
            st.session_state['feedback_submitted'] = True
            st.rerun()

    else:
        st.success(
            "✅ Thank you! Your feedback has been recorded and will be included "
            "in the project evaluation."
        )