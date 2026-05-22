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
import plotly.graph_objects as go
import gspread
from google.oauth2.service_account import Credentials

RDLogger.DisableLog('rdApp.*')

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Can This Mix? | Drug Interaction Checker",
    page_icon="💊",
    layout="wide"
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #f4f6fb; }

    /* ── MEGA CARDS (The "Big Stuff" Results) ── */
    .mega-card {
        padding: 36px;
        border-radius: 20px;
        margin-bottom: 24px;
        color: #ffffff !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .mega-safe {
        background: linear-gradient(135deg, #0a1f11 0%, #051008 100%);
        border: 1px solid #1a4225;
    }
    .mega-danger {
        background: linear-gradient(135deg, #2b0e0e 0%, #140505 100%);
        border: 1px solid #521818;
    }
    .mega-card h2 {
        margin-top: 0;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .mega-safe h2 { color: #68d391 !important; }
    .mega-danger h2 { color: #fc8181 !important; }

    .mega-card p {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #a0aec0 !important;
        margin-bottom: 28px;
    }
    .mega-card strong { color: #e2e8f0 !important; }

    /* ── CONFIDENCE PILL ── */
    .conf-badge {
        display: inline-block;
        font-size: 1.9rem;
        font-weight: 700;
        padding: 12px 36px;
        border-radius: 50px;
        letter-spacing: -0.5px;
    }
    .conf-badge-safe {
        background: rgba(72,187,120,0.15);
        border: 2px solid rgba(72,187,120,0.5);
        color: #68d391;
    }
    .conf-badge-danger {
        background: rgba(229,62,62,0.15);
        border: 2px solid rgba(229,62,62,0.5);
        color: #fc8181;
    }

    /* ── INFO ALERTS ── */
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

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# ── LOAD ASSETS (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model     = joblib.load('best_model.pkl')
    drug_dict = joblib.load('drug_smiles_dictionary.pkl')
    return model, drug_dict

@st.cache_resource
def get_sheet():
    """Connect to Google Sheet using service account credentials."""
    if not os.path.exists("credentials.json"):
        return None, "credentials.json not found — place it in the same folder as app.py"
    try:
        creds  = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet  = client.open_by_key("1UNhKPi4loMFx2sUf_EevnL5u-hUUIA-qx0rOnVM3OWU").sheet1
        if sheet.row_values(1) == []:
            sheet.append_row([
                "timestamp", "drug_1", "drug_2", "prediction",
                "usefulness_1_to_5", "ease_of_use_1_to_5", "comment"
            ])
        return sheet, None
    except gspread.exceptions.SpreadsheetNotFound:
        return None, "Could not open the Google Sheet — check that feedback@canthismix.iam.gserviceaccount.com is added as Editor"
    except Exception as e:
        return None, f"Google Sheets connection failed: {e}"

model, drug_dict = load_assets()
drug_names = sorted(drug_dict.keys())
sheet, sheet_error = get_sheet()

# ── CORE FUNCTIONS ─────────────────────────────────────────────────────────────
def get_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS), dtype=np.int8)

def predict_interaction(drug_a: str, drug_b: str):
    name1, name2 = sorted([drug_a, drug_b])
    fp1 = get_fingerprint(drug_dict[name1])
    fp2 = get_fingerprint(drug_dict[name2])

    if fp1 is None or fp2 is None: return None, None

    X     = np.hstack([fp1, fp2]).reshape(1, -1)
    proba = model.predict_proba(X)[0]
    prediction  = int(np.argmax(proba))
    probability = float(proba[1])

    safety_score = 100 - int(round(probability * 100))
    return prediction, safety_score

def create_horizontal_gauge(safety_score: int):
    """Generates a sleek, wide Plotly gauge chart with absolutely no side numbers."""
    fig = go.Figure(go.Indicator(
        mode = "gauge",
        value = safety_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': "rgba(0,0,0,0)", 'thickness': 0},
            'steps': [
                {'range': [0, 35],  'color': "#E24B4A"},
                {'range': [35, 65], 'color': "#EF9F27"},
                {'range': [65, 100],'color': "#97C459"}
            ],
            'threshold': {
                'line': {'color': "#2b6cb0", 'width': 4},
                'thickness': 1,
                'value': safety_score
            }
        }
    ))
    fig.update_layout(
        height=70,
        margin=dict(t=10, b=20, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def render_info_boxes(safety_score: int):
    """Highlights the active box and keeps inactive boxes readable in dark mode."""
    inactive_bg     = "rgba(255,255,255,0.05)"
    inactive_text   = "#a0aec0"
    inactive_border = "transparent"

    bg1, bg2, bg3   = inactive_bg, inactive_bg, inactive_bg
    txt1, txt2, txt3 = inactive_text, inactive_text, inactive_text
    bd1, bd2, bd3   = inactive_border, inactive_border, inactive_border

    if safety_score <= 35:
        bg1, txt1, bd1 = "#fcebeb", "#791F1F", "#E24B4A"
    elif safety_score <= 65:
        bg2, txt2, bd2 = "#faeeda", "#633806", "#EF9F27"
    else:
        bg3, txt3, bd3 = "#eaf3de", "#27500A", "#97C459"

    return f"""
    <div style="display: flex; gap: 12px; margin-top: 0px; margin-bottom: 20px;">
        <div style="flex: 1; padding: 16px; border-radius: 8px; background: {bg1}; border: 2px solid {bd1}; color: {txt1}; font-size: 0.8rem; line-height: 1.5; text-align: justify; transition: all 0.3s ease;">
            <div style="text-align: center; margin-bottom: 8px;">
                <strong style="font-size: 0.9rem; letter-spacing: 0.5px;">LOW SAFETY</strong><br>
                <span style="font-size: 0.75rem; opacity: 0.8;">0 – 35%</span>
            </div>
            This combination has been associated with adverse reactions. The available data strongly suggests a high likelihood of a dangerous interaction. Do not take together without speaking to a doctor.
        </div>
        <div style="flex: 1; padding: 16px; border-radius: 8px; background: {bg2}; border: 2px solid {bd2}; color: {txt2}; font-size: 0.8rem; line-height: 1.5; text-align: justify; transition: all 0.3s ease;">
             <div style="text-align: center; margin-bottom: 8px;">
                <strong style="font-size: 0.9rem; letter-spacing: 0.5px;">UNCLEAR</strong><br>
                <span style="font-size: 0.75rem; opacity: 0.8;">35 – 65%</span>
            </div>
            We couldn't find enough information about this combination. That doesn't mean it's safe. Consult a pharmacist or doctor before taking these together.
        </div>
        <div style="flex: 1; padding: 16px; border-radius: 8px; background: {bg3}; border: 2px solid {bd3}; color: {txt3}; font-size: 0.8rem; line-height: 1.5; text-align: justify; transition: all 0.3s ease;">
             <div style="text-align: center; margin-bottom: 8px;">
                <strong style="font-size: 0.9rem; letter-spacing: 0.5px;">HIGH SAFETY</strong><br>
                <span style="font-size: 0.75rem; opacity: 0.8;">65 – 100%</span>
            </div>
            This combination does not appear in known interaction records. No database is complete — always confirm with a healthcare professional before making changes.
        </div>
    </div>
    """

def save_feedback(drug1, drug2, prediction_label, usefulness, ease, comment):
    row = [
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        drug1, drug2, prediction_label,
        usefulness, ease, comment.strip()
    ]
    if sheet is not None:
        try:
            sheet.append_row(row)
            return True, None
        except Exception as e:
            err = f"Sheet write failed: {e}"
    else:
        err = sheet_error

    try:
        file_exists = os.path.exists(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp','drug_1','drug_2','prediction','usefulness_1_to_5','ease_of_use_1_to_5','comment'])
            writer.writerow(row)
        return False, err
    except Exception as csv_err:
        return False, f"{err} | CSV also failed: {csv_err}"

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display: flex; align-items: center; gap: 20px; margin-bottom: 8px;">
    <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" width="85">
    <h1 style="font-size: 3.5rem; margin: 0; font-weight: 800; letter-spacing: -1px;">Can This Mix?</h1>
</div>
<p style="font-size: 1.1rem; color: #a0aec0; margin-bottom: 30px;">
    <strong>AI-powered drug interaction checker</strong> — enter two medications to see if they may cause a dangerous reaction when taken together.
</p>
""", unsafe_allow_html=True)

st.divider()

# ── MAIN LAYOUT ────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("<h2 style='font-size: 2rem; margin-top: 0; margin-bottom: 16px;'>🔍 Search Medications</h2>", unsafe_allow_html=True)

    drug1 = st.selectbox("First medication:", drug_names, index=0)
    drug2 = st.selectbox("Second medication:", drug_names, index=min(1, len(drug_names) - 1))

    st.write("")
    check_btn = st.button("Check for Interaction", type="primary", use_container_width=True)

    st.divider()
    st.caption(f"🗄️ **{len(drug_names):,} drugs** in the database")
    st.caption("⚡ Results delivered in under 100 ms")

# ── RESULT PANEL ───────────────────────────────────────────────────────────────
with col_result:

    if check_btn:
        if drug1 == drug2:
            st.warning("Please choose two *different* medications.")
        else:
            t0 = time.time()
            prediction, safety_score = predict_interaction(drug1, drug2)
            latency_ms = (time.time() - t0) * 1000

            if prediction is None:
                st.error("⚠️ We couldn't read the chemical structure for one or both drugs. Please try a different combination.")
            else:
                st.session_state['last_result'] = {'drug1': drug1, 'drug2': drug2, 'prediction': prediction}
                st.session_state['feedback_submitted'] = False

                if prediction == 1:
                    st.markdown(f"""
                    <div class="mega-card mega-danger">
                        <h2>🔴 Potential Interaction Detected</h2>
                        <p>
                            <strong>{drug1}</strong> and <strong>{drug2}</strong> have been flagged with a <strong>High Interaction Risk</strong>.
                            A detailed molecular comparison has detected significant similarities to known high-risk interaction pairs.
                        </p>
                        <div class="conf-badge conf-badge-danger">
                            {safety_score}% safety confidence
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.plotly_chart(create_horizontal_gauge(safety_score), use_container_width=True)
                    st.markdown(render_info_boxes(safety_score), unsafe_allow_html=True)

                    st.markdown("""
                    <div class="card-info">
                        ⚠️ <strong>What to do:</strong> Do not take these medications together without first speaking to your doctor or pharmacist. They can advise on a safe alternative.
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                    <div class="mega-card mega-safe">
                        <h2>🟢 No Known Interaction Found</h2>
                        <p>
                            <strong>{drug1}</strong> and <strong>{drug2}</strong> were <strong>not flagged</strong> as a dangerous combination in our training database.
                            Their molecular profiles do not closely match known interacting pairs.
                        </p>
                        <div class="conf-badge conf-badge-safe">
                            {safety_score}% safety confidence
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.plotly_chart(create_horizontal_gauge(safety_score), use_container_width=True)
                    st.markdown(render_info_boxes(safety_score), unsafe_allow_html=True)

                    st.markdown("""
                    <div class="card-info">
                        ℹ️ <strong>Please note:</strong> A "no interaction" result does not guarantee these drugs are safe for <em>you</em> specifically. Always confirm with a healthcare professional.
                    </div>
                    """, unsafe_allow_html=True)

                st.caption(f"⚡ Analysis completed in `{latency_ms:.1f} ms`")

    elif 'last_result' not in st.session_state:
        pass

# ── FEEDBACK SECTION ───────────────────────────────────────────────────────────
if 'last_result' in st.session_state:
    st.divider()
    result = st.session_state['last_result']

    if not st.session_state.get('feedback_submitted', False):
        st.subheader("📝 Rate This Result")
        st.caption("Your feedback helps the research team.")

        USEFULNESS_LABELS = {1: "1 — Not useful", 2: "2 — Not really useful", 3: "3 — Neutral", 4: "4 — Somewhat useful", 5: "5 — Extremely useful"}
        EASE_LABELS = {1: "1 — Very difficult", 2: "2 — Difficult", 3: "3 — Neutral", 4: "4 — Easy", 5: "5 — Very easy"}

        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            usefulness = st.radio("How **useful** was this result?", options=[1, 2, 3, 4, 5], format_func=lambda x: USEFULNESS_LABELS[x], index=2)
        with fb_col2:
            ease = st.radio("How **easy** was the app to use?", options=[1, 2, 3, 4, 5], format_func=lambda x: EASE_LABELS[x], index=2)

        comment = st.text_area("Any comments or suggestions?", height=70)

        if st.button("Submit Feedback", type="secondary"):
            pred_label = "Interaction Detected" if result['prediction'] == 1 else "No Interaction"
            sent_to_sheets, err = save_feedback(result['drug1'], result['drug2'], pred_label, usefulness, ease, comment)
            st.session_state['feedback_submitted'] = True
            st.session_state['feedback_sheets_ok'] = sent_to_sheets
            st.session_state['feedback_err'] = err
            st.rerun()
    else:
        if st.session_state.get('feedback_sheets_ok'):
            st.success("✅ Thank you! Your feedback has been recorded.")
        else:
            st.success("✅ Thank you! Your feedback has been saved locally.")
            if st.session_state.get('feedback_err'):
                st.warning(f"⚠️ Google Sheets unavailable — {st.session_state['feedback_err']}. Feedback saved to `user_feedback.csv` instead.")