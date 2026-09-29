# -*- coding: utf-8 -*-
"""
Companion web calculator for the viral pneumonia ICU mortality prediction model.

UI refresh (2026-09-29): light medical-blue theme, sidebar patient inputs,
risk card + risk gauge + SHAP explanation in the main panel.
Functional logic (model loading, feature order, 0.2/0.4 risk tiers, SHAP shape
handling, feature-importance fallback) is unchanged from the manuscript version.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap  # noqa: F401  (kept so the dependency is exercised at import time)
import streamlit as st

# ------------------------------------------------------------------ theme
BLUE, RED = "#1565C0", "#C62828"
TIER = {  # manuscript risk tiers: <20% low, 20-40% intermediate, >=40% high
    "Low": "#2E7D32",
    "Intermediate": "#F9A825",
    "High": "#C62828",
}

st.set_page_config(
    page_title="Viral Pneumonia ICU Mortality Risk Calculator",
    page_icon="🫁",
    layout="wide",
)

st.markdown(
    """
    <style>
      .card{background:#fff;border:1px solid #dbe4ef;border-radius:14px;
            padding:1.1rem 1.4rem;box-shadow:0 1px 4px rgba(21,101,192,.10);}
      .risknum{font-size:3rem;font-weight:800;line-height:1.05;}
      .muted{color:#5b6b7c;}
      .gauge{position:relative;height:24px;border-radius:8px;overflow:hidden;
             display:flex;}
      .gz{height:100%;}
      .gmark{position:absolute;top:0;width:3px;height:100%;background:#1f2933;
             transform:translateX(-50%);border-radius:2px;}
      .gaugecap{position:relative;height:1.1rem;font-size:.78rem;color:#5b6b7c;}
      .gaugecap span{position:absolute;transform:translateX(-50%);}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ model load
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load('medical_risk_model.pkl')
    if hasattr(model, 'feature_names_in_'):
        names = list(model.feature_names_in_)
    else:
        names = joblib.load('medical_risk_model_features.pkl')
    try:
        explainer = joblib.load('medical_risk_model_explainer.pkl')
    except Exception:
        explainer = None
    return model, names, explainer


try:
    model, feature_names, explainer = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ---------------------------------------------------------------- header
st.markdown(
    "<h1 style='margin-bottom:2px'>🫁 Viral Pneumonia ICU Mortality "
    "Risk Calculator</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "ExtraTrees machine-learning model for 30-day mortality in ICU patients with "
    "viral pneumonia · internal validation AUC 0.859 · external validation AUC 0.930"
)
st.caption("✔ Model loaded successfully")

# -------------------------------------------------------- sidebar inputs
# label, min, max, default  (defaults identical to the original calculator)
INPUTS = {
    'apsiii':                   ("APS III score", 0, 300, 50),
    'admission_age':            ("Admission age (years)", 0, 120, 65),
    'height':                   ("Height (cm)", 100.0, 220.0, 170.0),
    'resp_rate_min':            ("Minimum respiratory rate (breaths/min)", 0.0, 60.0, 12.0),
    'dbp_max':                  ("Maximum diastolic blood pressure (mmHg)", 0.0, 200.0, 80.0),
    'aado2_calc_min':           ("Minimum A-aDO2 (mmHg)", 0.0, 800.0, 100.0),
    'pao2fio2ratio_min':        ("Minimum PaO2/FiO2 ratio", 0.0, 600.0, 300.0),
    'bilirubin_total_min':      ("Minimum total bilirubin (mg/dL)", 0.0, 50.0, 1.0),
    'wbc_max':                  ("Maximum WBC count (x10^9/L)", 0.0, 50.0, 10.0),
    'ptt_max':                  ("Maximum PTT (seconds)", 0.0, 200.0, 35.0),
    'charlson_comorbidity_index': ("Charlson comorbidity index", 0, 20, 2),
}
GROUPS = [
    ("Demographics", ["admission_age", "height"]),
    ("Severity & physiology", ["apsiii", "resp_rate_min", "aado2_calc_min",
                               "pao2fio2ratio_min", "dbp_max"]),
    ("Laboratory tests & comorbidity", ["bilirubin_total_min", "wbc_max",
                                        "ptt_max", "charlson_comorbidity_index"]),
]

with st.sidebar:
    st.markdown("### 🧾 Patient parameters")
    values = {}
    for gname, feats in GROUPS:
        st.markdown(f"**{gname}**")
        for f in feats:
            label, lo, hi, default = INPUTS[f]
            values[f] = st.number_input(label, min_value=lo, max_value=hi,
                                        value=default, key=f)
    predict = st.button("Predict 30-day mortality risk", type="primary")

missing = [f for f in feature_names if f not in values]
if missing:
    st.error(f"Input panel is missing model features: {missing}")
    st.stop()

# ------------------------------------------------------------- placeholder
if not predict:
    st.markdown(
        "<div class='card'>Enter the patient's parameters in the left panel, then "
        "click <b>Predict 30-day mortality risk</b>.<br><br>Risk tiers: "
        f"<span style='color:{TIER['Low']}'><b>Low &lt; 20%</b></span> · "
        f"<span style='color:{TIER['Intermediate']}'><b>Intermediate 20–40%</b></span> · "
        f"<span style='color:{TIER['High']}'><b>High ≥ 40%</b></span></div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ------------------------------------------------------------- prediction
try:
    input_df = pd.DataFrame([[values[f] for f in feature_names]],
                            columns=feature_names)
    prob = float(model.predict_proba(input_df)[0][1])
    surv = 1.0 - prob
    tier = "Low" if prob < 0.2 else ("Intermediate" if prob < 0.4 else "High")
    color = TIER[tier]

    # ---- risk cards -----------------------------------------------------
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            f"<div class='card' style='border-left:6px solid {color}'>"
            f"<div class='muted' style='font-size:.95rem'>"
            f"Predicted 30-day mortality risk</div>"
            f"<div class='risknum' style='color:{color}'>{prob * 100:.1f}%</div>"
            f"<div class='muted'>predicted probability {prob:.3f} · "
            f"tier thresholds 20% / 40%</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='card' style='border-left:6px solid {color}'>"
            f"<div class='muted' style='font-size:.95rem'>Risk category</div>"
            f"<div style='font-size:1.9rem;font-weight:750;color:{color}'>"
            f"{tier} risk</div>"
            f"<div class='muted'>Survival probability {surv:.3f}</div></div>",
            unsafe_allow_html=True,
        )

    # ---- risk gauge -----------------------------------------------------
    pct = max(0.0, min(100.0, prob * 100))
    st.markdown(
        f"<div style='margin:.4rem 0 1.4rem'>"
        f"<div class='gauge'>"
        f"<div class='gz' style='width:20%;background:{TIER['Low']}'></div>"
        f"<div class='gz' style='width:20%;background:{TIER['Intermediate']}'></div>"
        f"<div class='gz' style='width:60%;background:{TIER['High']}'></div>"
        f"<div class='gmark' style='left:{pct:.1f}%'></div></div>"
        f"<div class='gaugecap'>"
        f"<span style='left:0%;transform:none'>0%</span>"
        f"<span style='left:20%'>20%</span>"
        f"<span style='left:40%'>40%</span>"
        f"<span style='left:100%;transform:translateX(-100%)'>100%</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ---- SHAP explanation ----------------------------------------------
    st.subheader("Why this prediction?")

    def shap_values_1d(expl, df, n):
        """Same shape-handling as the original calculator."""
        sv = expl.shap_values(df)
        if isinstance(sv, list):
            sv = sv[1][0] if len(sv) == 2 else sv[0][0]
        sv = np.array(sv).flatten()
        if len(sv) == 2 * n:
            sv = sv[n:]
        elif len(sv) != n:
            sv = sv[:n]
        return sv.astype(float)

    def importance_fallback_chart():
        """Model feature importance shown when SHAP is unavailable."""
        imp = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(imp)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.barh([INPUTS[feature_names[i]][0] for i in order],
                imp[order], color=BLUE, alpha=.85)
        ax.axvline(0, color="#9aa7b5", lw=.8)
        ax.set_xlabel("Feature importance")
        ax.set_title("Model feature importance ranking",
                     loc="left", fontweight="bold")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        return fig

    if explainer is not None:
        try:
            sv = shap_values_1d(explainer, input_df, len(feature_names))

            order = np.argsort(sv)
            labels = [INPUTS[feature_names[i]][0] for i in order]
            vals = sv[order]

            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            ax.barh(labels, vals,
                    color=[RED if v > 0 else BLUE for v in vals], alpha=.85)
            ax.axvline(0, color="#9aa7b5", lw=.8)
            ax.set_xlabel("SHAP value (impact on mortality risk)")
            ax.set_title("Feature impact on this prediction",
                         loc="left", fontweight="bold")
            span = max(abs(vals.min()), abs(vals.max()), 1e-3)
            ax.set_xlim(vals.min() - span * .28, vals.max() + span * .28)
            for y, v in enumerate(vals):
                if abs(v) > 1e-4:
                    ax.text(v + span * .02 * (1 if v >= 0 else -1), y,
                            f"{v:+.3f}", va="center",
                            ha="left" if v >= 0 else "right",
                            fontsize=9, color="#37474f")
            ax.tick_params(labelsize=9)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

            try:
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    ev = ev[1] if len(ev) == 2 else ev[0]
                st.caption(f"Base value: {float(ev):.3f}")
            except Exception:
                pass

            # ---- SHAP table ------------------------------------------------
            shap_table = pd.DataFrame({
                "Feature": [INPUTS[f][0] for f in feature_names],
                "SHAP value": np.round(sv, 3),
                "Effect": ["Increases risk" if v > 0 else "Decreases risk"
                           for v in sv],
            }).iloc[np.argsort(-np.abs(sv))]
            st.dataframe(shap_table.reset_index(drop=True),
                         use_container_width=True, hide_index=True)

        except Exception as shap_error:
            st.warning(f"SHAP explanation unavailable: {shap_error}")
            if hasattr(model, 'feature_importances_'):
                st.pyplot(importance_fallback_chart())
    else:
        st.warning("SHAP explainer not available.")
        if hasattr(model, 'feature_importances_'):
            st.pyplot(importance_fallback_chart())

    with st.expander("How to read the SHAP explanation"):
        st.markdown(
            f"- <span style='color:{RED}'><b>Red bars (positive SHAP values)</b></span> "
            "push this patient's prediction toward <b>higher</b> mortality risk\n"
            f"- <span style='color:{BLUE}'><b>Blue bars (negative SHAP values)</b></span> "
            "push the prediction toward <b>lower</b> mortality risk\n"
            "- Bar length = strength of the feature's influence on this prediction\n"
            "- Base value = the model's average prediction across training patients",
            unsafe_allow_html=True,
        )

except Exception as e:
    st.error(f"Prediction error: {e}")

# ---------------------------------------------------------------- footer
st.divider()
st.caption(
    "This calculator is a demonstration companion tool for the authors' manuscript "
    "and does not replace clinical judgement. Model: ExtraTrees classifier; "
    "internal validation AUC 0.859 (n = 163 held-out patients), external validation "
    "AUC 0.930 (n = 115 patients)."
)
