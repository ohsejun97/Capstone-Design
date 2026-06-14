"""
app.py — FusionDTI Streamlit Web Application

Drug-Target Interaction 예측 + 시각화 데모.

Run (프로젝트 루트에서):
    streamlit run web_pipeline/app.py
"""

import sys
import time
from io import BytesIO
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent   # Capstone-Design/
sys.path.insert(0, str(ROOT))

# ── Page config (반드시 첫 번째 Streamlit 호출) ────────────────────────────────
st.set_page_config(
    page_title="FusionDTI — Drug-Target Interaction Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background-color: #f7f9fc; }
[data-testid="stSidebar"]          { background-color: #1a1a2e; }
[data-testid="stSidebar"] * { color: #e8eaf6 !important; }
[data-testid="stSidebar"] hr { border-color: #3a3a5e; }
[data-testid="stSidebar"] .stButton button {
    background: #2c2c54; border: 1px solid #4a4a8a;
    color: #c5cae9; border-radius: 8px;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #3d3d7a; border-color: #7986cb;
}

/* ── Result cards ── */
.result-banner {
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 14px;
    font-size: 1rem;
    line-height: 1.6;
}
.banner-very-strong { background:#d1f2eb; border-left: 6px solid #1abc9c; }
.banner-strong      { background:#d5f5e3; border-left: 6px solid #2ecc71; }
.banner-moderate    { background:#fef9e7; border-left: 6px solid #f39c12; }
.banner-weak        { background:#fdecea; border-left: 6px solid #e74c3c; }

/* ── Info card ── */
.info-card {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.info-card h4 { margin: 0 0 10px 0; color: #1a1a2e; }

/* ── Pipeline badge ── */
.pipeline-step {
    display: inline-flex; align-items: center;
    background: #e8eaf6; border-radius: 20px;
    padding: 3px 12px; font-size: 0.78rem;
    color: #3949ab; margin: 3px 2px;
    border: 1px solid #c5cae9;
}

/* ── Section header ── */
.section-header {
    font-size: 1.05rem; font-weight: 700;
    color: #1a237e; margin-bottom: 4px;
    border-bottom: 2px solid #e8eaf6;
    padding-bottom: 6px;
}

/* ── Kd badge ── */
.kd-badge {
    display: inline-block; background: #e3f2fd;
    border-radius: 20px; padding: 4px 16px;
    font-size: 1.0rem; color: #1565c0;
    font-weight: 600; border: 1px solid #90caf9;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cached helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_dti_models():
    from tools.dti_tool import _load_models
    _load_models()


def _smiles_to_png(smiles: str, size: tuple = (380, 260)) -> bytes | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _pkd_to_kd_str(pkd: float) -> str:
    kd_M = 10 ** (-pkd)
    if kd_M < 1e-12:
        return f"{kd_M * 1e15:.1f} fM"
    if kd_M < 1e-9:
        return f"{kd_M * 1e12:.1f} pM"
    if kd_M < 1e-6:
        return f"{kd_M * 1e9:.2f} nM"
    if kd_M < 1e-3:
        return f"{kd_M * 1e6:.2f} µM"
    return f"{kd_M * 1e3:.2f} mM"


def _pkd_meta(pkd: float) -> dict:
    if pkd >= 9.0:
        return {"label": "Very Strong Binding", "color": "#1abc9c",
                "bg": "#d1f2eb", "cls": "banner-very-strong", "emoji": "✅"}
    if pkd >= 7.0:
        return {"label": "Strong Binding", "color": "#2ecc71",
                "bg": "#d5f5e3", "cls": "banner-strong", "emoji": "🟢"}
    if pkd >= 5.0:
        return {"label": "Moderate Binding", "color": "#f39c12",
                "bg": "#fef9e7", "cls": "banner-moderate", "emoji": "🟡"}
    return {"label": "Weak / No Significant Binding", "color": "#e74c3c",
            "bg": "#fdecea", "cls": "banner-weak", "emoji": "🔴"}


def _gauge_chart(pkd: float) -> go.Figure:
    m = _pkd_meta(pkd)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pkd,
        number={
            "font": {"size": 52, "color": m["color"], "family": "Arial Black"},
            "suffix": "",
        },
        title={
            "text": (
                f"pKd Score<br>"
                f"<span style='font-size:0.85em;color:{m['color']}'>"
                f"{m['emoji']} {m['label']}</span>"
            ),
            "font": {"size": 16, "color": "#1a1a2e"},
        },
        gauge={
            "axis": {
                "range": [0, 12],
                "tickwidth": 1,
                "tickcolor": "#90a4ae",
                "tickvals": [0, 3, 5, 7, 9, 12],
                "ticktext": ["0", "3", "5", "7", "9", "12"],
            },
            "bar": {"color": m["color"], "thickness": 0.22},
            "bgcolor": "#f8f9fa",
            "borderwidth": 2,
            "bordercolor": "#dee2e6",
            "steps": [
                {"range": [0,  5],  "color": "#fdecea"},
                {"range": [5,  7],  "color": "#fef9e7"},
                {"range": [7,  9],  "color": "#d5f5e3"},
                {"range": [9, 12],  "color": "#d1f2eb"},
            ],
            "threshold": {
                "line": {"color": "#1565c0", "width": 3},
                "thickness": 0.75,
                "value": 7.0,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=70, b=10, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial"},
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

EXAMPLES = {
    "Imatinib × ABL1": {
        "drug_mode": "Drug Name",
        "prot_mode": "Gene Name",
        "drug_val":  "Imatinib",
        "prot_val":  "ABL1",
    },
    "Aspirin × COX-1 (PTGS1)": {
        "drug_mode": "Drug Name",
        "prot_mode": "Gene Name",
        "drug_val":  "Aspirin",
        "prot_val":  "PTGS1",
    },
    "Sildenafil × PDE5A": {
        "drug_mode": "Drug Name",
        "prot_mode": "Gene Name",
        "drug_val":  "Sildenafil",
        "prot_val":  "PDE5A",
    },
    "Gefitinib × EGFR": {
        "drug_mode": "Drug Name",
        "prot_mode": "Gene Name",
        "drug_val":  "Gefitinib",
        "prot_val":  "EGFR",
    },
}

with st.sidebar:
    st.markdown(
        "<h2 style='color:#e8eaf6;margin-top:0'>🧬 Agentic FusionDTI</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#b0bec5;font-size:0.9rem'>Drug–Target Interaction Predictor</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("<p class='section-header' style='color:#90caf9'>⚙️ Model Architecture</p>", unsafe_allow_html=True)
    st.code(
        "Drug SMILES\n"
        "  → ft-ChemBERTa  → [768d]  ┐\n"
        "                             ├→ MLP → pKd\n"
        "Protein AA Seq               │\n"
        "  → SaProt-650M  → [1280d] ──┘",
        language=None,
    )

    st.markdown("<p class='section-header' style='color:#90caf9'>📊 Performance</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Pearson r", "0.892")
    c2.metric("RMSE", "1.01 pKd")
    st.caption("BindingDB test set (~16K pairs)")

    st.divider()
    st.markdown("<p class='section-header' style='color:#90caf9'>📌 Quick Examples</p>", unsafe_allow_html=True)

    for label, vals in EXAMPLES.items():
        if st.button(label, use_container_width=True, key=f"ex_{label}"):
            st.session_state["drug_mode_sel"]  = vals["drug_mode"]
            st.session_state["prot_mode_sel"]  = vals["prot_mode"]
            st.session_state["drug_text"]      = vals["drug_val"]
            st.session_state["prot_text"]      = vals["prot_val"]
            st.session_state["trigger_predict"] = True
            st.rerun()

    st.divider()
    st.markdown(
        "<p style='color:#78909c;font-size:0.8rem;text-align:center'>"
        "Capstone Design 2025<br>오세준 (2021270607)</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main layout
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## 🧬 Drug-Target Interaction Prediction")
st.markdown(
    "Binding affinity (pKd) prediction between a drug molecule and a protein target "
    "using **SaProt-650M** (protein structure encoder) + **fine-tuned ChemBERTa** (drug encoder)."
)
st.divider()

left, right = st.columns([1, 1.5], gap="large")

# ── LEFT: Input panel ─────────────────────────────────────────────────────────
with left:
    st.markdown("### 💊 Drug")

    drug_mode_options = ["Drug Name", "SMILES"]
    drug_mode_idx = drug_mode_options.index(
        st.session_state.get("drug_mode_sel", "Drug Name")
    )
    drug_mode = st.radio(
        "Input mode", drug_mode_options,
        index=drug_mode_idx,
        horizontal=True,
        key="drug_mode_radio",
    )

    if drug_mode == "Drug Name":
        drug_val = st.text_input(
            "Drug name (English generic name)",
            value=st.session_state.get("drug_text", "Imatinib"),
            placeholder="e.g. Imatinib, Aspirin, Sildenafil, Gefitinib",
            key="drug_name_input",
        )
    else:
        drug_val = st.text_area(
            "SMILES string",
            value=st.session_state.get("drug_text", ""),
            placeholder="e.g. CC1=C(C=C(C=C1)NC(=O)...",
            height=80,
            key="drug_smiles_input",
        )

    st.markdown("### 🧫 Protein Target")

    prot_mode_options = ["Gene Name", "Amino Acid Sequence"]
    prot_mode_idx = prot_mode_options.index(
        st.session_state.get("prot_mode_sel", "Gene Name")
    )
    prot_mode = st.radio(
        "Input mode", prot_mode_options,
        index=prot_mode_idx,
        horizontal=True,
        key="prot_mode_radio",
    )

    if prot_mode == "Gene Name":
        prot_val = st.text_input(
            "Gene / protein name",
            value=st.session_state.get("prot_text", "ABL1"),
            placeholder="e.g. ABL1, EGFR, HMGCR, PDE5A, PTGS1",
            key="prot_name_input",
        )
    else:
        prot_val = st.text_area(
            "Amino acid sequence (1-letter code)",
            value=st.session_state.get("prot_text", ""),
            placeholder="MKTAYIAKQRQISFVKSH...",
            height=130,
            key="prot_seq_input",
        )

    st.markdown("")
    predict_clicked = st.button(
        "🔬  Predict Binding Affinity",
        type="primary",
        use_container_width=True,
    )

    # Sidebar example shortcut triggers prediction immediately
    if st.session_state.get("trigger_predict"):
        predict_clicked = True
        del st.session_state["trigger_predict"]

    # ── Pipeline legend ───────────────────────────────────────────────────────
    st.markdown("")
    st.markdown(
        "<div style='font-size:0.78rem;color:#607d8b;line-height:1.8'>"
        "<span class='pipeline-step'>① PubChem</span>"
        "<span class='pipeline-step'>② UniProt</span>"
        "<span class='pipeline-step'>③ ChemBERTa</span>"
        "<span class='pipeline-step'>④ SaProt-650M</span>"
        "<span class='pipeline-step'>⑤ MLP Head → pKd</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── RIGHT: Results panel ──────────────────────────────────────────────────────
with right:
    st.markdown("### 📊 Results")

    if not predict_clicked:
        st.markdown("""
<div class='info-card'>
<h4>How to use</h4>
<ol style='margin:0;padding-left:1.2rem;color:#455a64'>
<li>Enter a <b>drug name</b> (e.g. <i>Imatinib</i>) or paste a SMILES string.</li>
<li>Enter a <b>gene/protein name</b> (e.g. <i>ABL1</i>) or paste a sequence.</li>
<li>Click <b>Predict Binding Affinity</b>.</li>
</ol>
</div>

<div class='info-card'>
<h4>pKd Interpretation Guide</h4>
<table style='width:100%;font-size:0.88rem;border-collapse:collapse'>
<tr style='background:#f5f5f5'><th style='padding:5px 8px;text-align:left'>pKd</th><th>K<sub>d</sub></th><th>Binding</th></tr>
<tr><td style='padding:4px 8px'>≥ 9.0</td><td>≤ 1 nM</td><td>✅ Very Strong</td></tr>
<tr><td>7.0 – 9.0</td><td>1–100 nM</td><td>🟢 Strong</td></tr>
<tr><td>5.0 – 7.0</td><td>0.1–10 µM</td><td>🟡 Moderate</td></tr>
<tr><td>< 5.0</td><td>> 10 µM</td><td>🔴 Weak / None</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    else:
        # ── Pipeline execution ────────────────────────────────────────────────
        final_smiles: str | None = None
        aa_seq: str | None = None
        drug_meta: dict = {}
        prot_meta: dict = {}
        pred_result: dict = {}
        error_msg: str | None = None

        t_start = time.time()

        with st.status("Running prediction pipeline...", expanded=True) as status_box:

            # Step 1: Model load
            st.write("⚙️ Ensuring DTI models are loaded...")
            _load_dti_models()
            st.write("✅ Models ready")

            # Step 2: Drug resolution
            drug_val_clean = (drug_val or "").strip()
            if not drug_val_clean:
                error_msg = "Drug input is empty."
            elif drug_mode == "Drug Name":
                st.write(f"💊 Resolving **{drug_val_clean}** via PubChem...")
                from tools.pubchem_tool import resolve_drug_name
                drug_meta = resolve_drug_name(drug_val_clean)
                if "error" in drug_meta:
                    error_msg = f"Drug not found: {drug_meta['error']}"
                else:
                    final_smiles = drug_meta["smiles"]
                    st.write(
                        f"✅ **{drug_meta['name']}** — "
                        f"CID {drug_meta['cid']}, "
                        f"MW {drug_meta.get('mol_weight', '?')} g/mol"
                    )
            else:
                final_smiles = drug_val_clean
                drug_meta = {"name": "Custom SMILES", "smiles": final_smiles}
                st.write("✅ Using provided SMILES")

            # Step 3: Protein resolution
            if not error_msg:
                prot_val_clean = (prot_val or "").strip()
                if not prot_val_clean:
                    error_msg = "Protein input is empty."
                elif prot_mode == "Gene Name":
                    st.write(f"🧫 Resolving **{prot_val_clean}** via UniProt...")
                    from tools.uniprot_tool import resolve_protein_name
                    prot_meta = resolve_protein_name(prot_val_clean)
                    if "error" in prot_meta:
                        error_msg = f"Protein not found: {prot_meta['error']}"
                    else:
                        aa_seq = prot_meta["sequence"]
                        reviewed = "✅ Swiss-Prot" if prot_meta.get("reviewed") else "TrEMBL"
                        st.write(
                            f"✅ **{prot_meta['gene']}** ({prot_meta['uniprot_id']}) "
                            f"— {prot_meta['seq_length']} aa, {reviewed}"
                        )
                else:
                    aa_seq = prot_val_clean.upper()
                    prot_meta = {
                        "gene": "Custom",
                        "uniprot_id": "N/A",
                        "seq_length": str(len(aa_seq)),
                        "sequence": aa_seq,
                    }
                    st.write(f"✅ Using provided sequence ({len(aa_seq)} aa)")

            # Step 4: DTI prediction
            if not error_msg:
                st.write("🔬 Running DTI prediction (SaProt-650M + ft-ChemBERTa + MLP)...")
                from tools.dti_tool import predict_binding
                pred_result = predict_binding(final_smiles, aa_seq)
                if "error" in pred_result:
                    error_msg = f"Prediction error: {pred_result['error']}"
                else:
                    elapsed = time.time() - t_start
                    status_box.update(
                        label=f"✅ Complete ({elapsed:.1f}s)",
                        state="complete",
                        expanded=False,
                    )

        # ── Render results ────────────────────────────────────────────────────
        if error_msg:
            st.error(f"❌ {error_msg}")

        elif pred_result and "error" not in pred_result:
            pkd   = pred_result["pKd"]
            meta  = _pkd_meta(pkd)
            kd_str = _pkd_to_kd_str(pkd)

            # ① pKd gauge
            st.plotly_chart(_gauge_chart(pkd), use_container_width=True)

            # ② Kd value + 3Di cache tag
            col_kd, col_cache = st.columns([1, 1])
            with col_kd:
                st.markdown(
                    f"<div style='text-align:center'>"
                    f"<span class='kd-badge'>K<sub>d</sub> ≈ {kd_str}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_cache:
                if pred_result.get("used_3di"):
                    st.success("🏗️ 3Di structure tokens used", icon=None)
                else:
                    st.warning("⚠️ No cached structure (3Di fallback)", icon=None)

            # ③ Interpretation banner
            st.markdown(
                f"<div class='result-banner {meta['cls']}'>"
                f"<b>{meta['emoji']} {meta['label']}</b><br>"
                f"<span style='font-size:0.88rem'>{pred_result['interpretation']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.divider()

            # ④ Drug + Protein cards
            dcol, pcol = st.columns(2, gap="medium")

            with dcol:
                st.markdown("<p class='section-header'>💊 Drug</p>", unsafe_allow_html=True)
                mol_png = _smiles_to_png(final_smiles)
                if mol_png:
                    st.image(mol_png, use_container_width=True, caption="2D Structure (RDKit)")
                else:
                    st.caption("(2D structure unavailable)")

                if drug_meta.get("formula"):
                    st.markdown(
                        f"| Property | Value |\n|---|---|\n"
                        f"| Formula | `{drug_meta['formula']}` |\n"
                        f"| MW | {drug_meta.get('mol_weight','?')} g/mol |\n"
                        f"| CID | {drug_meta.get('cid','N/A')} |"
                    )
                    if drug_meta.get("pubchem_url"):
                        st.markdown(f"[🔗 View on PubChem]({drug_meta['pubchem_url']})")

                with st.expander("SMILES", expanded=False):
                    st.code(final_smiles, language=None)

            with pcol:
                st.markdown("<p class='section-header'>🧫 Protein Target</p>", unsafe_allow_html=True)

                gene = prot_meta.get("gene", "N/A")
                uid  = prot_meta.get("uniprot_id", "N/A")
                slen = prot_meta.get("seq_length", str(len(aa_seq)))
                reviewed_str = "✅ Swiss-Prot reviewed" if prot_meta.get("reviewed") else ""

                st.markdown(
                    f"| Property | Value |\n|---|---|\n"
                    f"| Gene | **{gene}** |\n"
                    f"| UniProt | `{uid}` |\n"
                    f"| Protein | {prot_meta.get('name', 'N/A')} |\n"
                    f"| Organism | {prot_meta.get('organism', 'N/A')} |\n"
                    f"| Length | {slen} aa |"
                )
                if reviewed_str:
                    st.caption(reviewed_str)
                if uid not in ("N/A", "custom"):
                    st.markdown(
                        f"[🔗 View on UniProt](https://www.uniprot.org/uniprotkb/{uid})"
                    )

                with st.expander("Sequence preview", expanded=False):
                    preview = aa_seq[:120] + ("..." if len(aa_seq) > 120 else "")
                    st.code(preview, language=None)
                    st.caption(f"Full length: {len(aa_seq)} residues")


# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<p style='text-align:center;font-size:0.8rem;color:#90a4ae'>"
    "Agentic FusionDTI &nbsp;·&nbsp; Capstone Design 2025 &nbsp;·&nbsp; 오세준 (2021270607)<br>"
    "SaProt-650M + fine-tuned ChemBERTa + MLP Head &nbsp;·&nbsp; "
    "Trained on BindingDB ~80K pairs &nbsp;·&nbsp; Pearson r = 0.892"
    "</p>",
    unsafe_allow_html=True,
)
