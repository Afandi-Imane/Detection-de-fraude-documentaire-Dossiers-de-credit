"""
Interface Streamlit -- Système de détection de fraude documentaire (crédit).

Lancement :
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path

import streamlit as st

from pipeline import run_pipeline
from signals.visual_signal import generate_heatmap_overlay, get_model_bundle

# =====================================================================
# CONFIGURATION GENERALE
# =====================================================================

PROJECT_ROOT = Path(".")
RAW_DIR = PROJECT_ROOT / "data" / "raw"

TEMPLATE_LABELS = {
    "CIN_V1": "Carte d'identité (CIN)",
    "QUITTANCE_LYDEC_V1": "Quittance",
    "QUITTANCE_AMENDIS_V1": "Quittance",
    "QUITTANCE_MAROC_TELECOM_V1": "Quittance",
    "RIB_CIH_V1": "RIB (CIH)",
    "RIB_ATTIJARI_V1": "RIB (Attijariwafa Bank)",
    "RELEVE_CIH_V1": "Relevé bancaire (CIH)",
    "RELEVE_ATTIJARI_V1": "Relevé bancaire (Attijariwafa Bank)",
    "BULLETIN_SALAIRE_V1": "Bulletin de salaire",
    "ATTESTATION_TRAVAIL_V1": "Attestation de travail",
    "TEMPLATE_UNKNOWN": "Non reconnu",
}

SIGNAL_LABELS = {
    "S_business": "Règles métier",
    "S_intra": "Cohérence intra-dossier",
    "S_inter": "Cohérence inter-dossiers",
    "S_llm": "Analyse sémantique (LLM)",
    "S_visual": "Analyse visuelle (GANomaly)",
}

# Traduction des règles métier réellement utilisées (voir business_rules_signal.py)
BUSINESS_RULE_LABELS = {
    "THREE_CONSECUTIVE_PERIODS": ("Périodes des bulletins non consécutives", "Bulletins de salaire"),
    "ONE_SALARY_TRANSFER_PER_MONTH": ("Plusieurs virements de salaire le même mois", "Relevés bancaires"),
    "SALARY_AMOUNT_MATCH": ("Montant du salaire incohérent avec le virement bancaire", "Bulletins de salaire / Relevés bancaires"),
    "SALARY_LABEL_EMPLOYER": ("Nom de l'employeur incohérent entre les documents", "Bulletins de salaire / Attestation de travail"),
    "TRANSACTION_TOTALS": ("Totaux des transactions incohérents", "Relevés bancaires"),
    "BALANCE_EQUATION": ("Équation de solde incorrecte (solde départ + mouvements ≠ solde final)", "Relevés bancaires"),
    "BALANCE_CHAINING": ("Rupture de chaînage entre les soldes des relevés successifs", "Relevés bancaires"),
}

# Traduction des règles inter-dossiers réellement utilisées (voir inter_dossier_signal.py)
INTER_RULE_LABELS = {
    "CIN_MULTIPLE_NAMES": "Même numéro de CIN associé à des noms différents",
    "RIB_MULTIPLE_IDENTITIES": "Même RIB utilisé par plusieurs identités différentes",
    "IBAN_MULTIPLE_IDENTITIES": "Même IBAN utilisé par plusieurs identités différentes",
    "NAME_MULTIPLE_CIN": "Même nom complet associé à plusieurs numéros de CIN",
}


def prettify_rule_name(code: str) -> str:
    """Formatte un code de regle inconnu en texte lisible (repli generique)."""
    return code.replace("_", " ").capitalize()

VERDICT_STYLE = {
    "VALIDATION_AUTOMATIQUE": {"color": "#1E7A34", "bg": "#E7F6EC", "icon": "✅", "label": "VALIDATION AUTOMATIQUE"},
    "VERIFICATION_COMPLEMENTAIRE": {"color": "#8A5B00", "bg": "#FFF4DA", "icon": "🟠", "label": "VÉRIFICATION COMPLÉMENTAIRE"},
    "ESCALADE_ANALYSTE": {"color": "#A61B1B", "bg": "#FCE7E7", "icon": "🔴", "label": "ESCALADE VERS UN ANALYSTE"},
}

st.set_page_config(
    page_title="Détection de fraude — Dossiers de crédit",
    page_icon="🔍",
    layout="wide",
)

# =====================================================================
# STYLE (CSS personnalisé pour un rendu plus professionnel)
# =====================================================================

st.markdown("""
<style>
    .main { background-color: #F7F9FC; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .app-header {
        padding: 1.2rem 1.6rem; border-radius: 12px;
        background: linear-gradient(90deg, #10233F 0%, #1B3A66 100%);
        color: white; margin-bottom: 1.5rem;
    }
    .app-header h1 { margin: 0; font-size: 1.6rem; }
    .app-header p { margin: 0.2rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }
    .signal-card {
        border-radius: 10px; padding: 1rem; text-align: center;
        background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-top: 4px solid #1B3A66;
    }
    .signal-card .value { font-size: 1.7rem; font-weight: 700; color: #10233F; }
    .signal-card .label { font-size: 0.82rem; color: #55606E; margin-top: 2px; }
    .signal-card .weight { font-size: 0.72rem; color: #8A94A3; margin-top: 4px; }
    .verdict-banner {
        border-radius: 12px; padding: 1.4rem 1.8rem; margin: 1rem 0;
        display: flex; justify-content: space-between; align-items: center;
    }
    .verdict-banner .title { font-size: 1.3rem; font-weight: 700; }
    .verdict-banner .score { font-size: 2.2rem; font-weight: 800; }
    .doc-row {
        padding: 0.55rem 1rem; border-radius: 8px; margin-bottom: 0.4rem;
        display: flex; justify-content: space-between; align-items: center;
        color: #10233F !important;
    }
    .doc-row span { color: #10233F !important; }
    .doc-row .doc-name { font-weight: 700; }
    .doc-row .doc-type { font-size: 0.82rem; color: #55606E !important; font-weight: 400; }
    .doc-suspect { background: #FCE7E7; border-left: 5px solid #A61B1B; }
    .doc-normal { background: #EAF3EC; border-left: 5px solid #2E8B4F; }
    .rule-card {
        background: white; border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.5rem;
        border-left: 5px solid #A61B1B; color: #10233F !important;
    }
    .rule-card b { color: #10233F !important; }
    .rule-card .rule-doc { font-size: 0.8rem; color: #55606E !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>🔍 Système de détection de fraude — Dossiers de crédit</h1>
    <p>Analyse automatisée multi-signaux : règles métier, cohérence réseau, analyse sémantique et visuelle</p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# ETAT DE SESSION
# =====================================================================

if "result" not in st.session_state:
    st.session_state.result = None
if "dossier_id" not in st.session_state:
    st.session_state.dossier_id = None

# =====================================================================
# ZONE D'UPLOAD
# =====================================================================

st.subheader("1. Déposer les documents du dossier")

col_upload, col_id = st.columns([3, 1])
with col_upload:
    uploaded_files = st.file_uploader(
        "Sélectionnez tous les documents du dossier (CIN, quittance, RIB, relevés, bulletins, attestation)",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    )
with col_id:
    dossier_id_input = st.text_input("Identifiant du dossier", value=f"D_WEB_{int(time.time())}")

analyser_clic = st.button("🚀 Lancer l'analyse", type="primary", disabled=not uploaded_files)

# =====================================================================
# LANCEMENT DU PIPELINE
# =====================================================================

if analyser_clic and uploaded_files:
    dossier_id = dossier_id_input.strip() or f"D_WEB_{int(time.time())}"
    dossier_raw_dir = RAW_DIR / dossier_id

    if dossier_raw_dir.exists():
        shutil.rmtree(dossier_raw_dir)
    dossier_raw_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        dest_path = dossier_raw_dir / uploaded_file.name
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    with st.spinner(f"Analyse en cours pour le dossier {dossier_id}… (OCR, classification, 5 signaux)"):
        try:
            result = run_pipeline(input_dir=str(dossier_raw_dir), project_root=str(PROJECT_ROOT))
            st.session_state.result = result
            st.session_state.dossier_id = dossier_id
        except Exception as error:
            st.error(f"Erreur pendant l'analyse : {error}")
            st.session_state.result = None

# =====================================================================
# AFFICHAGE DES RESULTATS
# =====================================================================

result = st.session_state.result

if result is not None:
    dossier_id = st.session_state.dossier_id

    st.subheader("2. Résultat de l'analyse")

    # ------------------- Récapitulatif des documents -------------------
    manifest_path = PROJECT_ROOT / "data" / "classification" / f"{dossier_id}_manifest.json"
    template_counts: Counter = Counter()
    documents_list = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        documents_list = manifest.get("documents", [])
        template_counts = Counter(
            TEMPLATE_LABELS.get(d.get("template_id"), d.get("template_id", "Inconnu"))
            for d in documents_list if d.get("status") == "CLASSIFIED"
        )

    n_total = result.get("classification", {}).get("total", len(documents_list))
    n_classified = result.get("classification", {}).get("classified", 0)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Documents analysés", f"{n_classified} / {n_total}", help="Documents classifiés avec succès")
    with col_b:
        if template_counts:
            detail_txt = "  •  ".join(f"**{count}** {label}" for label, count in template_counts.items())
            st.markdown(f"**Détail par type :** {detail_txt}")

    # ------------------- Bandeau verdict -------------------
    decision = result.get("decision_finale", {})
    verdict = decision.get("verdict", "INCONNU")
    score_final = decision.get("score_final", 0.0)
    style = VERDICT_STYLE.get(verdict, {"color": "#333", "bg": "#EEE", "icon": "❔", "label": verdict})

    st.markdown(f"""
    <div class="verdict-banner" style="background:{style['bg']}; border-left: 8px solid {style['color']};">
        <div class="title" style="color:{style['color']};">{style['icon']} {style['label']}</div>
        <div class="score" style="color:{style['color']};">Score : {score_final:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------- 5 cartes signaux -------------------
    st.markdown("#### Détail des 5 signaux")
    scores_detail = decision.get("details_scores", {})
    poids = decision.get("poids_utilises", {})

    cols = st.columns(5)
    for col, (key, label) in zip(cols, SIGNAL_LABELS.items()):
        value = scores_detail.get(key, 0.0)
        weight = poids.get(key, 0.0)
        with col:
            st.markdown(f"""
            <div class="signal-card">
                <div class="value">{value:.3f}</div>
                <div class="label">{label}</div>
                <div class="weight">Poids : {weight:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------- Documents suspects (signal visuel) -------------------
    st.markdown("#### 📄 Documents analysés (signal visuel)")

    visual_result = result.get("signals", {}).get("visual")
    suspect_docs = []
    if visual_result and visual_result.get("documents"):
        for doc in visual_result["documents"]:
            is_suspect = doc.get("status") == "SUSPECT"
            css_class = "doc-suspect" if is_suspect else "doc-normal"
            icon = "🔴 SUSPECT" if is_suspect else "🟢 Normal"
            type_label = TEMPLATE_LABELS.get(doc.get("template_id"), doc.get("template_id"))
            st.markdown(f"""
            <div class="doc-row {css_class}">
                <span><span class="doc-name">{icon} — {doc.get('file')}</span><br>
                <span class="doc-type">{type_label}</span></span>
                <span><b>score : {doc.get('score', 0):.3f}</b></span>
            </div>
            """, unsafe_allow_html=True)
            if is_suspect:
                suspect_docs.append(doc)
    else:
        st.info("Signal visuel indisponible pour ce dossier.")

    # ------------------- Document original + heatmap, cote a cote -------------------
    if suspect_docs:
        st.markdown("#### 🔥 Documents suspects — zoom sur les zones anormales")
        for doc in suspect_docs:
            type_label = TEMPLATE_LABELS.get(doc.get("template_id"), doc.get("template_id"))
            st.markdown(f"**{doc.get('file')}** — {type_label} (score visuel : {doc.get('score', 0):.3f})")
            col_original, col_heatmap = st.columns(2)
            image_path = doc.get("image_path")
            try:
                with col_original:
                    st.image(image_path, caption="Document original", width=340)
                from signals.visual_signal import TEMPLATE_TO_MODEL_DIR
                model_key = TEMPLATE_TO_MODEL_DIR.get(doc["template_id"])
                bundle = get_model_bundle("signals/visual_engine/models", model_key)
                heatmap_img = generate_heatmap_overlay(bundle, image_path)
                with col_heatmap:
                    st.image(heatmap_img, caption="Zones suspectes (rouge/jaune)", width=340)
            except Exception as error:
                st.warning(f"Heatmap indisponible pour {doc['file']} : {error}")
            st.markdown("---")

    st.markdown("---")

    # ------------------- Détails par signal (dépliables) -------------------
    st.markdown("#### 🔎 Détails par signal")

    with st.expander("▶ Règles métier", expanded=True):
        business = result.get("signals", {}).get("business_rules")
        if business:
            summary = business.get("summary", {})
            triggered = summary.get("triggered_scoring_rules", [])
            st.write(f"**{summary.get('anomalies_count', 0)} anomalie(s)** détectée(s) sur {summary.get('checks_available', 0)} règles vérifiées.")
            if triggered:
                for code in triggered:
                    label, docs_concernes = BUSINESS_RULE_LABELS.get(code, (prettify_rule_name(code), "Document(s) non précisé(s)"))
                    st.markdown(f"""
                    <div class="rule-card">
                        <b>⚠️ {label}</b><br>
                        <span class="rule-doc">Type de fraude : <code>{code}</code> &nbsp;|&nbsp; Document(s) concerné(s) : {docs_concernes}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("Aucune règle métier déclenchée.")
        else:
            st.write("Signal indisponible pour ce dossier.")

    with st.expander("▶ Cohérence intra-dossier"):
        intra = result.get("signals", {}).get("intra_dossier")
        if intra:
            summary = intra.get("summary", {})
            st.write(f"**{summary.get('anomalies_count', 0)} anomalie(s)** ({summary.get('critical_anomalies_count', 0)} critique(s)) "
                     f"sur {summary.get('checks_available', 0)} vérifications de cohérence interne au dossier.")
            triggered = summary.get("triggered_rules") or summary.get("triggered_scoring_rules") or []
            for code in triggered:
                st.markdown(f"""
                <div class="rule-card">
                    <b>⚠️ {prettify_rule_name(code)}</b><br>
                    <span class="rule-doc">Code : <code>{code}</code></span>
                </div>
                """, unsafe_allow_html=True)
            if not triggered and summary.get("anomalies_count", 0) == 0:
                st.success("Aucune incohérence détectée entre les documents du dossier.")
        else:
            st.write("Signal indisponible pour ce dossier.")

    with st.expander("▶ Cohérence inter-dossiers"):
        inter = result.get("signals", {}).get("inter_dossiers")
        if inter:
            summary = inter.get("summary", {})
            st.write(f"**{summary.get('anomalies_count', 0)} anomalie(s)** ({summary.get('critical_anomalies_count', 0)} critique(s)) "
                     f"avec **{summary.get('linked_dossiers_count', 0)} dossier(s)** lié(s) dans l'historique.")
            triggered = summary.get("triggered_rules") or summary.get("triggered_scoring_rules") or []
            for code in triggered:
                label = INTER_RULE_LABELS.get(code, prettify_rule_name(code))
                st.markdown(f"""
                <div class="rule-card">
                    <b>⚠️ {label}</b><br>
                    <span class="rule-doc">Type de fraude réseau : <code>{code}</code></span>
                </div>
                """, unsafe_allow_html=True)
            if not triggered and summary.get("anomalies_count", 0) == 0:
                st.success("Aucun lien suspect avec d'autres dossiers du système.")
        else:
            st.write("Signal indisponible pour ce dossier.")

    with st.expander("▶ Analyse sémantique (LLM)"):
        llm = result.get("signals", {}).get("llm")
        if llm:
            st.write(f"**Verdict :** {llm.get('verdict')}")
            st.write(f"**Explication :** {llm.get('explication')}")
            if llm.get("anomalies"):
                st.write("**Anomalies relevées :**")
                for anomalie in llm["anomalies"]:
                    st.write(f"- {anomalie}")
        else:
            st.write("Signal indisponible pour ce dossier.")

    with st.expander("▶ Résultat brut complet (JSON)"):
        st.json(result)

else:
    st.info("Déposez les documents du dossier ci-dessus, puis cliquez sur *Lancer l'analyse*.")