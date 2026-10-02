from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from graph.dossier_graph_loader import load_dossier_json
from graph.neo4j_client import Neo4jClient

from ocr_extraction.dossier_builder import build_dossier_from_folder
from ocr_extraction.extractor_router import process_manifest
from ocr_extraction.extractors.common import load_json, save_json
from ocr_extraction.normalizer import normalize_document
from ocr_extraction.ocr_engine import process_folder as run_ocr_folder
from ocr_extraction.template_classifier import classify_folder
from ocr_extraction.validator import validate_document

from signals.business_rules_signal import run_business_rules_signal
from signals.inter_dossier_signal import (
    is_test_dossier_id,
    run_inter_dossier_signal,
)
from signals.intra_dossier_signal import run_intra_dossier_signal
from signals.llm_signal import analyser_dossier_llm
from signals.visual_signal_1 import clear_model_cache, run_visual_signal


# ============================================================
# UTILITAIRES
# ============================================================

def _clear_json_outputs(directory: Path) -> None:
    """
    Supprime les anciens JSON générés dans un dossier.
    """
    directory.mkdir(parents=True, exist_ok=True)

    for path in directory.glob("*.json"):
        path.unlink()


# ============================================================
# NORMALISATION + VALIDATION
# ============================================================

def _normalize_and_validate(
    extracted_dir: Path,
    validated_dir: Path,
) -> dict[str, int]:

    _clear_json_outputs(validated_dir)

    counters = {
        "processed": 0,
        "valid": 0,
        "to_review": 0,
        "incomplete": 0,
        "invalid": 0,
        "errors": 0,
    }

    for extracted_path in sorted(extracted_dir.glob("*.json")):

        try:
            extracted = load_json(extracted_path)

            normalized = normalize_document(extracted)

            validated = validate_document(normalized)

            save_json(
                validated,
                validated_dir / extracted_path.name,
            )

            counters["processed"] += 1

            status_key = str(
                validated.get("validation_status", "")
            ).lower()

            if status_key in counters:
                counters[status_key] += 1

            print(
                "[NORMALISÉ + VALIDÉ QUALITÉ] "
                f"{extracted_path.name} : "
                f"{validated.get('validation_status')}"
            )

        except Exception as error:

            counters["errors"] += 1

            print(
                f"[ERREUR] {extracted_path.name} : {error}"
            )

    return counters


# ============================================================
# ÉTAT VIDE D'UN SIGNAL
# ============================================================

def _empty_signal_state(
    status: str,
    error: str | None = None,
) -> dict[str, Any]:

    return {
        "status": status,
        "available": False,
        "error": error,
        "result": None,
    }


# ============================================================
# DÉCISION FINALE (poids calibrés -- régression logistique)
# ============================================================

def _load_calibration(project_root: Path) -> dict[str, Any]:
    """Charge poids_calibration.json depuis la racine du projet."""
    calibration_path = project_root / "poids_calibration.json"
    with calibration_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compute_final_verdict(
    calibration: dict[str, Any],
    business_result: dict[str, Any] | None,
    visual_result: dict[str, Any] | None,
    llm_result: dict[str, Any] | None,
    intra_result: dict[str, Any] | None,
    inter_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Combine les 5 signaux avec les poids calibrés (régression logistique)
    en un score final, puis applique les seuils de décision calibrés.

    Un signal indisponible (None) est compté comme 0.0 -- comportement
    identique à celui utilisé pendant la calibration.
    """

    poids = calibration["poids"]
    seuils = calibration["seuils"]

    scores = {
        "S_business": business_result["score"] if business_result else 0.0,
        "S_intra": intra_result["score"] if intra_result else 0.0,
        "S_inter": inter_result["score"] if inter_result else 0.0,
        "S_llm": llm_result["score"] if llm_result else 0.0,
        "S_visual": visual_result["score"] if visual_result else 0.0,
    }

    score_final = sum(scores[key] * poids[key] for key in scores)

    if score_final < seuils["validation_automatique_sous"]:
        verdict = "VALIDATION_AUTOMATIQUE"
    elif score_final < seuils["escalade_analyste_des"]:
        verdict = "VERIFICATION_COMPLEMENTAIRE"
    else:
        verdict = "ESCALADE_ANALYSTE"

    return {
        "score_final": round(score_final, 4),
        "verdict": verdict,
        "details_scores": scores,
        "poids_utilises": poids,
        "seuils_utilises": seuils,
    }


# ============================================================
# SIGNAL RÈGLES MÉTIER
# ============================================================

def _run_business_rules_branch(
    dossier_json_path: Path,
    signals_root: Path,
) -> dict[str, Any]:

    """
    Calcule les règles métier depuis dossier.json,
    sans Neo4j.
    """

    try:

        result = run_business_rules_signal(
            dossier_json_path=dossier_json_path,
            output_root=signals_root,
        )

        print(
            "[SIGNAL RÈGLES MÉTIER] "
            f"score={result['score']} | "
            f"statut={result['status']} | "
            f"anomalies="
            f"{result['summary']['anomalies_count']} | "
            f"indisponibles="
            f"{result['summary']['checks_unavailable']}"
        )

        print(
            f"[SIGNAL MÉTIER JSON] "
            f"{result['output_path']}"
        )

        return {
            "status": "OK",
            "available": bool(result["available"]),
            "error": None,
            "result": result,
        }

    except Exception as error:

        print(
            "[AVERTISSEMENT] "
            "Signal règles métier indisponible : "
            f"{error}"
        )

        return _empty_signal_state(
            "ERROR",
            str(error),
        )


# ============================================================
# SIGNAL VISUEL (GANomaly)
# ============================================================

def _run_visual_signal_branch(
    manifest_path: Path,
    models_root: Path,
    signals_root: Path,
) -> dict[str, Any]:

    """
    Calcule le signal visuel (GANomaly) directement depuis le manifeste
    de classification.

    Ne dépend PAS de ready_for_signals : ce signal a seulement besoin
    des images déjà classées par template, pas des champs extraits/
    normalisés/validés.
    """

    try:

        result = run_visual_signal(
            classification_manifest_path=manifest_path,
            models_root=models_root,
            output_root=signals_root,
        )

        print(
            "[SIGNAL VISUEL] "
            f"score={result['score']} | "
            f"statut={result['status']} | "
            f"suspects={result['summary']['suspect_count']} | "
            f"ignorés={result['summary']['documents_skipped']}"
        )

        print(
            f"[SIGNAL VISUEL JSON] "
            f"{result['output_path']}"
        )

        return {
            "status": "OK",
            "available": bool(result["available"]),
            "error": None,
            "result": result,
        }

    except Exception as error:

        print(
            "[AVERTISSEMENT] "
            "Signal visuel indisponible : "
            f"{error}"
        )

        return _empty_signal_state(
            "ERROR",
            str(error),
        )


# ============================================================
# SIGNAL LLM GROQ
# ============================================================

def _run_llm_signal_branch(
    dossier_json_path: Path,
    dossier_id: str,
    signals_root: Path,
    ready_for_signals: bool,
) -> dict[str, Any]:

    """
    Calcule le signal LLM à partir du dossier.json final.

    Le LLM reçoit le dossier complet construit après :
        OCR
        -> classification
        -> extraction
        -> normalisation
        -> validation
        -> construction du dossier

    Le LLM analyse principalement les incohérences sémantiques
    difficiles à capturer par des règles déterministes.
    """

    if not ready_for_signals:

        print(
            "[SIGNAL LLM IGNORÉ] "
            "Le dossier n'est pas suffisamment prêt "
            "pour les signaux."
        )

        return _empty_signal_state(
            "SKIPPED_DATA_QUALITY"
        )

    try:

        dossier = load_json(dossier_json_path)

        result = analyser_dossier_llm(dossier)

        if not isinstance(result, dict):
            raise ValueError(
                "Le signal LLM n'a pas retourné un dictionnaire."
            )

        score = result.get("score", 0.0)

        verdict = result.get(
            "verdict",
            "NORMAL",
        )

        anomalies = result.get(
            "anomalies",
            [],
        )

        explication = result.get(
            "explication",
            "",
        )

        llm_output_dir = signals_root / dossier_id

        llm_output_dir.mkdir(
           parents=True,
           exist_ok=True,
        )

        output_path = llm_output_dir / "llm.json"

        llm_result = {
            "dossier_id": dossier_id,
            "score": score,
            "verdict": verdict,
            "anomalies": anomalies,
            "explication": explication,
        }

        save_json(
            llm_result,
            output_path,
        )

        print(
            "[SIGNAL LLM] "
            f"score={score} | "
            f"verdict={verdict} | "
            f"anomalies={len(anomalies)}"
        )

        print(
            f"[SIGNAL LLM JSON] "
            f"{output_path}"
        )

        return {
            "status": "OK",
            "available": True,
            "error": None,
            "result": llm_result,
        }

    except Exception as error:

        print(
            "[AVERTISSEMENT] "
            "Signal LLM indisponible : "
            f"{error}"
        )

        return _empty_signal_state(
            "ERROR",
            str(error),
        )


# ============================================================
# NEO4J + SIGNALS INTRA / INTER
# ============================================================

def _run_graph_and_signals(
    dossier_json_path: Path,
    dossier_id: str,
    signals_root: Path,
    ready_for_signals: bool,
) -> dict[str, Any]:

    """
    Charge le dossier dans Neo4j, puis calcule :

    - signal intra-dossier
    - signal inter-dossiers

    Une panne d'un signal ne supprime pas les autres résultats.
    """

    if not ready_for_signals:

        print(
            "[SIGNAUX NEO4J IGNORÉS] "
            "Le dossier n'est pas prêt pour les signaux."
        )

        return {
            "status": "SKIPPED_DATA_QUALITY",
            "error": None,
            "graph": None,
            "intra": _empty_signal_state(
                "SKIPPED_DATA_QUALITY"
            ),
            "inter": _empty_signal_state(
                "SKIPPED_DATA_QUALITY"
            ),
        }

    client: Neo4jClient | None = None

    try:

        client = Neo4jClient()

        client.verifier_connexion()

        graph_result = load_dossier_json(
            dossier_json_path,
            client=client,
        )

        print(
            "[NEO4J] "
            f"{graph_result['documents']} documents, "
            f"{graph_result['fields']} champs, "
            f"{graph_result['transactions']} transactions "
            "chargés."
        )

        # ====================================================
        # 1. SIGNAL INTRA-DOSSIER
        # ====================================================

        try:

            intra_result = run_intra_dossier_signal(
                dossier_id=dossier_id,
                output_root=signals_root,
                client=client,
            )

            intra_state = {
                "status": "OK",
                "available": bool(
                    intra_result["available"]
                ),
                "error": None,
                "result": intra_result,
            }

            print(
                "[SIGNAL INTRA-DOSSIER] "
                f"score={intra_result['score']} | "
                f"anomalies="
                f"{intra_result['summary']['anomalies_count']} | "
                f"indisponibles="
                f"{intra_result['summary']['checks_unavailable']}"
            )

            print(
                "[SIGNAL INTRA JSON] "
                f"{intra_result['output_path']}"
            )

        except Exception as error:

            intra_state = _empty_signal_state(
                "ERROR",
                str(error),
            )

            print(
                "[AVERTISSEMENT] "
                "Signal intra-dossier indisponible : "
                f"{error}"
            )

        # ====================================================
        # 2. SIGNAL INTER-DOSSIERS
        # ====================================================

        if is_test_dossier_id(dossier_id):

            inter_state = _empty_signal_state(
                "SKIPPED_TEST_DOSSIER"
            )

            print(
                "[SIGNAL INTER-DOSSIERS IGNORÉ] "
                "L'identifiant correspond à un "
                "dossier de test."
            )

        else:

            try:

                inter_result = run_inter_dossier_signal(
                    dossier_id=dossier_id,
                    output_root=signals_root,
                    client=client,
                )

                inter_state = {
                    "status": "OK",
                    "available": bool(
                        inter_result["available"]
                    ),
                    "error": None,
                    "result": inter_result,
                }

                print(
                    "[SIGNAL INTER-DOSSIERS] "
                    f"score={inter_result['score']} | "
                    f"anomalies="
                    f"{inter_result['summary']['anomalies_count']} | "
                    f"critiques="
                    f"{inter_result['summary']['critical_anomalies_count']} | "
                    f"dossiers_liés="
                    f"{inter_result['summary']['linked_dossiers_count']} | "
                    f"indisponibles="
                    f"{inter_result['summary']['checks_unavailable']}"
                )

                print(
                    "[SIGNAL INTER JSON] "
                    f"{inter_result['output_path']}"
                )

            except Exception as error:

                inter_state = _empty_signal_state(
                    "ERROR",
                    str(error),
                )

                print(
                    "[AVERTISSEMENT] "
                    "Signal inter-dossiers indisponible : "
                    f"{error}"
                )

        if (
            intra_state["status"] == "OK"
            and inter_state["status"]
            in {
                "OK",
                "SKIPPED_TEST_DOSSIER",
            }
        ):

            branch_status = "OK"

        else:

            branch_status = "PARTIAL"

        return {
            "status": branch_status,
            "error": None,
            "graph": graph_result,
            "intra": intra_state,
            "inter": inter_state,
        }

    except Exception as error:

        print(
            "[AVERTISSEMENT NEO4J] "
            "Graphe et signaux intra/inter "
            f"indisponibles : {error}"
        )

        print(
            "Les résultats OCR, classification, extraction et "
            "validation restent disponibles. Les signaux règles "
            "métier, visuel et LLM vont quand même être calculés."
        )

        return {
            "status": "UNAVAILABLE_NEO4J",
            "error": str(error),
            "graph": None,
            "intra": _empty_signal_state(
                "UNAVAILABLE_NEO4J",
                str(error),
            ),
            "inter": _empty_signal_state(
                "UNAVAILABLE_NEO4J",
                str(error),
            ),
        }

    finally:

        if client is not None:
            client.fermer()


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def run_pipeline(
    input_dir: str | Path,
    project_root: str | Path = ".",
    templates_dir: str | Path = "templates",
    overwrite_ocr: bool = False,
) -> dict[str, Any]:

    source = Path(input_dir)

    if not source.exists():

        raise FileNotFoundError(
            f"Dossier introuvable : {source}"
        )

    root = Path(project_root)

    dossier_id = source.name

    # Charge une seule fois les poids/seuils calibres (poids_calibration.json
    # a la racine du projet).
    calibration = _load_calibration(root)

    # --------------------------------------------------------
    # Chemins de travail
    # --------------------------------------------------------

    ocr_dir = (
        root
        / "data"
        / "ocr_cache"
        / dossier_id
    )

    manifest_path = (
        root
        / "data"
        / "classification"
        / f"{dossier_id}_manifest.json"
    )

    classified_dir = (
        root
        / "data"
        / "docs_classified"
        / dossier_id
    )

    extracted_dir = (
        root
        / "data"
        / "json_extracted"
        / dossier_id
    )

    validated_dir = (
        root
        / "data"
        / "json_validated"
        / dossier_id
    )

    dossier_json_path = (
        validated_dir
        / "dossier.json"
    )

    signals_root = (
        root
        / "data"
        / "signals"
    )

    visual_models_root = (
        root
        / "signals"
        / "visual_engine"
        / "models"
    )

    # ========================================================
    # 1. OCR
    # ========================================================

    print(
        "\n========== 1. OCR =========="
    )

    ocr_summary = run_ocr_folder(
        source,
        ocr_dir,
        overwrite_ocr,
    )

    # ========================================================
    # 2. CLASSIFICATION
    # ========================================================

    print(
        "\n========== 2. CLASSIFICATION "
        "TYPE + TEMPLATE =========="
    )

    classification = classify_folder(
        source,
        ocr_dir,
        manifest_path,
        classified_dir,
        templates_dir,
    )

    # ========================================================
    # 3. EXTRACTION
    # ========================================================

    print(
        "\n========== 3. EXTRACTION "
        "STRUCTURÉE =========="
    )

    _clear_json_outputs(
        extracted_dir
    )

    extraction = process_manifest(
        manifest_path,
        extracted_dir,
        only_implemented=True,
    )

    # ========================================================
    # 4. NORMALISATION + VALIDATION
    # ========================================================

    print(
        "\n========== 4. NORMALISATION + "
        "VALIDATION DE QUALITÉ =========="
    )

    quality = _normalize_and_validate(
        extracted_dir,
        validated_dir,
    )

    # ========================================================
    # 5. CONSTRUCTION DU DOSSIER
    # ========================================================

    print(
        "\n========== 5. CONSTRUCTION "
        "DU DOSSIER =========="
    )

    dossier = build_dossier_from_folder(
        validated_dir,
        dossier_json_path,
        dossier_id=dossier_id,
    )

    ready_for_signals = bool(
        dossier["data_quality"][
            "ready_for_signals"
        ]
    )

    print(
        f"[DOSSIER] {dossier_json_path}"
    )

    print(
        f"Prêt pour les signaux : "
        f"{ready_for_signals}"
    )

    # ========================================================
    # 6. NEO4J + SIGNAUX INTRA / INTER
    # ========================================================
    #
    # NOUVEAU : deplace en premier parmi les signaux (avant regles
    # metier, visuel et LLM), pour que la connexion et le chargement
    # Neo4j se fassent pendant que la RAM est encore libre, avant que
    # les modeles GANomaly (signal visuel) ou l'appel LLM ne prennent
    # de la memoire/des ressources.

    print(
        "\n========== 6. NEO4J + "
        "SIGNAUX INTRA/INTER =========="
    )

    graph_state = _run_graph_and_signals(
        dossier_json_path=dossier_json_path,
        dossier_id=dossier_id,
        signals_root=signals_root,
        ready_for_signals=ready_for_signals,
    )

    # ========================================================
    # 7. SIGNAL RÈGLES MÉTIER
    # ========================================================

    print(
        "\n========== 7. SIGNAL RÈGLES MÉTIER "
        "(SANS NEO4J) =========="
    )

    business_state = (
        _run_business_rules_branch(
            dossier_json_path=dossier_json_path,
            signals_root=signals_root,
        )
    )

    # ========================================================
    # 8. SIGNAL VISUEL (GANomaly)
    # ========================================================

    print(
        "\n========== 8. SIGNAL VISUEL "
        "(GANomaly) =========="
    )

    visual_state = _run_visual_signal_branch(
        manifest_path=manifest_path,
        models_root=visual_models_root,
        signals_root=signals_root,
    )

    # Libere la RAM occupee par les modeles GANomaly (torch) charges
    # pendant le signal visuel.
    clear_model_cache()
    gc.collect()

    # ========================================================
    # 9. SIGNAL LLM GROQ
    # ========================================================

    print(
        "\n========== 9. SIGNAL LLM "
        "(GROQ) =========="
    )

    llm_state = _run_llm_signal_branch(
        dossier_json_path=dossier_json_path,
        dossier_id=dossier_id,
        signals_root=signals_root,
        ready_for_signals=ready_for_signals,
    )

    # ========================================================
    # RÉCUPÉRATION DES RÉSULTATS
    # ========================================================

    business_result = (
        business_state["result"]
    )

    visual_result = (
        visual_state["result"]
    )

    llm_result = (
        llm_state["result"]
    )

    intra_result = (
        graph_state["intra"]["result"]
    )

    inter_result = (
        graph_state["inter"]["result"]
    )

    # ========================================================
    # RÉSULTAT FINAL
    # ========================================================

    final = {

        "dossier_id": dossier_id,

        "ocr": ocr_summary,

        "classification": (
            classification["summary"]
        ),

        "extraction": (
            extraction["summary"]
        ),

        "data_quality": quality,

        "dossier_quality": (
            dossier["data_quality"]
        ),

        "dossier_json": str(
            dossier_json_path.resolve()
        ),

        "neo4j": {

            "status": (
                graph_state["status"]
            ),

            "error": (
                graph_state["error"]
            ),

            "graph": (
                graph_state["graph"]
            ),
        },

        "signals": {

            "business_rules": (
                business_result
            ),

            "visual": (
                visual_result
            ),

            "llm": (
                llm_result
            ),

            "intra_dossier": (
                intra_result
            ),

            "inter_dossiers": (
                inter_result
            ),
        },

        "signal_status": {

            "business_rules": (
                business_state
            ),

            "visual": (
                visual_state
            ),

            "llm": (
                llm_state
            ),

            "intra_dossier": (
                graph_state["intra"]
            ),

            "inter_dossiers": (
                graph_state["inter"]
            ),
        },
    }

    # ========================================================
    # AFFICHAGE FINAL
    # ========================================================

    print(
        "\n========== RÉSULTAT FINAL =========="
    )

    print(
        f"Dossier : {dossier_id}"
    )

    print(
        f"JSON dossier : "
        f"{dossier_json_path}"
    )

    # --------------------------------------------------------
    # Intra
    # --------------------------------------------------------

    print(
        "Statut intra-dossier : "
        f"{graph_state['intra']['status']}"
    )

    if intra_result is not None:

        print(
            "Score intra-dossier : "
            f"{intra_result['score']}"
        )

        print(
            "Résultat intra : "
            f"{intra_result['output_path']}"
        )

    # --------------------------------------------------------
    # Inter
    # --------------------------------------------------------

    print(
        "Statut inter-dossiers : "
        f"{graph_state['inter']['status']}"
    )

    if inter_result is not None:

        print(
            "Score inter-dossiers : "
            f"{inter_result['score']}"
        )

        print(
            "Résultat inter : "
            f"{inter_result['output_path']}"
        )

    # --------------------------------------------------------
    # Règles métier
    # --------------------------------------------------------

    print(
        "Statut règles métier : "
        f"{business_state['status']}"
    )

    if business_result is not None:

        print(
            "Score règles métier : "
            f"{business_result['score']}"
        )

        print(
            "Résultat règles métier : "
            f"{business_result['output_path']}"
        )

    # --------------------------------------------------------
    # Visuel
    # --------------------------------------------------------

    print(
        "Statut signal visuel : "
        f"{visual_state['status']}"
    )

    if visual_result is not None:

        print(
            "Score visuel : "
            f"{visual_result['score']}"
        )

        print(
            "Résultat visuel : "
            f"{visual_result['output_path']}"
        )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    print(
        "Statut signal LLM : "
        f"{llm_state['status']}"
    )

    if llm_result is not None:

        print(
            "Score LLM : "
            f"{llm_result['score']}"
        )

        print(
            "Verdict LLM : "
            f"{llm_result['verdict']}"
        )

        print(
            "Anomalies LLM : "
            f"{len(llm_result['anomalies'])}"
        )

        if llm_state["status"] == "OK":

            llm_output_path = (
                  signals_root
                  / dossier_id
                  / "llm.json"
            )

            print(
                "Résultat LLM : "
                f"{llm_output_path}"
            )

    # --------------------------------------------------------
    # Décision finale (poids calibrés)
    # --------------------------------------------------------

    verdict_final = compute_final_verdict(
        calibration=calibration,
        business_result=business_result,
        visual_result=visual_result,
        llm_result=llm_result,
        intra_result=intra_result,
        inter_result=inter_result,
    )
    final["decision_finale"] = verdict_final

    print(
        "\nSCORE FINAL : "
        f"{verdict_final['score_final']} -> {verdict_final['verdict']}"
    )

    return final


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description=(
            "Pipeline : OCR -> classification "
            "-> extraction -> normalisation "
            "-> validation -> dossier.json "
            "-> Neo4j + signaux intra/inter "
            "-> règles métier -> signal visuel "
            "(GANomaly) -> LLM Groq."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Exemple : data/raw/D232",
    )

    parser.add_argument(
        "--project-root",
        default=".",
    )

    parser.add_argument(
        "--templates-dir",
        default="templates",
    )

    parser.add_argument(
        "--overwrite-ocr",
        action="store_true",
    )

    args = parser.parse_args()

    run_pipeline(
        args.input,
        args.project_root,
        args.templates_dir,
        args.overwrite_ocr,
    )