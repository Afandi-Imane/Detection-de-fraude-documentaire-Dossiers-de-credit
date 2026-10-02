from __future__ import annotations

import argparse
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


def _clear_json_outputs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.json"):
        path.unlink()


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


def _run_business_rules_branch(
    dossier_json_path: Path,
    signals_root: Path,
) -> dict[str, Any]:
    """
    Calcule les règles métier depuis dossier.json, sans Neo4j.
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
            f"[SIGNAL MÉTIER JSON] {result['output_path']}"
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


def _run_graph_and_signals(
    dossier_json_path: Path,
    dossier_id: str,
    signals_root: Path,
    ready_for_signals: bool,
) -> dict[str, Any]:
    """
    Charge le dossier dans Neo4j, puis calcule :
    - le signal intra-dossier ;
    - le signal inter-dossiers.

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

        # 1. Signal intra-dossier
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

        # 2. Signal inter-dossiers
        # Les dossiers de test sont exclus du réseau historique.
        if is_test_dossier_id(dossier_id):
            inter_state = _empty_signal_state(
                "SKIPPED_TEST_DOSSIER"
            )
            print(
                "[SIGNAL INTER-DOSSIERS IGNORÉ] "
                "L'identifiant correspond à un dossier "
                "de test."
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
            in {"OK", "SKIPPED_TEST_DOSSIER"}
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
            "Les résultats OCR, classification, extraction, "
            "validation et règles métier restent disponibles."
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

    ocr_dir = (
        root / "data" / "ocr_cache" / dossier_id
    )
    manifest_path = (
        root
        / "data"
        / "classification"
        / f"{dossier_id}_manifest.json"
    )
    classified_dir = (
        root / "data" / "docs_classified" / dossier_id
    )
    extracted_dir = (
        root / "data" / "json_extracted" / dossier_id
    )
    validated_dir = (
        root / "data" / "json_validated" / dossier_id
    )
    dossier_json_path = (
        validated_dir / "dossier.json"
    )
    signals_root = (
        root / "data" / "signals"
    )

    print("\n========== 1. OCR ==========")
    ocr_summary = run_ocr_folder(
        source,
        ocr_dir,
        overwrite_ocr,
    )

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

    print(
        "\n========== 3. EXTRACTION "
        "STRUCTURÉE =========="
    )
    _clear_json_outputs(extracted_dir)
    extraction = process_manifest(
        manifest_path,
        extracted_dir,
        only_implemented=True,
    )

    print(
        "\n========== 4. NORMALISATION + "
        "VALIDATION DE QUALITÉ =========="
    )
    quality = _normalize_and_validate(
        extracted_dir,
        validated_dir,
    )

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
        dossier["data_quality"]["ready_for_signals"]
    )

    print(f"[DOSSIER] {dossier_json_path}")
    print(
        f"Prêt pour les signaux : {ready_for_signals}"
    )

    print(
        "\n========== 6. SIGNAL RÈGLES MÉTIER "
        "(SANS NEO4J) =========="
    )
    business_state = _run_business_rules_branch(
        dossier_json_path=dossier_json_path,
        signals_root=signals_root,
    )

    print(
        "\n========== 7. NEO4J + "
        "SIGNAUX INTRA/INTER =========="
    )
    graph_state = _run_graph_and_signals(
        dossier_json_path=dossier_json_path,
        dossier_id=dossier_id,
        signals_root=signals_root,
        ready_for_signals=ready_for_signals,
    )

    business_result = business_state["result"]
    intra_result = graph_state["intra"]["result"]
    inter_result = graph_state["inter"]["result"]

    final = {
        "dossier_id": dossier_id,
        "ocr": ocr_summary,
        "classification": classification["summary"],
        "extraction": extraction["summary"],
        "data_quality": quality,
        "dossier_quality": dossier["data_quality"],
        "dossier_json": str(
            dossier_json_path.resolve()
        ),
        "neo4j": {
            "status": graph_state["status"],
            "error": graph_state["error"],
            "graph": graph_state["graph"],
        },
        "signals": {
            "business_rules": business_result,
            "intra_dossier": intra_result,
            "inter_dossiers": inter_result,
        },
        "signal_status": {
            "business_rules": business_state,
            "intra_dossier": graph_state["intra"],
            "inter_dossiers": graph_state["inter"],
        },
    }

    print("\n========== RÉSULTAT FINAL ==========")
    print(f"Dossier : {dossier_id}")
    print(f"JSON dossier : {dossier_json_path}")

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

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline : OCR -> classification -> extraction "
            "-> normalisation -> validation -> dossier.json "
            "-> règles métier -> Neo4j -> "
            "signaux intra/inter."
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
