from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from graph.dossier_graph_loader import load_dossier_json
from graph.neo4j_client import Neo4jClient

from ocr_extraction.dossier_builder_V2 import (
    build_dossier_from_corrected_folder,
)
from ocr_extraction.extractors.common import load_json, save_json

from signals.business_rules_signal import run_business_rules_signal

from signals.inter_dossier_signal import (
    is_test_dossier_id,
    run_inter_dossier_signal,
)

from signals.intra_dossier_signal import (
    run_intra_dossier_signal,
)

from signals.llm_signal import (
    analyser_dossier_llm,
)


# ============================================================
# UTILITAIRE
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
# SIGNAL RÈGLES MÉTIER
# ============================================================

def _run_business_rules_signal(
    dossier_json_path: Path,
    signals_root: Path,
) -> dict[str, Any]:

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
            f"{result['summary']['anomalies_count']}"
        )

        if "output_path" in result:
            print(
                "[BUSINESS JSON] "
                f"{result['output_path']}"
            )

        return {
            "status": "OK",
            "available": bool(
                result.get("available", True)
            ),
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
# SIGNAL LLM
# ============================================================

def _run_llm_signal(
    dossier_json_path: Path,
    dossier_id: str,
    signals_root: Path,
) -> dict[str, Any]:

    try:

        # ----------------------------------------------------
        # Charger le dossier.json
        # ----------------------------------------------------

        dossier = load_json(
            dossier_json_path
        )

        # ----------------------------------------------------
        # Appel LLM
        # ----------------------------------------------------

        result = analyser_dossier_llm(
            dossier
        )

        if not isinstance(result, dict):
            raise ValueError(
                "Le signal LLM n'a pas retourné "
                "un dictionnaire."
            )

        score = result.get(
            "score",
            0.0,
        )

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

        # ----------------------------------------------------
        # Sauvegarde
        # ----------------------------------------------------

        output_dir = (
            signals_root
            / dossier_id
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / "llm.json"
        )

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
            "[LLM JSON] "
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
# NEO4J + SIGNAL INTRA / INTER
# ============================================================

def _run_neo4j_signals(
    dossier_json_path: Path,
    dossier_id: str,
    signals_root: Path,
) -> dict[str, Any]:

    client: Neo4jClient | None = None

    try:

        # ----------------------------------------------------
        # Connexion Neo4j
        # ----------------------------------------------------

        client = Neo4jClient()

        client.verifier_connexion()

        # ----------------------------------------------------
        # Charger dossier.json dans Neo4j
        # ----------------------------------------------------

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
        # SIGNAL INTRA-DOSSIER
        # ====================================================

        try:

            intra_result = (
                run_intra_dossier_signal(
                    dossier_id=dossier_id,
                    output_root=signals_root,
                    client=client,
                )
            )

            intra_state = {
                "status": "OK",
                "available": bool(
                    intra_result.get(
                        "available",
                        True,
                    )
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
                "[INTRA JSON] "
                f"{intra_result['output_path']}"
            )

        except Exception as error:

            print(
                "[AVERTISSEMENT] "
                "Signal intra-dossier indisponible : "
                f"{error}"
            )

            intra_state = _empty_signal_state(
                "ERROR",
                str(error),
            )

        # ====================================================
        # SIGNAL INTER-DOSSIERS
        # ====================================================

        if is_test_dossier_id(
            dossier_id
        ):

            inter_state = _empty_signal_state(
                "SKIPPED_TEST_DOSSIER"
            )

            print(
                "[SIGNAL INTER-DOSSIERS IGNORÉ] "
                "Dossier de test."
            )

        else:

            try:

                inter_result = (
                    run_inter_dossier_signal(
                        dossier_id=dossier_id,
                        output_root=signals_root,
                        client=client,
                    )
                )

                inter_state = {
                    "status": "OK",
                    "available": bool(
                        inter_result.get(
                            "available",
                            True,
                        )
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
                    "[INTER JSON] "
                    f"{inter_result['output_path']}"
                )

            except Exception as error:

                print(
                    "[AVERTISSEMENT] "
                    "Signal inter-dossiers indisponible : "
                    f"{error}"
                )

                inter_state = _empty_signal_state(
                    "ERROR",
                    str(error),
                )

        return {
            "status": "OK",
            "error": None,
            "graph": graph_result,
            "intra": intra_state,
            "inter": inter_state,
        }

    except Exception as error:

        print(
            "[AVERTISSEMENT NEO4J] "
            "Neo4j / signaux intra-inter "
            f"indisponibles : {error}"
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
# PIPELINE V2 PRINCIPAL
# ============================================================

def run_pipeline_v2(
    input_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:

    # ========================================================
    # 1. VÉRIFIER L'ENTRÉE
    # ========================================================

    source = Path(
        input_dir
    )

    if not source.exists():

        raise FileNotFoundError(
            f"Dossier json_corrected introuvable : "
            f"{source}"
        )

    if not source.is_dir():

        raise ValueError(
            "L'entrée doit être un dossier contenant "
            f"les JSON corrected : {source}"
        )

    root = Path(
        project_root
    )

    dossier_id = source.name

    print()
    print("=" * 60)
    print(
        f" PIPELINE V2 - {dossier_id}"
    )
    print("=" * 60)

    # ========================================================
    # 2. CHEMINS
    # ========================================================

    dossiers_v2_root = (
        root
        / "data"
        / "dossiers_V2"
    )

    dossier_output_dir = (
        dossiers_v2_root
        / dossier_id
    )

    dossier_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dossier_json_path = (
        dossier_output_dir
        / "dossier.json"
    )

    signals_root = (
        root
        / "data"
        / "signals"
    )

    # ========================================================
    # 3. CONSTRUIRE DOSSIER.JSON
    # ========================================================

    print()
    print(
        "========== 1. DOSSIER BUILDER V2 =========="
    )

    print(
        f"[SOURCE] {source}"
    )

    # IMPORTANT :
    # On passe le DOSSIER json_corrected,
    # pas un fichier JSON individuel.

    dossier = (
        build_dossier_from_corrected_folder(
            corrected_dir=source,
            output_path=dossier_json_path,
            dossier_id=dossier_id,
        )
    )

    print()
    print(
        "[DOSSIER V2] "
        f"{dossier_json_path}"
    )

    print(
        "[DOSSIER V2] "
        f"documents="
        f"{dossier['data_quality']['document_count']}"
    )

    print(
        "[DOSSIER V2] "
        f"ready_for_signals="
        f"{dossier['data_quality']['ready_for_signals']}"
    )

    # ========================================================
    # 4. SIGNAL RÈGLES MÉTIER
    # ========================================================

    print()
    print(
        "========== 2. SIGNAL RÈGLES MÉTIER =========="
    )

    business_state = (
        _run_business_rules_signal(
            dossier_json_path=dossier_json_path,
            signals_root=signals_root,
        )
    )

    # ========================================================
    # 5. SIGNAL LLM
    # ========================================================

    print()
    print(
        "========== 3. SIGNAL LLM =========="
    )

    llm_state = (
        _run_llm_signal(
            dossier_json_path=dossier_json_path,
            dossier_id=dossier_id,
            signals_root=signals_root,
        )
    )

    # ========================================================
    # 6. NEO4J + INTRA + INTER
    # ========================================================

    print()
    print(
        "========== 4. NEO4J + INTRA + INTER =========="
    )

    graph_state = (
        _run_neo4j_signals(
            dossier_json_path=dossier_json_path,
            dossier_id=dossier_id,
            signals_root=signals_root,
        )
    )

    # ========================================================
    # 7. RÉSULTATS
    # ========================================================

    business_result = (
        business_state["result"]
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
    # 8. RESULTAT FINAL
    # ========================================================

    final = {

        "dossier_id": dossier_id,

        "source": {
            "json_corrected_dir": str(
                source.resolve()
            ),
        },

        "dossier": {
            "path": str(
                dossier_json_path.resolve()
            ),
            "quality": dossier[
                "data_quality"
            ],
        },

        "neo4j": {
            "status": graph_state[
                "status"
            ],
            "error": graph_state[
                "error"
            ],
            "graph": graph_state[
                "graph"
            ],
        },

        "signals": {

            "business_rules": (
                business_result
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

            "llm": (
                llm_state
            ),

            "intra_dossier": (
                graph_state[
                    "intra"
                ]
            ),

            "inter_dossiers": (
                graph_state[
                    "inter"
                ]
            ),
        },
    }

    # ========================================================
    # 9. SAUVEGARDE DU RÉSULTAT GLOBAL
    # ========================================================

    final_output_dir = (
        signals_root
        / dossier_id
    )

    final_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_output_path = (
        final_output_dir
        / "signals_v2.json"
    )

    save_json(
        final,
        final_output_path,
    )

    # ========================================================
    # 10. AFFICHAGE FINAL
    # ========================================================

    print()
    print("=" * 60)
    print(
        " RÉSULTAT FINAL"
    )
    print("=" * 60)

    print(
        f"Dossier : {dossier_id}"
    )

    print(
        f"dossier.json : "
        f"{dossier_json_path}"
    )

    print(
        f"Résultat global : "
        f"{final_output_path}"
    )

    if business_result is not None:

        print(
            f"Score métier : "
            f"{business_result.get('score')}"
        )

    if llm_result is not None:

        print(
            f"Score LLM : "
            f"{llm_result.get('score')}"
        )

    if intra_result is not None:

        print(
            f"Score intra : "
            f"{intra_result.get('score')}"
        )

    if inter_result is not None:

        print(
            f"Score inter : "
            f"{inter_result.get('score')}"
        )

    print()
    print(
        "PIPELINE V2 TERMINÉ."
    )

    return final


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Pipeline V2 : "
            "json_corrected -> dossier.json "
            "-> signaux métier + LLM + Neo4j "
            "-> intra/inter."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Dossier json_corrected. "
            "Exemple : data/json_corrected/D005"
        ),
    )

    parser.add_argument(
        "--project-root",
        default=".",
        help="Racine du projet.",
    )

    args = parser.parse_args()

    run_pipeline_v2(
        input_dir=args.input,
        project_root=args.project_root,
    )