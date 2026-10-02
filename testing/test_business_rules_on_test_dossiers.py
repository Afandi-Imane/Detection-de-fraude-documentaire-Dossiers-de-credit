from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.neo4j_client import Neo4jClient
from testing.audit_business_rules_values import (
    CHECKERS,
    RULE_IDS,
    build_graph_snapshot,
    execute,
)


OUTPUT_PATH = Path(
    "data/testing/business_rules/test_dossiers_report.json"
)

# Ces règles peuvent entrer plus tard dans le vrai signal.
SCORABLE_RULES = {
    "THREE_CONSECUTIVE_PERIODS",
    "ONE_SALARY_TRANSFER_PER_MONTH",
    "SALARY_AMOUNT_MATCH",
    "SALARY_LABEL_EMPLOYER",
    "TRANSACTION_TOTALS",
    "BALANCE_EQUATION",
    "BALANCE_CHAINING",
    "NON_NEGATIVE_BALANCE",
}

# Cette règle reste observée, mais pas utilisée dans le score pour le moment.
DIAGNOSTIC_RULES = {
    "ADDITIONAL_CREDITS_COUNT",
}


def find_test_dossier_ids(
    client: Neo4jClient,
) -> list[str]:
    rows = execute(
        client,
        """
        MATCH (d:Dossier)
        WHERE
            d.dossier_id = 'D_TEST'
            OR d.dossier_id ENDS WITH '_D_TEST'
            OR d.dossier_id STARTS WITH 'TEST_'
            OR d.dossier_id CONTAINS '_TEST'
        RETURN DISTINCT d.dossier_id AS dossier_id
        ORDER BY dossier_id
        """,
    )
    return [
        str(row["dossier_id"])
        for row in rows
        if row.get("dossier_id")
    ]


def load_selected_dossiers(
    client: Neo4jClient,
    dossier_ids: list[str],
) -> dict[str, Any]:
    field_rows = execute(
        client,
        """
        MATCH (d:Dossier)-[:CONTIENT]->(doc:Document)-[:POSSEDE]->(c:Champ)
        WHERE d.dossier_id IN $dossier_ids
        RETURN
            d.dossier_id AS dossier_id,
            doc.doc_id AS doc_id,
            doc.doc_type AS doc_type,
            c.nom AS field_name,
            c.valeur AS raw_value,
            c.valeur_normalisee AS normalized_value
        ORDER BY dossier_id, doc_id, field_name
        """,
        dossier_ids=dossier_ids,
    )

    transaction_rows = execute(
        client,
        """
        MATCH (d:Dossier)-[:CONTIENT]->(doc:Document)
              -[:CONTIENT_TRANSACTION]->(t:Transaction)
        WHERE d.dossier_id IN $dossier_ids
        RETURN
            d.dossier_id AS dossier_id,
            doc.doc_id AS doc_id,
            doc.doc_type AS doc_type,
            t.transaction_id AS transaction_id,
            t.mois_dossier AS mois_dossier,
            t.date AS date,
            t.periode AS periode,
            t.libelle AS libelle,
            t.montant AS montant,
            t.sens AS sens,
            t.est_salaire AS est_salaire,
            t.employeur_virement AS employeur_virement
        ORDER BY dossier_id, doc_id, transaction_id
        """,
        dossier_ids=dossier_ids,
    )

    return build_graph_snapshot(
        field_rows=field_rows,
        transaction_rows=transaction_rows,
    )


def evaluate_dossier(
    dossier: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    for rule_id in RULE_IDS:
        checker = CHECKERS[rule_id]

        try:
            result = checker(dossier)
        except Exception as error:
            result = {
                "applicable": False,
                "violations": [],
                "details": [],
                "error": str(error),
            }

        if not result.get("applicable"):
            status = "UNAVAILABLE"
        elif result.get("violations"):
            status = "TRIGGERED"
        else:
            status = "NORMAL"

        results[rule_id] = {
            "status": status,
            "used_in_future_score": rule_id in SCORABLE_RULES,
            **result,
        }

    return results


def print_results(
    dossier_results: dict[str, dict[str, Any]],
) -> None:
    print("\n===== TEST DES RÈGLES SUR LES DOSSIERS DE TEST =====")

    for dossier_id, rules in dossier_results.items():
        print(f"\nDossier : {dossier_id}")

        triggered = 0
        unavailable = 0

        for rule_id, result in rules.items():
            status = result["status"]

            if status == "TRIGGERED":
                triggered += 1
            elif status == "UNAVAILABLE":
                unavailable += 1

            usage = (
                "SCORE"
                if result["used_in_future_score"]
                else "DIAGNOSTIC"
            )

            print(
                f"- {rule_id} | "
                f"{status} | "
                f"{usage}"
            )

        print(
            f"Résumé : règles déclenchées={triggered}, "
            f"indisponibles={unavailable}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Teste les règles métier sur des dossiers Neo4j de test. "
            "Aucune donnée n'est modifiée."
        )
    )
    parser.add_argument(
        "--dossiers",
        nargs="*",
        default=None,
        help=(
            "Identifiants précis, par exemple : "
            "--dossiers D147_D_TEST D200_D_TEST"
        ),
    )
    args = parser.parse_args()

    client = Neo4jClient()

    try:
        client.verifier_connexion()

        dossier_ids = (
            [str(value) for value in args.dossiers]
            if args.dossiers
            else find_test_dossier_ids(client)
        )

        if not dossier_ids:
            print(
                "Aucun dossier de test trouvé automatiquement."
            )
            print(
                "Utilisation : python -m "
                "testing.test_business_rules_on_test_dossiers "
                "--dossiers D147_D_TEST"
            )
            return

        dossiers = load_selected_dossiers(
            client=client,
            dossier_ids=dossier_ids,
        )

        missing = sorted(set(dossier_ids) - set(dossiers))
        if missing:
            print(
                "Dossiers absents ou sans champs/transactions : "
                + ", ".join(missing)
            )

        dossier_results = {
            dossier_id: evaluate_dossier(dossier)
            for dossier_id, dossier in sorted(dossiers.items())
        }

        report = {
            "audit_type": "BUSINESS_RULES_TEST_DOSSIERS",
            "mutates_neo4j": False,
            "scores_fraud": False,
            "requested_dossiers": dossier_ids,
            "analyzed_dossiers": list(dossier_results),
            "scorable_rules": sorted(SCORABLE_RULES),
            "diagnostic_rules": sorted(DIAGNOSTIC_RULES),
            "results": dossier_results,
        }

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        OUTPUT_PATH.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print_results(dossier_results)
        print(f"\nRapport : {OUTPUT_PATH}")
        print("Aucune donnée Neo4j n'a été modifiée.")

    finally:
        client.fermer()


if __name__ == "__main__":
    main()
