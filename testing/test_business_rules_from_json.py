from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from testing.audit_business_rules_values import (
    CHECKERS,
    RULE_IDS,
    normalize_doc_type,
    normalize_text,
)


DEFAULT_VALIDATED_ROOT = Path("data/json_validated")
DEFAULT_OUTPUT = Path(
    "data/testing/business_rules/json_dossiers_report.json"
)

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

DIAGNOSTIC_RULES = {
    "ADDITIONAL_CREDITS_COUNT",
}


def effective_field_value(field: dict[str, Any]) -> Any:
    value = field.get("normalized_value")
    if value is None or value == "":
        value = field.get("raw_value")
    return value


def extract_transactions(
    document: dict[str, Any],
) -> list[dict[str, Any]]:
    transactions: list[dict[str, Any]] = []

    transaction_field = None
    for field in document.get("fields", []):
        if normalize_text(field.get("field_name")) == "TRANSACTIONS":
            transaction_field = field
            break

    if transaction_field is None:
        return transactions

    values = effective_field_value(transaction_field)
    if not isinstance(values, list):
        return transactions

    for index, transaction in enumerate(values, start=1):
        if not isinstance(transaction, dict):
            continue

        libelle = transaction.get("libelle")
        normalized_label = normalize_text(libelle)

        transactions.append({
            "transaction_id": (
                f"{document.get('doc_id')}__JSON_TX{index:03d}"
            ),
            "doc_id": document.get("doc_id"),
            "date": transaction.get("date"),
            "periode": None,
            "libelle": libelle,
            "montant": transaction.get("montant"),
            "sens": transaction.get("sens"),
            "est_salaire": "SALAIRE" in normalized_label,
            "employeur_virement": None,
            "ocr_confidence": transaction.get("ocr_confidence"),
        })

    return transactions


def adapt_dossier_json(
    dossier_json: dict[str, Any],
) -> dict[str, Any]:
    documents: dict[str, Any] = {}

    for document in dossier_json.get("documents", []):
        doc_id = str(document.get("doc_id") or "")
        if not doc_id:
            continue

        adapted_fields = []
        for field in document.get("fields", []):
            field_name = normalize_text(field.get("field_name"))
            if not field_name:
                continue

            adapted_fields.append({
                "field_name": field_name,
                "raw_value": field.get("raw_value"),
                "normalized_value": field.get(
                    "normalized_value"
                ),
            })

        documents[doc_id] = {
            "doc_id": doc_id,
            "doc_type": normalize_doc_type(
                document.get("doc_type")
            ),
            "fields": adapted_fields,
            "transactions": extract_transactions(document),
        }

    return {"documents": documents}


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


def resolve_paths(
    root: Path,
    dossier_ids: list[str] | None,
    all_dossiers: bool,
) -> list[Path]:
    if dossier_ids:
        return [
            root / dossier_id / "dossier.json"
            for dossier_id in dossier_ids
        ]

    if all_dossiers:
        return sorted(root.glob("*/dossier.json"))

    raise ValueError(
        "Indiquer --dossiers ou --all."
    )


def print_results(
    dossier_results: dict[str, dict[str, Any]],
) -> None:
    print(
        "\n===== TEST RÈGLES MÉTIER DEPUIS dossier.json ====="
    )

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
                f"- {rule_id} | {status} | {usage}"
            )

        print(
            f"Résumé : déclenchées={triggered}, "
            f"indisponibles={unavailable}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Teste les règles métier directement depuis "
            "data/json_validated, sans Neo4j."
        )
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_VALIDATED_ROOT),
        help="Racine contenant les dossiers validés.",
    )
    parser.add_argument(
        "--dossiers",
        nargs="*",
        default=None,
        help=(
            "Exemple : --dossiers D145_D_TEST D147_D_TEST"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Teste tous les dossier.json présents sous la racine."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )
    args = parser.parse_args()

    root = Path(args.root)
    paths = resolve_paths(
        root=root,
        dossier_ids=args.dossiers,
        all_dossiers=args.all,
    )

    dossier_results: dict[str, dict[str, Any]] = {}
    missing_paths: list[str] = []

    for path in paths:
        if not path.exists():
            missing_paths.append(str(path))
            continue

        dossier_json = json.loads(
            path.read_text(encoding="utf-8")
        )
        dossier_id = str(
            dossier_json.get("dossier_id")
            or path.parent.name
        )

        adapted = adapt_dossier_json(dossier_json)
        dossier_results[dossier_id] = evaluate_dossier(
            adapted
        )

    if missing_paths:
        print("Fichiers absents :")
        for path in missing_paths:
            print(f"- {path}")

    report = {
        "audit_type": (
            "BUSINESS_RULES_FROM_VALIDATED_JSON"
        ),
        "uses_neo4j": False,
        "mutates_data": False,
        "scores_fraud": False,
        "analyzed_dossiers": list(dossier_results),
        "scorable_rules": sorted(SCORABLE_RULES),
        "diagnostic_rules": sorted(DIAGNOSTIC_RULES),
        "results": dossier_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_results(dossier_results)
    print(f"\nRapport : {output_path}")
    print("Neo4j n'a pas été utilisé.")
    print("Aucune donnée n'a été modifiée.")


if __name__ == "__main__":
    main()
