from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from testing.test_business_rules_from_corrected_json import (
    load_corrected_documents,
    normalize_text,
    parse_period,
)


DEFAULT_SOURCE_ROOT = Path("data/json_corrected")
DEFAULT_OUTPUT_ROOT = Path(
    "data/testing/business_rules/fraud_cases"
)

CASE_IDS = {
    "THREE_CONSECUTIVE_PERIODS": "D9001",
    "ONE_SALARY_TRANSFER_PER_MONTH": "D9002",
    "SALARY_AMOUNT_MATCH": "D9003",
    "SALARY_LABEL_EMPLOYER": "D9004",
    "TRANSACTION_TOTALS": "D9005",
    "BALANCE_EQUATION": "D9006",
    "BALANCE_CHAINING": "D9007",
}


def doc_type(document: dict[str, Any]) -> str:
    return str(
        document.get("_canonical_doc_type")
        or normalize_text(document.get("doc_type")).replace(" ", "_")
    )


def statements(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = [
        document
        for document in documents
        if doc_type(document) == "RELEVE_BANCAIRE"
    ]

    def key(document: dict[str, Any]) -> tuple[int, int]:
        period = (
            parse_period(document.get("date_solde_cloture"))
            or parse_period(document.get("periode"))
            or (9999, 12)
        )
        return period

    return sorted(result, key=key)


def bulletins(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = [
        document
        for document in documents
        if doc_type(document) == "BULLETIN_SALAIRE"
    ]

    def key(document: dict[str, Any]) -> tuple[int, int]:
        return parse_period(document.get("periode")) or (9999, 12)

    return sorted(result, key=key)


def salary_transactions(
    statement: dict[str, Any],
) -> list[dict[str, Any]]:
    transactions = statement.get("transactions")
    if not isinstance(transactions, list):
        return []

    return [
        transaction
        for transaction in transactions
        if isinstance(transaction, dict)
        and normalize_text(transaction.get("sens")) == "C"
        and "SALAIRE" in normalize_text(transaction.get("libelle"))
    ]


def clean_document(
    document: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(document)
    for key in (
        "_source_path",
        "_doc_id",
        "_canonical_doc_type",
    ):
        result.pop(key, None)
    return result


def update_document_file_name(
    document: dict[str, Any],
    new_stem: str,
) -> None:
    old_file = document.get("_file")
    if isinstance(old_file, str) and old_file:
        suffix = Path(old_file).suffix or ".jpg"
        document["_file"] = f"{new_stem}{suffix}"


def write_case(
    documents: list[dict[str, Any]],
    case_id: str,
    rule_id: str,
    output_root: Path,
) -> None:
    case_dir = output_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    for index, document in enumerate(documents, start=1):
        source_stem = str(
            document.get("_doc_id")
            or Path(
                str(document.get("_source_path") or f"document_{index}")
            ).stem
        )
        source_stem = re.sub(
            r"^D\d+_?",
            "",
            source_stem,
            flags=re.IGNORECASE,
        )
        new_stem = f"{case_id}_{source_stem}"
        clean = clean_document(document)
        update_document_file_name(clean, new_stem)

        output_path = case_dir / f"{new_stem}.json"
        output_path.write_text(
            json.dumps(
                clean,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest = {
        "case_id": case_id,
        "expected_triggered_rule": rule_id,
        "source_is_copy": True,
        "original_data_modified": False,
    }
    (case_dir / "case_manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_case_three_consecutive(
    source: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents = copy.deepcopy(source)
    salary_slips = bulletins(documents)
    if len(salary_slips) != 3:
        raise ValueError(
            "Il faut exactement trois bulletins."
        )

    salary_slips[-1]["periode"] = "12/2099"
    return documents


def build_case_two_salaries(
    source: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents = copy.deepcopy(source)
    bank_statements = statements(documents)
    if not bank_statements:
        raise ValueError("Aucun relevé bancaire.")

    first = bank_statements[0]
    transactions = first.get("transactions")
    if not isinstance(transactions, list):
        raise ValueError("Transactions absentes.")

    salaries = salary_transactions(first)
    if not salaries:
        raise ValueError("Virement salaire introuvable.")

    duplicate = copy.deepcopy(salaries[0])
    duplicate["libelle"] = (
        f"{duplicate.get('libelle', '')} DOUBLON TEST"
    ).strip()
    duplicate["montant"] = 0.0
    transactions.append(duplicate)
    return documents


def build_case_salary_amount(
    source: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents = copy.deepcopy(source)
    salary_slips = bulletins(documents)
    if not salary_slips:
        raise ValueError("Aucun bulletin.")

    current = float(salary_slips[0]["net_a_payer"])
    salary_slips[0]["net_a_payer"] = current + 100.0
    return documents


def build_case_salary_employer(
    source: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents = copy.deepcopy(source)
    bank_statements = statements(documents)
    changed = 0

    for statement in bank_statements:
        salaries = salary_transactions(statement)
        for transaction in salaries:
            transaction["libelle"] = (
                "VIR SALAIRE EMPLOYEUR INCONNU"
            )
            changed += 1

    if changed == 0:
        raise ValueError("Aucun libellé salaire trouvé.")
    return documents


def build_case_transaction_totals(
    source: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents = copy.deepcopy(source)
    bank_statements = statements(documents)
    if not bank_statements:
        raise ValueError("Aucun relevé bancaire.")

    transactions = bank_statements[0].get("transactions")
    if not isinstance(transactions, list):
        raise ValueError("Transactions absentes.")

    for index, transaction in enumerate(transactions):
        if (
            isinstance(transaction, dict)
            and normalize_text(transaction.get("sens")) == "D"
        ):
            transactions.pop(index)
            return documents

    raise ValueError("Aucun débit à retirer.")


def build_case_balance_equation(
    source: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents = copy.deepcopy(source)
    bank_statements = statements(documents)
    if len(bank_statements) != 3:
        raise ValueError(
            "Il faut exactement trois relevés."
        )

    last = bank_statements[-1]
    last["solde_cloture"] = (
        float(last["solde_cloture"]) + 100.0
    )
    return documents


def build_case_balance_chaining(
    source: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents = copy.deepcopy(source)
    bank_statements = statements(documents)
    if len(bank_statements) != 3:
        raise ValueError(
            "Il faut exactement trois relevés."
        )

    for statement in bank_statements[1:]:
        statement["solde_ouverture"] = (
            float(statement["solde_ouverture"]) + 100.0
        )
        statement["solde_cloture"] = (
            float(statement["solde_cloture"]) + 100.0
        )

    return documents


BUILDERS = {
    "THREE_CONSECUTIVE_PERIODS":
        build_case_three_consecutive,
    "ONE_SALARY_TRANSFER_PER_MONTH":
        build_case_two_salaries,
    "SALARY_AMOUNT_MATCH":
        build_case_salary_amount,
    "SALARY_LABEL_EMPLOYER":
        build_case_salary_employer,
    "TRANSACTION_TOTALS":
        build_case_transaction_totals,
    "BALANCE_EQUATION":
        build_case_balance_equation,
    "BALANCE_CHAINING":
        build_case_balance_chaining,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crée sept copies frauduleuses contrôlées "
            "pour tester les règles métier."
        )
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
    )
    parser.add_argument(
        "--source-dossier",
        default="D001",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    source_dossier = args.source_dossier.upper()

    grouped, errors = load_corrected_documents(source_root)
    if errors:
        print(
            f"Attention : {len(errors)} erreur(s) "
            "pendant le chargement."
        )

    if source_dossier not in grouped:
        raise KeyError(
            f"Dossier source introuvable : {source_dossier}"
        )

    source_documents = grouped[source_dossier]

    if output_root.exists():
        for child in output_root.iterdir():
            if (
                child.is_dir()
                and child.name in CASE_IDS.values()
            ):
                for path in sorted(
                    child.rglob("*"),
                    reverse=True,
                ):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                child.rmdir()

    output_root.mkdir(parents=True, exist_ok=True)

    for rule_id, builder in BUILDERS.items():
        case_id = CASE_IDS[rule_id]
        case_documents = builder(source_documents)
        write_case(
            case_documents,
            case_id,
            rule_id,
            output_root,
        )
        print(
            f"[CRÉÉ] {case_id} -> {rule_id}"
        )

    print(
        f"\nCas générés dans : {output_root}"
    )
    print("Les JSON originaux n'ont pas été modifiés.")


if __name__ == "__main__":
    main()
