from __future__ import annotations

import csv
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.neo4j_client import Neo4jClient


OUTPUT_DIR = Path("data/testing/business_rules")
REPORT_PATH = OUTPUT_DIR / "audit_rule_outcomes.json"
SUMMARY_CSV_PATH = OUTPUT_DIR / "audit_rule_outcomes_summary.csv"

TEST_FILTER = """
d.dossier_id <> 'D_TEST'
AND NOT d.dossier_id ENDS WITH '_D_TEST'
AND NOT d.dossier_id STARTS WITH 'TEST_'
"""

RULE_IDS = [
    "THREE_CONSECUTIVE_PERIODS",
    "ONE_SALARY_TRANSFER_PER_MONTH",
    "SALARY_AMOUNT_MATCH",
    "SALARY_LABEL_EMPLOYER",
    "TRANSACTION_TOTALS",
    "BALANCE_EQUATION",
    "BALANCE_CHAINING",
    "NON_NEGATIVE_BALANCE",
    "ADDITIONAL_CREDITS_COUNT",
]

FIELD_ALIASES = {
    "PERIODE": ["PERIODE", "MOIS", "MOIS_DOCUMENT"],
    "DATE_SOLDE_CLOTURE": ["DATE_SOLDE_CLOTURE"],
    "DATE_SOLDE_OUVERTURE": ["DATE_SOLDE_OUVERTURE"],
    "NET_SALARY": [
        "NET_A_PAYER",
        "SALAIRE_NET",
        "NET_PAYE",
        "NET_IMPOSABLE",
    ],
    "EMPLOYEUR": ["EMPLOYEUR"],
    "TOTAL_CREDITS": ["TOTAL_CREDITS", "TOTAL_CREDIT"],
    "TOTAL_DEBITS": ["TOTAL_DEBITS", "TOTAL_DEBIT"],
    "SOLDE_OUVERTURE": ["SOLDE_OUVERTURE", "SOLDE_DEPART"],
    "SOLDE_CLOTURE": ["SOLDE_CLOTURE", "SOLDE_FINAL"],
}

MONTHS_FR = {
    "JANVIER": 1,
    "JANV": 1,
    "FEVRIER": 2,
    "FEVR": 2,
    "MARS": 3,
    "AVRIL": 4,
    "AVR": 4,
    "MAI": 5,
    "JUIN": 6,
    "JUILLET": 7,
    "JUIL": 7,
    "AOUT": 8,
    "SEPTEMBRE": 9,
    "SEPT": 9,
    "OCTOBRE": 10,
    "OCT": 10,
    "NOVEMBRE": 11,
    "NOV": 11,
    "DECEMBRE": 12,
    "DEC": 12,
}

GENERIC_EMPLOYER_TOKENS = {
    "SA",
    "SARL",
    "SAS",
    "MAROC",
    "MAROCAINE",
    "GROUPE",
    "STE",
    "SOCIETE",
}


def execute(client: Neo4jClient, query: str, **parameters: Any) -> list[dict[str, Any]]:
    records, _, _ = client.driver.execute_query(
        query,
        database_=client.database,
        **parameters,
    )
    return [record.data() for record in records]


def strip_accents(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or ""))
    return "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )


def normalize_text(value: Any) -> str:
    text = strip_accents(value).upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return " ".join(text.split())


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return normalize_text(value) in {"TRUE", "1", "OUI", "YES"}


def parse_amount(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = str(value).strip()
    if not text:
        return None

    text = (
        text.replace("\u00a0", "")
        .replace("\u202f", "")
        .replace(" ", "")
    )
    text = re.sub(r"[^0-9,.\-]", "", text)

    if not text or text in {"-", ".", ","}:
        return None

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        decimal_part = text.split(",")[-1]
        if len(decimal_part) in {1, 2}:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")

    try:
        number = float(text)
    except ValueError:
        return None

    return number if math.isfinite(number) else None


def parse_period(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None

    text = normalize_text(value)
    if not text:
        return None

    patterns = [
        r"\b(20\d{2})[\s\-/](0?[1-9]|1[0-2])\b",
        r"\b(0?[1-9]|1[0-2])[\s\-/](20\d{2})\b",
    ]

    match = re.search(patterns[0], text)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.search(patterns[1], text)
    if match:
        return int(match.group(2)), int(match.group(1))

    # Accepter une date complète ISO ou DD/MM/YYYY.
    date_patterns = [
        ("%Y-%m-%d", r"\b20\d{2}-\d{2}-\d{2}\b"),
        ("%d/%m/%Y", r"\b\d{1,2}/\d{1,2}/20\d{2}\b"),
    ]

    raw_text = str(value).strip()
    for date_format, regex in date_patterns:
        date_match = re.search(regex, raw_text)
        if not date_match:
            continue
        try:
            parsed = datetime.strptime(date_match.group(0), date_format)
            return parsed.year, parsed.month
        except ValueError:
            pass

    year_match = re.search(r"\b(20\d{2})\b", text)
    if year_match:
        year = int(year_match.group(1))
        for month_name, month_number in MONTHS_FR.items():
            if re.search(rf"\b{month_name}\b", text):
                return year, month_number

    return None


def period_label(period: tuple[int, int] | None) -> str | None:
    if period is None:
        return None
    year, month = period
    return f"{year:04d}-{month:02d}"


def consecutive(periods: list[tuple[int, int]]) -> bool:
    if len(periods) != 3 or len(set(periods)) != 3:
        return False

    indexes = sorted(year * 12 + month for year, month in periods)
    return indexes[1] - indexes[0] == 1 and indexes[2] - indexes[1] == 1


def get_field(
    document: dict[str, Any],
    alias_group: str,
) -> dict[str, Any] | None:
    aliases = {
        normalize_text(alias)
        for alias in FIELD_ALIASES[alias_group]
    }
    for field in document.get("fields", []):
        if field["field_name"] in aliases:
            value = field.get("normalized_value")
            if value is None or str(value).strip() == "":
                value = field.get("raw_value")
            if value is not None and str(value).strip() != "":
                return {**field, "effective_value": value}
    return None


def normalize_doc_type(value: Any) -> str:
    """Normalise les types de documents sans perdre les séparateurs logiques."""
    return normalize_text(value).replace(" ", "_")


def normalize_sens(value: Any) -> str:
    text = normalize_text(value)
    if text in {"D", "DEBIT", "DEBITS"}:
        return "D"
    if text in {"C", "CREDIT", "CREDITS"}:
        return "C"
    return ""


def employer_similarity(employer: Any, transaction: dict[str, Any]) -> float:
    employer_text = normalize_text(employer)
    tx_employer = normalize_text(transaction.get("employeur_virement"))
    label = normalize_text(transaction.get("libelle"))

    if not employer_text:
        return 0.0

    employer_compact = employer_text.replace(" ", "")
    label_compact = label.replace(" ", "")
    tx_employer_compact = tx_employer.replace(" ", "")

    if (
        employer_text in label
        or employer_text == tx_employer
        or employer_compact in label_compact
        or employer_compact == tx_employer_compact
    ):
        return 1.0

    employer_tokens = {
        token
        for token in employer_text.split()
        if token not in GENERIC_EMPLOYER_TOKENS
    }

    candidate_text = " ".join(
        part for part in [tx_employer, label] if part
    )
    candidate_tokens = set(candidate_text.split())

    if employer_tokens:
        containment = len(employer_tokens & candidate_tokens) / len(employer_tokens)
    else:
        containment = 0.0

    sequence = SequenceMatcher(
        None,
        employer_text,
        tx_employer or label,
    ).ratio()

    return max(containment, sequence)


def build_graph_snapshot(
    field_rows: list[dict[str, Any]],
    transaction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    dossiers: dict[str, Any] = {}

    for row in field_rows:
        dossier = dossiers.setdefault(
            row["dossier_id"],
            {"documents": {}},
        )
        document = dossier["documents"].setdefault(
            row["doc_id"],
            {
                "doc_id": row["doc_id"],
                "doc_type": normalize_doc_type(row["doc_type"]),
                "fields": [],
                "transactions": [],
            },
        )
        document["fields"].append({
            "field_name": normalize_text(row["field_name"]),
            "raw_value": row.get("raw_value"),
            "normalized_value": row.get("normalized_value"),
        })

    for row in transaction_rows:
        dossier = dossiers.setdefault(
            row["dossier_id"],
            {"documents": {}},
        )
        document = dossier["documents"].setdefault(
            row["doc_id"],
            {
                "doc_id": row["doc_id"],
                "doc_type": normalize_doc_type(row["doc_type"]),
                "fields": [],
                "transactions": [],
            },
        )
        document["transactions"].append(row)

    return dossiers


def document_period(document: dict[str, Any]) -> tuple[int, int] | None:
    """Resolve a document period from fields, then from transaction metadata."""
    for alias_group in [
        "PERIODE",
        "DATE_SOLDE_CLOTURE",
        "DATE_SOLDE_OUVERTURE",
    ]:
        field = get_field(document, alias_group)
        period = parse_period(
            field["effective_value"]
            if field
            else None
        )
        if period is not None:
            return period

    for transaction in document.get("transactions", []):
        period = parse_period(transaction.get("periode"))
        if period is not None:
            return period

    return None


def empty_rule_result() -> dict[str, Any]:
    return {
        "applicable": False,
        "violations": [],
        "details": [],
    }


def check_three_consecutive_periods(dossier: dict[str, Any]) -> dict[str, Any]:
    result = empty_rule_result()
    families = {
        "BULLETIN_SALAIRE": [],
        "RELEVE_BANCAIRE": [],
    }

    for document in dossier["documents"].values():
        if document["doc_type"] not in families:
            continue

        period = document_period(document)

        families[document["doc_type"]].append({
            "doc_id": document["doc_id"],
            "period": period,
            "period_label": period_label(period),
        })

    result["applicable"] = all(families.values())

    if not result["applicable"]:
        return result

    for family, elements in families.items():
        periods = [
            element["period"]
            for element in elements
            if element["period"] is not None
        ]

        if len(elements) != 3 or len(periods) != 3 or not consecutive(periods):
            result["violations"].append({
                "family": family,
                "documents": elements,
                "reason": "Les trois périodes ne sont pas toutes présentes et consécutives.",
            })

    result["details"] = families
    return result


def check_salary_transfer_count(dossier: dict[str, Any]) -> dict[str, Any]:
    result = empty_rule_result()

    statements = [
        document
        for document in dossier["documents"].values()
        if document["doc_type"] == "RELEVE_BANCAIRE"
    ]

    if not statements:
        return result

    result["applicable"] = True

    for document in statements:
        salary_transactions = [
            transaction
            for transaction in document["transactions"]
            if parse_bool(transaction.get("est_salaire"))
        ]

        detail = {
            "doc_id": document["doc_id"],
            "salary_count": len(salary_transactions),
            "salary_transactions": [
                transaction.get("transaction_id")
                for transaction in salary_transactions
            ],
        }
        result["details"].append(detail)

        if len(salary_transactions) != 1:
            result["violations"].append({
                **detail,
                "reason": "Le relevé doit contenir exactement un virement de salaire.",
            })

    return result


def build_period_maps(dossier: dict[str, Any]) -> tuple[dict[Any, Any], dict[Any, Any]]:
    bulletins: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    statements: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for document in dossier["documents"].values():
        period = document_period(document)

        if period is None:
            continue

        if document["doc_type"] == "BULLETIN_SALAIRE":
            bulletins[period].append(document)
        elif document["doc_type"] == "RELEVE_BANCAIRE":
            statements[period].append(document)

    return bulletins, statements


def check_salary_amount_match(dossier: dict[str, Any]) -> dict[str, Any]:
    result = empty_rule_result()
    bulletins, statements = build_period_maps(dossier)

    common_periods = sorted(set(bulletins) & set(statements))
    if not common_periods:
        return result

    result["applicable"] = True

    for period in common_periods:
        bulletin_docs = bulletins[period]
        statement_docs = statements[period]

        net_values = []
        for document in bulletin_docs:
            field = get_field(document, "NET_SALARY")
            amount = parse_amount(field["effective_value"] if field else None)
            if amount is not None:
                net_values.append({
                    "doc_id": document["doc_id"],
                    "amount": amount,
                })

        salary_values = []
        for document in statement_docs:
            for transaction in document["transactions"]:
                if not parse_bool(transaction.get("est_salaire")):
                    continue
                amount = parse_amount(transaction.get("montant"))
                if amount is not None:
                    salary_values.append({
                        "doc_id": document["doc_id"],
                        "transaction_id": transaction.get("transaction_id"),
                        "amount": amount,
                    })

        detail = {
            "period": period_label(period),
            "bulletin_net_values": net_values,
            "salary_transactions": salary_values,
        }
        result["details"].append(detail)

        if len(net_values) != 1 or len(salary_values) != 1:
            result["violations"].append({
                **detail,
                "reason": "Impossible d'obtenir exactement un net bulletin et un virement salaire.",
            })
            continue

        difference = abs(net_values[0]["amount"] - salary_values[0]["amount"])
        if difference > 0.01:
            result["violations"].append({
                **detail,
                "difference": round(difference, 2),
                "reason": "Le virement salaire diffère du net du bulletin.",
            })

    return result


def check_salary_label_employer(dossier: dict[str, Any]) -> dict[str, Any]:
    result = empty_rule_result()

    employer_values = []
    for document in dossier["documents"].values():
        field = get_field(document, "EMPLOYEUR")
        if field:
            employer_values.append(field["effective_value"])

    employer_values_normalized = [
        normalize_text(value)
        for value in employer_values
        if normalize_text(value)
    ]

    if not employer_values_normalized:
        return result

    employer = max(
        set(employer_values_normalized),
        key=employer_values_normalized.count,
    )

    salary_transactions = []
    for document in dossier["documents"].values():
        for transaction in document["transactions"]:
            if parse_bool(transaction.get("est_salaire")):
                salary_transactions.append(transaction)

    if not salary_transactions:
        return result

    result["applicable"] = True

    for transaction in salary_transactions:
        similarity = employer_similarity(employer, transaction)
        detail = {
            "employer": employer,
            "transaction_id": transaction.get("transaction_id"),
            "libelle": transaction.get("libelle"),
            "employeur_virement": transaction.get("employeur_virement"),
            "similarity": round(similarity, 3),
        }
        result["details"].append(detail)

        # Seuil volontairement souple pour l'audit.
        if similarity < 0.70:
            result["violations"].append({
                **detail,
                "reason": "Le libellé du salaire ne permet pas de relier clairement le virement à l'employeur.",
            })

    return result


def statement_numeric_context(document: dict[str, Any]) -> dict[str, Any]:
    context = {}
    for logical_name in [
        "TOTAL_CREDITS",
        "TOTAL_DEBITS",
        "SOLDE_OUVERTURE",
        "SOLDE_CLOTURE",
    ]:
        field = get_field(document, logical_name)
        context[logical_name] = {
            "field": field,
            "amount": parse_amount(
                field["effective_value"]
                if field
                else None
            ),
        }

    debit_sum = 0.0
    credit_sum = 0.0
    usable_transactions = 0

    for transaction in document["transactions"]:
        amount = parse_amount(transaction.get("montant"))
        sens = normalize_sens(transaction.get("sens"))
        if amount is None or not sens:
            continue

        usable_transactions += 1
        if sens == "D":
            debit_sum += amount
        elif sens == "C":
            credit_sum += amount

    context["TRANSACTION_DEBITS"] = round(debit_sum, 2)
    context["TRANSACTION_CREDITS"] = round(credit_sum, 2)
    context["USABLE_TRANSACTIONS"] = usable_transactions
    return context


def check_transaction_totals(dossier: dict[str, Any]) -> dict[str, Any]:
    result = empty_rule_result()

    for document in dossier["documents"].values():
        if document["doc_type"] != "RELEVE_BANCAIRE":
            continue

        context = statement_numeric_context(document)
        declared_debits = context["TOTAL_DEBITS"]["amount"]
        declared_credits = context["TOTAL_CREDITS"]["amount"]

        if (
            declared_debits is None
            or declared_credits is None
            or context["USABLE_TRANSACTIONS"] == 0
        ):
            continue

        result["applicable"] = True

        debit_difference = abs(
            declared_debits - context["TRANSACTION_DEBITS"]
        )
        credit_difference = abs(
            declared_credits - context["TRANSACTION_CREDITS"]
        )

        detail = {
            "doc_id": document["doc_id"],
            "declared_debits": declared_debits,
            "calculated_debits": context["TRANSACTION_DEBITS"],
            "declared_credits": declared_credits,
            "calculated_credits": context["TRANSACTION_CREDITS"],
            "debit_difference": round(debit_difference, 2),
            "credit_difference": round(credit_difference, 2),
        }
        result["details"].append(detail)

        if debit_difference > 0.01 or credit_difference > 0.01:
            result["violations"].append({
                **detail,
                "reason": "Les totaux déclarés ne correspondent pas à la somme des transactions.",
            })

    return result


def check_balance_equation(dossier: dict[str, Any]) -> dict[str, Any]:
    result = empty_rule_result()

    for document in dossier["documents"].values():
        if document["doc_type"] != "RELEVE_BANCAIRE":
            continue

        context = statement_numeric_context(document)
        required = [
            context["SOLDE_OUVERTURE"]["amount"],
            context["TOTAL_CREDITS"]["amount"],
            context["TOTAL_DEBITS"]["amount"],
            context["SOLDE_CLOTURE"]["amount"],
        ]

        if any(value is None for value in required):
            continue

        result["applicable"] = True

        opening, credits, debits, closing = required
        calculated = opening + credits - debits
        difference = abs(calculated - closing)

        detail = {
            "doc_id": document["doc_id"],
            "opening": opening,
            "credits": credits,
            "debits": debits,
            "declared_closing": closing,
            "calculated_closing": round(calculated, 2),
            "difference": round(difference, 2),
        }
        result["details"].append(detail)

        if difference > 0.01:
            result["violations"].append({
                **detail,
                "reason": "L'équation du solde n'est pas respectée.",
            })

    return result


def check_balance_chaining(dossier: dict[str, Any]) -> dict[str, Any]:
    result = empty_rule_result()
    statements = []

    for document in dossier["documents"].values():
        if document["doc_type"] != "RELEVE_BANCAIRE":
            continue

        period = document_period(document)
        context = statement_numeric_context(document)
        opening = context["SOLDE_OUVERTURE"]["amount"]
        closing = context["SOLDE_CLOTURE"]["amount"]

        if period is None or opening is None or closing is None:
            continue

        statements.append({
            "doc_id": document["doc_id"],
            "period": period,
            "opening": opening,
            "closing": closing,
        })

    if len(statements) < 2:
        return result

    result["applicable"] = True
    statements.sort(key=lambda item: item["period"])

    for previous, current in zip(statements, statements[1:]):
        difference = abs(previous["closing"] - current["opening"])
        detail = {
            "previous_doc": previous["doc_id"],
            "previous_period": period_label(previous["period"]),
            "previous_closing": previous["closing"],
            "current_doc": current["doc_id"],
            "current_period": period_label(current["period"]),
            "current_opening": current["opening"],
            "difference": round(difference, 2),
        }
        result["details"].append(detail)

        if difference > 0.01:
            result["violations"].append({
                **detail,
                "reason": "Le solde de clôture ne correspond pas à l'ouverture du relevé suivant.",
            })

    return result


def check_non_negative_balance(dossier: dict[str, Any]) -> dict[str, Any]:
    result = empty_rule_result()

    for document in dossier["documents"].values():
        if document["doc_type"] != "RELEVE_BANCAIRE":
            continue

        context = statement_numeric_context(document)
        balances = {
            "opening": context["SOLDE_OUVERTURE"]["amount"],
            "closing": context["SOLDE_CLOTURE"]["amount"],
        }

        available = {
            name: value
            for name, value in balances.items()
            if value is not None
        }

        if not available:
            continue

        result["applicable"] = True
        detail = {
            "doc_id": document["doc_id"],
            **available,
        }
        result["details"].append(detail)

        negative = {
            name: value
            for name, value in available.items()
            if value < -0.01
        }
        if negative:
            result["violations"].append({
                **detail,
                "negative_values": negative,
                "reason": "Un solde négatif a été trouvé.",
            })

    return result


def check_additional_credits_count(dossier: dict[str, Any]) -> dict[str, Any]:
    result = empty_rule_result()

    statements = [
        document
        for document in dossier["documents"].values()
        if document["doc_type"] == "RELEVE_BANCAIRE"
    ]

    if not statements:
        return result

    result["applicable"] = True

    for document in statements:
        credits = [
            transaction
            for transaction in document["transactions"]
            if normalize_sens(transaction.get("sens")) == "C"
        ]
        salary_credits = [
            transaction
            for transaction in credits
            if parse_bool(transaction.get("est_salaire"))
        ]
        additional = [
            transaction
            for transaction in credits
            if not parse_bool(transaction.get("est_salaire"))
        ]

        detail = {
            "doc_id": document["doc_id"],
            "credits_count": len(credits),
            "salary_credits_count": len(salary_credits),
            "additional_credits_count": len(additional),
        }
        result["details"].append(detail)

        if len(salary_credits) != 1 or not 1 <= len(additional) <= 3:
            result["violations"].append({
                **detail,
                "reason": "Le relevé doit contenir un salaire et de 1 à 3 crédits additionnels.",
            })

    return result


CHECKERS = {
    "THREE_CONSECUTIVE_PERIODS": check_three_consecutive_periods,
    "ONE_SALARY_TRANSFER_PER_MONTH": check_salary_transfer_count,
    "SALARY_AMOUNT_MATCH": check_salary_amount_match,
    "SALARY_LABEL_EMPLOYER": check_salary_label_employer,
    "TRANSACTION_TOTALS": check_transaction_totals,
    "BALANCE_EQUATION": check_balance_equation,
    "BALANCE_CHAINING": check_balance_chaining,
    "NON_NEGATIVE_BALANCE": check_non_negative_balance,
    "ADDITIONAL_CREDITS_COUNT": check_additional_credits_count,
}


def summarize(
    dossier_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = []

    for rule_id in RULE_IDS:
        applicable = []
        violated = []
        unavailable = []

        for dossier_id, results in dossier_results.items():
            result = results[rule_id]
            if not result["applicable"]:
                unavailable.append(dossier_id)
            elif result["violations"]:
                violated.append(dossier_id)
            else:
                applicable.append(dossier_id)

        summary.append({
            "rule_id": rule_id,
            "passed_dossiers": len(applicable),
            "violated_dossiers": len(violated),
            "unavailable_dossiers": len(unavailable),
            "violation_rate_among_available": round(
                len(violated) / (len(applicable) + len(violated)),
                4,
            ) if (len(applicable) + len(violated)) else None,
            "example_violations": violated[:10],
            "example_unavailable": unavailable[:10],
        })

    return summary


def write_summary_csv(rows: list[dict[str, Any]]) -> None:
    with SUMMARY_CSV_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "rule_id",
            "passed_dossiers",
            "violated_dossiers",
            "unavailable_dossiers",
            "violation_rate_among_available",
            "example_violations",
            "example_unavailable",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                **row,
                "example_violations": "|".join(row["example_violations"]),
                "example_unavailable": "|".join(row["example_unavailable"]),
            })


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = Neo4jClient()
    try:
        client.verifier_connexion()

        field_rows = execute(
            client,
            f"""
            MATCH (d:Dossier)-[:CONTIENT]->(doc:Document)-[:POSSEDE]->(c:Champ)
            WHERE {TEST_FILTER}
            RETURN
                d.dossier_id AS dossier_id,
                doc.doc_id AS doc_id,
                doc.doc_type AS doc_type,
                c.nom AS field_name,
                c.valeur AS raw_value,
                c.valeur_normalisee AS normalized_value
            ORDER BY dossier_id, doc_id, field_name
            """,
        )

        transaction_rows = execute(
            client,
            f"""
            MATCH (d:Dossier)-[:CONTIENT]->(doc:Document)
                  -[:CONTIENT_TRANSACTION]->(t:Transaction)
            WHERE {TEST_FILTER}
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
        )

        dossiers = build_graph_snapshot(
            field_rows=field_rows,
            transaction_rows=transaction_rows,
        )

        dossier_results: dict[str, dict[str, Any]] = {}
        for dossier_id, dossier in sorted(dossiers.items()):
            dossier_results[dossier_id] = {}
            for rule_id, checker in CHECKERS.items():
                try:
                    dossier_results[dossier_id][rule_id] = checker(dossier)
                except Exception as error:
                    dossier_results[dossier_id][rule_id] = {
                        "applicable": False,
                        "violations": [],
                        "details": [],
                        "error": str(error),
                    }

        summary = summarize(dossier_results)

        report = {
            "audit_type": "BUSINESS_RULES_OUTCOMES_READ_ONLY",
            "mutates_neo4j": False,
            "scores_fraud": False,
            "dossiers_analyzed": len(dossiers),
            "rules_analyzed": RULE_IDS,
            "summary": summary,
            "dossier_results": dossier_results,
            "notes": [
                "Cet audit mesure le comportement des règles sur les dossiers légitimes.",
                "Une violation peut venir d'une vraie incohérence du dataset ou d'un problème d'extraction.",
                "Aucun seuil de score final n'est défini à cette étape.",
                "RIB, IBAN et Benford sont exclus.",
            ],
        }

        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_summary_csv(summary)

        print("\n===== AUDIT RÉSULTATS RÈGLES MÉTIER =====")
        print(f"Dossiers analysés : {len(dossiers)}")

        for row in summary:
            print(
                f"- {row['rule_id']} | "
                f"OK={row['passed_dossiers']} | "
                f"violations={row['violated_dossiers']} | "
                f"indisponibles={row['unavailable_dossiers']}"
            )

        print(f"\nRapport détaillé : {REPORT_PATH}")
        print(f"Résumé CSV : {SUMMARY_CSV_PATH}")
        print("Aucun score de fraude n'a été calculé.")
        print("Aucune donnée Neo4j n'a été modifiée.")

    finally:
        client.fermer()


if __name__ == "__main__":
    main()
