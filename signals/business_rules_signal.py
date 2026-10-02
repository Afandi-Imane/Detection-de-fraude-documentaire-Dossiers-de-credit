from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable


SIGNAL_ID = "BUSINESS_RULES"
SIGNAL_VERSION = "1.0"

SCORABLE_RULES = [
    "THREE_CONSECUTIVE_PERIODS",
    "ONE_SALARY_TRANSFER_PER_MONTH",
    "SALARY_AMOUNT_MATCH",
    "SALARY_LABEL_EMPLOYER",
    "TRANSACTION_TOTALS",
    "BALANCE_EQUATION",
    "BALANCE_CHAINING",
]

DIAGNOSTIC_RULES = [
    "NON_NEGATIVE_BALANCE",
]

ALL_RULES = SCORABLE_RULES + DIAGNOSTIC_RULES

# ADDITIONAL_CREDITS_COUNT est volontairement absente.
AMOUNT_TOLERANCE = 1.0
EMPLOYER_SIMILARITY_MIN = 0.75


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Le fichier dossier.json doit contenir un objet JSON.")
    return data


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return output


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_direction(value: Any) -> str:
    text = normalize_text(value)
    if text in {"C", "CREDIT", "CREDITS"}:
        return "C"
    if text in {"D", "DEBIT", "DEBITS"}:
        return "D"
    return text


def to_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = str(value).strip().replace("\u00a0", " ")
    if not text:
        return None

    text = text.replace(" ", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    else:
        text = text.replace(",", ".")

    text = re.sub(r"[^0-9.+-]", "", text)
    try:
        number = float(text)
    except ValueError:
        return None

    return number if math.isfinite(number) else None


def parse_period(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None

    text = str(value).strip()

    match = re.fullmatch(r"(0?[1-9]|1[0-2])[/_-](\d{4})", text)
    if match:
        return int(match.group(2)), int(match.group(1))

    match = re.fullmatch(r"(\d{4})[/_-](0?[1-9]|1[0-2])", text)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.fullmatch(
        r"\d{1,2}[/_-](0?[1-9]|1[0-2])[/_-](\d{4})",
        text,
    )
    if match:
        return int(match.group(2)), int(match.group(1))

    match = re.fullmatch(
        r"(\d{4})[/_-](0?[1-9]|1[0-2])[/_-]\d{1,2}",
        text,
    )
    if match:
        return int(match.group(1)), int(match.group(2))

    return None


def period_label(period: tuple[int, int] | None) -> str | None:
    if period is None:
        return None
    return f"{period[0]:04d}-{period[1]:02d}"


def are_three_consecutive(periods: list[tuple[int, int]]) -> bool:
    if len(periods) != 3 or len(set(periods)) != 3:
        return False
    indexes = sorted(year * 12 + month for year, month in periods)
    return (
        indexes[1] == indexes[0] + 1
        and indexes[2] == indexes[1] + 1
    )


def canonical_doc_type(value: Any) -> str:
    compact = normalize_text(value).replace(" ", "_")
    aliases = {
        "RELEVE_BANCAIRE": "RELEVE_BANCAIRE",
        "RELEVÉ_BANCAIRE": "RELEVE_BANCAIRE",
        "BULLETIN_SALAIRE": "BULLETIN_SALAIRE",
        "BULLETIN_DE_SALAIRE": "BULLETIN_SALAIRE",
        "ATTESTATION_TRAVAIL": "ATTESTATION_TRAVAIL",
        "ATTESTATION_DE_TRAVAIL": "ATTESTATION_TRAVAIL",
        "CIN": "CIN",
        "QUITTANCE": "QUITTANCE",
        "RIB": "RIB",
    }
    return aliases.get(compact, compact)


def usable_field_value(field: dict[str, Any]) -> Any:
    status = normalize_text(field.get("validation_status"))
    if status in {
        "MISSING",
        "INVALID FORMAT",
        "ABSENT DU TEMPLATE",
        "OPTIONAL MISSING",
    }:
        return None

    if "normalized_value" in field:
        value = field.get("normalized_value")
        if value is not None:
            return value

    return field.get("raw_value")


def adapt_dossier_documents(
    dossier: dict[str, Any],
) -> list[dict[str, Any]]:
    documents = dossier.get("documents")
    if not isinstance(documents, list):
        raise ValueError(
            "Le dossier.json doit contenir une liste 'documents'."
        )

    adapted: list[dict[str, Any]] = []

    for index, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            continue

        flat: dict[str, Any] = {
            "_doc_id": (
                document.get("doc_id")
                or document.get("document_id")
                or document.get("file")
                or document.get("_file")
                or f"DOCUMENT_{index}"
            ),
            "_canonical_doc_type": canonical_doc_type(
                document.get("doc_type")
                or document.get("document_type")
                or document.get("type")
            ),
        }

        fields = document.get("fields")
        if isinstance(fields, list):
            for field in fields:
                if not isinstance(field, dict):
                    continue
                name = str(field.get("field_name") or "").strip()
                if not name:
                    continue
                flat[name] = usable_field_value(field)

        # Compatibilité avec des champs déjà présents au niveau du document.
        for key, value in document.items():
            if key not in {
                "fields",
                "doc_type",
                "document_type",
                "type",
            } and key not in flat:
                flat[key] = value

        # Certains builders conservent les transactions directement.
        if "transactions" not in flat and isinstance(
            document.get("transactions"),
            list,
        ):
            flat["transactions"] = document["transactions"]

        adapted.append(flat)

    return adapted


def documents_of_type(
    documents: list[dict[str, Any]],
    document_type: str,
) -> list[dict[str, Any]]:
    return [
        document
        for document in documents
        if document.get("_canonical_doc_type") == document_type
    ]


def statement_period(
    document: dict[str, Any],
) -> tuple[int, int] | None:
    return (
        parse_period(document.get("periode"))
        or parse_period(document.get("date_solde_cloture"))
        or parse_period(document.get("date_solde_ouverture"))
    )


def bulletin_period(
    document: dict[str, Any],
) -> tuple[int, int] | None:
    return parse_period(document.get("periode"))


def salary_transactions(
    statement: dict[str, Any],
) -> list[dict[str, Any]]:
    transactions = statement.get("transactions")
    if not isinstance(transactions, list):
        return []

    salaries = []
    for transaction in transactions:
        if not isinstance(transaction, dict):
            continue

        is_salary = transaction.get("est_salaire") is True or (
            "SALAIRE" in normalize_text(transaction.get("libelle"))
        )
        if (
            is_salary
            and normalize_direction(transaction.get("sens")) == "C"
        ):
            salaries.append(transaction)

    return salaries


def make_rule_result(
    available: bool,
    triggered: bool = False,
    evidence: Any = None,
    violations: list[dict[str, Any]] | None = None,
    diagnostic: bool = False,
) -> dict[str, Any]:
    if not available:
        return {
            "available": False,
            "status": "UNAVAILABLE",
            "score": None,
            "enabled_in_score": not diagnostic,
            "evidence": evidence,
            "violations": violations or [],
        }

    return {
        "available": True,
        "status": "TRIGGERED" if triggered else "NORMAL",
        "score": 1 if triggered else 0,
        "enabled_in_score": not diagnostic,
        "evidence": evidence,
        "violations": violations or [],
    }


def check_three_consecutive_periods(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(
        documents,
        "RELEVE_BANCAIRE",
    )
    bulletins = documents_of_type(
        documents,
        "BULLETIN_SALAIRE",
    )

    statement_periods = [
        statement_period(document)
        for document in statements
    ]
    bulletin_periods = [
        bulletin_period(document)
        for document in bulletins
    ]

    evidence = {
        "statement_count": len(statements),
        "bulletin_count": len(bulletins),
        "statement_periods": [
            period_label(period)
            for period in statement_periods
        ],
        "bulletin_periods": [
            period_label(period)
            for period in bulletin_periods
        ],
    }

    if (
        len(statements) != 3
        or len(bulletins) != 3
        or any(period is None for period in statement_periods)
        or any(period is None for period in bulletin_periods)
    ):
        return make_rule_result(
            available=False,
            evidence=evidence,
        )

    statement_values = [
        period
        for period in statement_periods
        if period is not None
    ]
    bulletin_values = [
        period
        for period in bulletin_periods
        if period is not None
    ]

    triggered = not (
        are_three_consecutive(statement_values)
        and are_three_consecutive(bulletin_values)
        and set(statement_values) == set(bulletin_values)
    )

    violations = []
    if triggered:
        violations.append({
            "reason": (
                "Les relevés et bulletins ne couvrent pas "
                "les mêmes trois mois consécutifs."
            )
        })

    return make_rule_result(
        True,
        triggered,
        evidence,
        violations,
    )


def check_one_salary_transfer_per_month(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(
        documents,
        "RELEVE_BANCAIRE",
    )
    if not statements:
        return make_rule_result(False)

    evidence = []
    violations = []

    for statement in statements:
        transactions = statement.get("transactions")
        if not isinstance(transactions, list):
            return make_rule_result(
                False,
                evidence={
                    "doc_id": statement.get("_doc_id"),
                    "reason": "Transactions absentes.",
                },
            )

        count = len(salary_transactions(statement))
        item = {
            "doc_id": statement.get("_doc_id"),
            "period": period_label(statement_period(statement)),
            "salary_count": count,
        }
        evidence.append(item)

        if count != 1:
            violations.append({
                **item,
                "reason": (
                    "Le relevé doit contenir exactement "
                    "un virement de salaire."
                ),
            })

    return make_rule_result(
        True,
        bool(violations),
        evidence,
        violations,
    )


def check_salary_amount_match(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(
        documents,
        "RELEVE_BANCAIRE",
    )
    bulletins = documents_of_type(
        documents,
        "BULLETIN_SALAIRE",
    )

    bulletin_by_period = {
        period: document
        for document in bulletins
        if (period := bulletin_period(document)) is not None
    }
    statement_by_period = {
        period: document
        for document in statements
        if (period := statement_period(document)) is not None
    }

    common_periods = sorted(
        set(bulletin_by_period) & set(statement_by_period)
    )
    if not common_periods:
        return make_rule_result(False)

    evidence = []
    violations = []

    for period in common_periods:
        bulletin = bulletin_by_period[period]
        statement = statement_by_period[period]
        net_amount = to_number(bulletin.get("net_a_payer"))
        salaries = salary_transactions(statement)

        if net_amount is None or len(salaries) != 1:
            return make_rule_result(
                False,
                evidence={
                    "period": period_label(period),
                    "reason": (
                        "Net à payer absent ou nombre de virements "
                        "salaire différent de 1."
                    ),
                },
            )

        transfer_amount = to_number(salaries[0].get("montant"))
        if transfer_amount is None:
            return make_rule_result(
                False,
                evidence={
                    "period": period_label(period),
                    "reason": "Montant du virement salaire absent.",
                },
            )

        difference = abs(net_amount - transfer_amount)
        item = {
            "period": period_label(period),
            "bulletin_doc": bulletin.get("_doc_id"),
            "statement_doc": statement.get("_doc_id"),
            "net_a_payer": net_amount,
            "salary_transfer": transfer_amount,
            "difference": round(difference, 2),
        }
        evidence.append(item)

        if difference > AMOUNT_TOLERANCE:
            violations.append({
                **item,
                "reason": (
                    "Le net à payer est différent "
                    "du virement salaire."
                ),
            })

    return make_rule_result(
        True,
        bool(violations),
        evidence,
        violations,
    )


def extract_reference_employer(
    documents: list[dict[str, Any]],
) -> str | None:
    candidates = []

    for document in documents:
        if document.get("_canonical_doc_type") not in {
            "BULLETIN_SALAIRE",
            "ATTESTATION_TRAVAIL",
        }:
            continue

        employer = normalize_text(document.get("employeur"))
        if employer:
            candidates.append(employer)

    if not candidates:
        return None

    return Counter(candidates).most_common(1)[0][0]


def employer_similarity(
    employer: str,
    label: str,
) -> float:
    normalized_employer = normalize_text(employer)
    normalized_label = normalize_text(label)

    if not normalized_employer or not normalized_label:
        return 0.0

    compact_employer = normalized_employer.replace(" ", "")
    compact_label = normalized_label.replace(" ", "")

    if compact_employer in compact_label:
        return 1.0

    employer_tokens = set(normalized_employer.split())
    label_tokens = set(normalized_label.split())

    token_score = (
        len(employer_tokens & label_tokens) / len(employer_tokens)
        if employer_tokens
        else 0.0
    )
    sequence_score = SequenceMatcher(
        None,
        normalized_employer,
        normalized_label,
    ).ratio()

    return max(token_score, sequence_score)


def check_salary_label_employer(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    employer = extract_reference_employer(documents)
    statements = documents_of_type(
        documents,
        "RELEVE_BANCAIRE",
    )

    if employer is None or not statements:
        return make_rule_result(False)

    evidence = []
    violations = []

    for statement in statements:
        salaries = salary_transactions(statement)
        if len(salaries) != 1:
            return make_rule_result(
                False,
                evidence={
                    "doc_id": statement.get("_doc_id"),
                    "reason": (
                        "Impossible de choisir un unique "
                        "virement salaire."
                    ),
                },
            )

        label = str(salaries[0].get("libelle") or "")
        similarity = employer_similarity(employer, label)
        item = {
            "doc_id": statement.get("_doc_id"),
            "period": period_label(statement_period(statement)),
            "employer": employer,
            "salary_label": label,
            "similarity": round(similarity, 4),
        }
        evidence.append(item)

        if similarity < EMPLOYER_SIMILARITY_MIN:
            violations.append({
                **item,
                "reason": (
                    "Le libellé du salaire ne correspond "
                    "pas à l'employeur."
                ),
            })

    return make_rule_result(
        True,
        bool(violations),
        evidence,
        violations,
    )


def transaction_sums(
    statement: dict[str, Any],
) -> tuple[float, float] | None:
    transactions = statement.get("transactions")
    if not isinstance(transactions, list):
        return None

    debits = 0.0
    credits = 0.0

    for transaction in transactions:
        if not isinstance(transaction, dict):
            continue

        amount = to_number(transaction.get("montant"))
        direction = normalize_direction(transaction.get("sens"))

        if amount is None:
            return None

        if direction == "D":
            debits += abs(amount)
        elif direction == "C":
            credits += abs(amount)
        else:
            return None

    return debits, credits


def check_transaction_totals(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(
        documents,
        "RELEVE_BANCAIRE",
    )
    if not statements:
        return make_rule_result(False)

    evidence = []
    violations = []

    for statement in statements:
        sums = transaction_sums(statement)
        declared_debits = to_number(
            statement.get("total_debits")
        )
        declared_credits = to_number(
            statement.get("total_credits")
        )

        if (
            sums is None
            or declared_debits is None
            or declared_credits is None
        ):
            return make_rule_result(
                False,
                evidence={
                    "doc_id": statement.get("_doc_id"),
                    "reason": "Totaux ou transactions absents.",
                },
            )

        calculated_debits, calculated_credits = sums
        debit_difference = abs(
            declared_debits - calculated_debits
        )
        credit_difference = abs(
            declared_credits - calculated_credits
        )

        item = {
            "doc_id": statement.get("_doc_id"),
            "declared_debits": declared_debits,
            "calculated_debits": round(calculated_debits, 2),
            "declared_credits": declared_credits,
            "calculated_credits": round(calculated_credits, 2),
            "debit_difference": round(debit_difference, 2),
            "credit_difference": round(
                credit_difference,
                2,
            ),
        }
        evidence.append(item)

        if (
            debit_difference > AMOUNT_TOLERANCE
            or credit_difference > AMOUNT_TOLERANCE
        ):
            violations.append({
                **item,
                "reason": (
                    "Les totaux déclarés ne correspondent "
                    "pas aux transactions."
                ),
            })

    return make_rule_result(
        True,
        bool(violations),
        evidence,
        violations,
    )


def check_balance_equation(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(
        documents,
        "RELEVE_BANCAIRE",
    )
    if not statements:
        return make_rule_result(False)

    evidence = []
    violations = []

    for statement in statements:
        opening = to_number(statement.get("solde_ouverture"))
        closing = to_number(statement.get("solde_cloture"))
        credits = to_number(statement.get("total_credits"))
        debits = to_number(statement.get("total_debits"))

        if any(
            value is None
            for value in (opening, closing, credits, debits)
        ):
            return make_rule_result(
                False,
                evidence={
                    "doc_id": statement.get("_doc_id"),
                    "reason": "Solde ou totaux absents.",
                },
            )

        calculated_closing = opening + credits - debits
        difference = abs(closing - calculated_closing)

        item = {
            "doc_id": statement.get("_doc_id"),
            "period": period_label(statement_period(statement)),
            "opening": opening,
            "credits": credits,
            "debits": debits,
            "declared_closing": closing,
            "calculated_closing": round(
                calculated_closing,
                2,
            ),
            "difference": round(difference, 2),
        }
        evidence.append(item)

        if difference > AMOUNT_TOLERANCE:
            violations.append({
                **item,
                "reason": (
                    "L'équation du solde n'est pas respectée."
                ),
            })

    return make_rule_result(
        True,
        bool(violations),
        evidence,
        violations,
    )


def check_balance_chaining(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(
        documents,
        "RELEVE_BANCAIRE",
    )
    sortable = []

    for statement in statements:
        period = statement_period(statement)
        opening = to_number(statement.get("solde_ouverture"))
        closing = to_number(statement.get("solde_cloture"))

        if (
            period is not None
            and opening is not None
            and closing is not None
        ):
            sortable.append(
                (period, statement, opening, closing)
            )

    if len(sortable) < 2:
        return make_rule_result(False)

    sortable.sort(key=lambda item: item[0])
    evidence = []
    violations = []

    for previous, current in zip(sortable, sortable[1:]):
        (
            previous_period,
            previous_document,
            _,
            previous_closing,
        ) = previous
        (
            current_period,
            current_document,
            current_opening,
            _,
        ) = current

        difference = abs(
            previous_closing - current_opening
        )
        item = {
            "previous_doc": previous_document.get("_doc_id"),
            "previous_period": period_label(previous_period),
            "previous_closing": previous_closing,
            "current_doc": current_document.get("_doc_id"),
            "current_period": period_label(current_period),
            "current_opening": current_opening,
            "difference": round(difference, 2),
        }
        evidence.append(item)

        if difference > AMOUNT_TOLERANCE:
            violations.append({
                **item,
                "reason": (
                    "Le solde final du mois précédent "
                    "diffère du solde initial suivant."
                ),
            })

    return make_rule_result(
        True,
        bool(violations),
        evidence,
        violations,
    )


def check_non_negative_balance(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(
        documents,
        "RELEVE_BANCAIRE",
    )
    if not statements:
        return make_rule_result(
            False,
            diagnostic=True,
        )

    evidence = []
    violations = []

    for statement in statements:
        opening = to_number(statement.get("solde_ouverture"))
        closing = to_number(statement.get("solde_cloture"))

        if opening is None or closing is None:
            return make_rule_result(
                False,
                evidence={
                    "doc_id": statement.get("_doc_id"),
                    "reason": "Solde absent.",
                },
                diagnostic=True,
            )

        item = {
            "doc_id": statement.get("_doc_id"),
            "period": period_label(statement_period(statement)),
            "opening": opening,
            "closing": closing,
        }
        evidence.append(item)

        if opening < 0 or closing < 0:
            violations.append({
                **item,
                "reason": (
                    "Présence d'un solde négatif. "
                    "Diagnostic uniquement."
                ),
            })

    return make_rule_result(
        True,
        bool(violations),
        evidence,
        violations,
        diagnostic=True,
    )


CHECKERS: dict[
    str,
    Callable[[list[dict[str, Any]]], dict[str, Any]],
] = {
    "THREE_CONSECUTIVE_PERIODS":
        check_three_consecutive_periods,
    "ONE_SALARY_TRANSFER_PER_MONTH":
        check_one_salary_transfer_per_month,
    "SALARY_AMOUNT_MATCH":
        check_salary_amount_match,
    "SALARY_LABEL_EMPLOYER":
        check_salary_label_employer,
    "TRANSACTION_TOTALS":
        check_transaction_totals,
    "BALANCE_EQUATION":
        check_balance_equation,
    "BALANCE_CHAINING":
        check_balance_chaining,
    "NON_NEGATIVE_BALANCE":
        check_non_negative_balance,
}


def compute_business_rules_signal(
    dossier: dict[str, Any],
    dossier_json_path: str | Path | None = None,
) -> dict[str, Any]:
    documents = adapt_dossier_documents(dossier)
    rules = {
        rule_id: CHECKERS[rule_id](documents)
        for rule_id in ALL_RULES
    }

    available_scores = [
        rules[rule_id]["score"]
        for rule_id in SCORABLE_RULES
        if rules[rule_id]["score"] is not None
    ]

    score = (
        max(available_scores)
        if available_scores
        else None
    )
    available = bool(available_scores)

    if score is None:
        status = "UNAVAILABLE"
    elif score == 0:
        status = "NORMAL"
    else:
        status = "TRIGGERED"

    dossier_id = str(
        dossier.get("dossier_id")
        or dossier.get("id")
        or (
            Path(dossier_json_path).parent.name
            if dossier_json_path is not None
            else "UNKNOWN_DOSSIER"
        )
    )

    triggered_scoring = [
        rule_id
        for rule_id in SCORABLE_RULES
        if rules[rule_id]["status"] == "TRIGGERED"
    ]
    triggered_diagnostic = [
        rule_id
        for rule_id in DIAGNOSTIC_RULES
        if rules[rule_id]["status"] == "TRIGGERED"
    ]

    return {
        "signal_id": SIGNAL_ID,
        "signal_version": SIGNAL_VERSION,
        "dossier_id": dossier_id,
        "available": available,
        "score": score,
        "status": status,
        "aggregation": "MAX_AVAILABLE_RULE_SCORES",
        "uses_neo4j": False,
        "source": {
            "type": "DOSSIER_JSON",
            "path": (
                str(Path(dossier_json_path))
                if dossier_json_path is not None
                else None
            ),
        },
        "scorable_rules": SCORABLE_RULES,
        "diagnostic_rules": DIAGNOSTIC_RULES,
        "rules": rules,
        "summary": {
            "anomalies_count": len(triggered_scoring),
            "diagnostic_alerts_count": len(
                triggered_diagnostic
            ),
            "checks_available": sum(
                rules[rule_id]["available"]
                for rule_id in ALL_RULES
            ),
            "checks_unavailable": sum(
                not rules[rule_id]["available"]
                for rule_id in ALL_RULES
            ),
            "triggered_scoring_rules": triggered_scoring,
            "triggered_diagnostic_rules":
                triggered_diagnostic,
        },
    }


def run_business_rules_signal(
    dossier_json_path: str | Path,
    output_root: str | Path = "data/signals",
) -> dict[str, Any]:
    source = Path(dossier_json_path)
    dossier = load_json(source)
    result = compute_business_rules_signal(
        dossier,
        dossier_json_path=source,
    )

    output_path = (
        Path(output_root)
        / result["dossier_id"]
        / "business_rules.json"
    )
    result["output_path"] = str(output_path)
    save_json(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calcule le signal de règles métier directement "
            "depuis un dossier.json, sans Neo4j."
        )
    )
    parser.add_argument(
        "--dossier-json",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        default="data/signals",
    )
    args = parser.parse_args()

    result = run_business_rules_signal(
        args.dossier_json,
        args.output_root,
    )

    print("\n===== SIGNAL RÈGLES MÉTIER =====")
    print(f"Dossier : {result['dossier_id']}")
    print(f"Disponible : {result['available']}")
    print(f"Score : {result['score']}")
    print(f"Statut : {result['status']}")
    print(
        "Règles déclenchées : "
        f"{result['summary']['triggered_scoring_rules']}"
    )
    print(
        "Règles indisponibles : "
        f"{result['summary']['checks_unavailable']}"
    )
    print(f"Résultat : {result['output_path']}")
    print("Neo4j n'a pas été utilisé.")


if __name__ == "__main__":
    main()
