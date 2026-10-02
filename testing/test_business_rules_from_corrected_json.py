from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("data/json_corrected")
DEFAULT_OUTPUT = Path(
    "data/testing/business_rules/corrected_json_report.json"
)

SCORABLE_RULES = [
    "THREE_CONSECUTIVE_PERIODS",
    "ONE_SALARY_TRANSFER_PER_MONTH",
    "SALARY_AMOUNT_MATCH",
    "SALARY_LABEL_EMPLOYER",
    "TRANSACTION_TOTALS",
    "BALANCE_EQUATION",
    "BALANCE_CHAINING",
]

# Conservée comme information, mais pas dans le score :
DIAGNOSTIC_RULES = [
    "NON_NEGATIVE_BALANCE",
]

ALL_RULES = SCORABLE_RULES + DIAGNOSTIC_RULES

# ADDITIONAL_CREDITS_COUNT est volontairement supprimée.
TOLERANCE_MONTANT = 1.0
EMPLOYER_SIMILARITY_MIN = 0.75


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
    patterns = [
        r"^(0?[1-9]|1[0-2])[/_-](\d{4})$",
        r"^(\d{4})[/_-](0?[1-9]|1[0-2])$",
        r"^\d{2}[/_-](0?[1-9]|1[0-2])[/_-](\d{4})$",
    ]

    match = re.match(patterns[0], text)
    if match:
        return int(match.group(2)), int(match.group(1))

    match = re.match(patterns[1], text)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.match(patterns[2], text)
    if match:
        return int(match.group(2)), int(match.group(1))

    return None


def period_label(period: tuple[int, int] | None) -> str | None:
    if period is None:
        return None
    return f"{period[0]:04d}-{period[1]:02d}"


def consecutive(periods: list[tuple[int, int]]) -> bool:
    if len(periods) != 3 or len(set(periods)) != 3:
        return False

    indexes = sorted(year * 12 + month for year, month in periods)
    return indexes[1] == indexes[0] + 1 and indexes[2] == indexes[1] + 1


def canonical_doc_type(value: Any) -> str:
    compact = normalize_text(value).replace(" ", "_")
    aliases = {
        "RELEVE_BANCAIRE": "RELEVE_BANCAIRE",
        "BULLETIN_SALAIRE": "BULLETIN_SALAIRE",
        "ATTESTATION_TRAVAIL": "ATTESTATION_TRAVAIL",
        "CIN": "CIN",
        "QUITTANCE": "QUITTANCE",
        "RIB": "RIB",
    }
    return aliases.get(compact, compact)


def dossier_id_from_path(path: Path) -> str | None:
    match = re.match(r"^(D\d+)", path.stem, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    for parent in path.parents:
        if re.fullmatch(r"D\d+", parent.name, re.IGNORECASE):
            return parent.name.upper()

    return None


def load_corrected_documents(
    root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[dict[str, str]] = []

    for path in sorted(root.rglob("*.json")):
        dossier_id = dossier_id_from_path(path)
        if dossier_id is None:
            continue

        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append({
                "path": str(path),
                "error": str(error),
            })
            continue

        if not isinstance(document, dict):
            continue

        doc_type = canonical_doc_type(document.get("doc_type"))
        if doc_type not in {
            "RELEVE_BANCAIRE",
            "BULLETIN_SALAIRE",
            "ATTESTATION_TRAVAIL",
            "CIN",
            "QUITTANCE",
            "RIB",
        }:
            continue

        document = dict(document)
        document["_source_path"] = str(path)
        document["_doc_id"] = path.stem
        document["_canonical_doc_type"] = doc_type
        grouped[dossier_id].append(document)

    return dict(sorted(grouped.items())), errors


def documents_of_type(
    documents: list[dict[str, Any]],
    doc_type: str,
) -> list[dict[str, Any]]:
    return [
        document for document in documents
        if document.get("_canonical_doc_type") == doc_type
    ]


def statement_period(document: dict[str, Any]) -> tuple[int, int] | None:
    return (
        parse_period(document.get("date_solde_cloture"))
        or parse_period(document.get("periode"))
        or parse_period(document.get("date_solde_ouverture"))
    )


def bulletin_period(document: dict[str, Any]) -> tuple[int, int] | None:
    return parse_period(document.get("periode"))


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
        and "SALAIRE" in normalize_text(transaction.get("libelle"))
        and normalize_text(transaction.get("sens")) == "C"
    ]


def make_result(
    available: bool,
    triggered: bool = False,
    details: Any = None,
    violations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not available:
        return {
            "status": "UNAVAILABLE",
            "score": None,
            "available": False,
            "details": details,
            "violations": violations or [],
        }

    return {
        "status": "TRIGGERED" if triggered else "NORMAL",
        "score": 1 if triggered else 0,
        "available": True,
        "details": details,
        "violations": violations or [],
    }


def check_three_consecutive_periods(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(documents, "RELEVE_BANCAIRE")
    bulletins = documents_of_type(documents, "BULLETIN_SALAIRE")

    statement_periods = [statement_period(doc) for doc in statements]
    bulletin_periods = [bulletin_period(doc) for doc in bulletins]

    if (
        len(statements) != 3
        or len(bulletins) != 3
        or any(period is None for period in statement_periods)
        or any(period is None for period in bulletin_periods)
    ):
        return make_result(
            available=False,
            details={
                "statement_count": len(statements),
                "bulletin_count": len(bulletins),
                "statement_periods": [
                    period_label(period) for period in statement_periods
                ],
                "bulletin_periods": [
                    period_label(period) for period in bulletin_periods
                ],
            },
        )

    statement_periods_ok = [
        period for period in statement_periods if period is not None
    ]
    bulletin_periods_ok = [
        period for period in bulletin_periods if period is not None
    ]

    triggered = not (
        consecutive(statement_periods_ok)
        and consecutive(bulletin_periods_ok)
        and set(statement_periods_ok) == set(bulletin_periods_ok)
    )

    details = {
        "statement_periods": sorted(
            period_label(period) for period in statement_periods_ok
        ),
        "bulletin_periods": sorted(
            period_label(period) for period in bulletin_periods_ok
        ),
    }
    violations = []
    if triggered:
        violations.append({
            "reason": (
                "Les relevés et bulletins ne couvrent pas "
                "les mêmes trois mois consécutifs."
            )
        })

    return make_result(True, triggered, details, violations)


def check_one_salary_transfer_per_month(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(documents, "RELEVE_BANCAIRE")
    if not statements:
        return make_result(False)

    details = []
    violations = []

    for statement in statements:
        transactions = statement.get("transactions")
        if not isinstance(transactions, list):
            return make_result(False, details={
                "reason": "Transactions absentes.",
                "doc_id": statement.get("_doc_id"),
            })

        salaries = salary_transactions(statement)
        item = {
            "doc_id": statement.get("_doc_id"),
            "period": period_label(statement_period(statement)),
            "salary_count": len(salaries),
        }
        details.append(item)

        if len(salaries) != 1:
            violations.append({
                **item,
                "reason": (
                    "Le relevé doit contenir exactement "
                    "un virement de salaire."
                ),
            })

    return make_result(
        True,
        bool(violations),
        details,
        violations,
    )


def check_salary_amount_match(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(documents, "RELEVE_BANCAIRE")
    bulletins = documents_of_type(documents, "BULLETIN_SALAIRE")

    bulletin_by_period: dict[tuple[int, int], dict[str, Any]] = {}
    for bulletin in bulletins:
        period = bulletin_period(bulletin)
        if period is not None:
            bulletin_by_period[period] = bulletin

    statement_by_period: dict[tuple[int, int], dict[str, Any]] = {}
    for statement in statements:
        period = statement_period(statement)
        if period is not None:
            statement_by_period[period] = statement

    common_periods = sorted(
        set(bulletin_by_period) & set(statement_by_period)
    )
    if not common_periods:
        return make_result(False)

    details = []
    violations = []

    for period in common_periods:
        bulletin = bulletin_by_period[period]
        statement = statement_by_period[period]
        net = to_number(bulletin.get("net_a_payer"))
        salaries = salary_transactions(statement)

        if net is None or len(salaries) != 1:
            return make_result(False, details={
                "period": period_label(period),
                "reason": (
                    "Net à payer absent ou nombre de virements "
                    "salaire différent de 1."
                ),
            })

        salary_amount = to_number(salaries[0].get("montant"))
        if salary_amount is None:
            return make_result(False, details={
                "period": period_label(period),
                "reason": "Montant du virement salaire absent.",
            })

        difference = abs(net - salary_amount)
        item = {
            "period": period_label(period),
            "bulletin_doc": bulletin.get("_doc_id"),
            "statement_doc": statement.get("_doc_id"),
            "net_a_payer": net,
            "salary_transfer": salary_amount,
            "difference": round(difference, 2),
        }
        details.append(item)

        if difference > TOLERANCE_MONTANT:
            violations.append({
                **item,
                "reason": (
                    "Le net à payer est différent du "
                    "virement salaire."
                ),
            })

    return make_result(
        True,
        bool(violations),
        details,
        violations,
    )


def extract_employer(
    documents: list[dict[str, Any]],
) -> str | None:
    candidates = []

    for document in documents:
        if document.get("_canonical_doc_type") in {
            "BULLETIN_SALAIRE",
            "ATTESTATION_TRAVAIL",
        }:
            employer = normalize_text(document.get("employeur"))
            if employer:
                candidates.append(employer)

    if not candidates:
        return None

    counts: dict[str, int] = defaultdict(int)
    for employer in candidates:
        counts[employer] += 1

    return max(counts, key=lambda value: counts[value])


def employer_similarity(employer: str, label: str) -> float:
    employer_compact = normalize_text(employer).replace(" ", "")
    label_compact = normalize_text(label).replace(" ", "")

    if employer_compact in label_compact:
        return 1.0
    normalized_label = normalize_text(label)
    if not employer or not normalized_label:
        return 0.0
    if employer in normalized_label:
        return 1.0

    employer_tokens = set(employer.split())
    label_tokens = set(normalized_label.split())
    token_score = (
        len(employer_tokens & label_tokens) / len(employer_tokens)
        if employer_tokens
        else 0.0
    )
    sequence_score = SequenceMatcher(
        None,
        employer,
        normalized_label,
    ).ratio()
    return max(token_score, sequence_score)


def check_salary_label_employer(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    employer = extract_employer(documents)
    statements = documents_of_type(documents, "RELEVE_BANCAIRE")

    if employer is None or not statements:
        return make_result(False)

    details = []
    violations = []

    for statement in statements:
        salaries = salary_transactions(statement)
        if len(salaries) != 1:
            return make_result(False, details={
                "doc_id": statement.get("_doc_id"),
                "reason": (
                    "Impossible de choisir un unique "
                    "virement salaire."
                ),
            })

        label = str(salaries[0].get("libelle") or "")
        similarity = employer_similarity(employer, label)
        item = {
            "doc_id": statement.get("_doc_id"),
            "period": period_label(statement_period(statement)),
            "employer": employer,
            "salary_label": label,
            "similarity": round(similarity, 4),
        }
        details.append(item)

        if similarity < EMPLOYER_SIMILARITY_MIN:
            violations.append({
                **item,
                "reason": (
                    "Le libellé du salaire ne correspond "
                    "pas à l'employeur."
                ),
            })

    return make_result(
        True,
        bool(violations),
        details,
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
        sens = normalize_text(transaction.get("sens"))
        if amount is None:
            return None

        if sens == "D":
            debits += abs(amount)
        elif sens == "C":
            credits += abs(amount)
        else:
            return None

    return debits, credits


def check_transaction_totals(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(documents, "RELEVE_BANCAIRE")
    if not statements:
        return make_result(False)

    details = []
    violations = []

    for statement in statements:
        sums = transaction_sums(statement)
        declared_debits = to_number(statement.get("total_debits"))
        declared_credits = to_number(statement.get("total_credits"))

        if (
            sums is None
            or declared_debits is None
            or declared_credits is None
        ):
            return make_result(False, details={
                "doc_id": statement.get("_doc_id"),
                "reason": "Totaux ou transactions absents.",
            })

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
            "credit_difference": round(credit_difference, 2),
        }
        details.append(item)

        if (
            debit_difference > TOLERANCE_MONTANT
            or credit_difference > TOLERANCE_MONTANT
        ):
            violations.append({
                **item,
                "reason": (
                    "Les totaux déclarés ne correspondent "
                    "pas aux transactions."
                ),
            })

    return make_result(
        True,
        bool(violations),
        details,
        violations,
    )


def check_balance_equation(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(documents, "RELEVE_BANCAIRE")
    if not statements:
        return make_result(False)

    details = []
    violations = []

    for statement in statements:
        opening = to_number(statement.get("solde_ouverture"))
        closing = to_number(statement.get("solde_cloture"))
        credits = to_number(statement.get("total_credits"))
        debits = to_number(statement.get("total_debits"))

        if None in {opening, closing, credits, debits}:
            return make_result(False, details={
                "doc_id": statement.get("_doc_id"),
                "reason": (
                    "Solde ou totaux absents."
                ),
            })

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
        details.append(item)

        if difference > TOLERANCE_MONTANT:
            violations.append({
                **item,
                "reason": (
                    "L'équation du solde n'est pas respectée."
                ),
            })

    return make_result(
        True,
        bool(violations),
        details,
        violations,
    )


def check_balance_chaining(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(documents, "RELEVE_BANCAIRE")
    sortable = []

    for statement in statements:
        period = statement_period(statement)
        opening = to_number(statement.get("solde_ouverture"))
        closing = to_number(statement.get("solde_cloture"))
        if period is None or opening is None or closing is None:
            continue
        sortable.append((period, statement, opening, closing))

    if len(sortable) < 2:
        return make_result(False)

    sortable.sort(key=lambda item: item[0])
    details = []
    violations = []

    for previous, current in zip(sortable, sortable[1:]):
        previous_period, previous_doc, _, previous_closing = previous
        current_period, current_doc, current_opening, _ = current

        difference = abs(previous_closing - current_opening)
        item = {
            "previous_doc": previous_doc.get("_doc_id"),
            "previous_period": period_label(previous_period),
            "previous_closing": previous_closing,
            "current_doc": current_doc.get("_doc_id"),
            "current_period": period_label(current_period),
            "current_opening": current_opening,
            "difference": round(difference, 2),
        }
        details.append(item)

        if difference > TOLERANCE_MONTANT:
            violations.append({
                **item,
                "reason": (
                    "Le solde final du mois précédent "
                    "diffère du solde initial suivant."
                ),
            })

    return make_result(
        True,
        bool(violations),
        details,
        violations,
    )


def check_non_negative_balance(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    statements = documents_of_type(documents, "RELEVE_BANCAIRE")
    if not statements:
        return make_result(False)

    details = []
    violations = []

    for statement in statements:
        opening = to_number(statement.get("solde_ouverture"))
        closing = to_number(statement.get("solde_cloture"))

        if opening is None or closing is None:
            return make_result(False, details={
                "doc_id": statement.get("_doc_id"),
                "reason": "Solde absent.",
            })

        item = {
            "doc_id": statement.get("_doc_id"),
            "period": period_label(statement_period(statement)),
            "opening": opening,
            "closing": closing,
        }
        details.append(item)

        if opening < 0 or closing < 0:
            violations.append({
                **item,
                "reason": (
                    "Présence d'un solde négatif. "
                    "Information diagnostique uniquement."
                ),
            })

    return make_result(
        True,
        bool(violations),
        details,
        violations,
    )


CHECKERS = {
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


def evaluate_dossier(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    rules = {
        rule_id: CHECKERS[rule_id](documents)
        for rule_id in ALL_RULES
    }

    available_scores = [
        rules[rule_id]["score"]
        for rule_id in SCORABLE_RULES
        if rules[rule_id]["score"] is not None
    ]
    business_score = (
        max(available_scores)
        if available_scores
        else None
    )

    if business_score is None:
        status = "UNAVAILABLE"
    elif business_score == 0:
        status = "NORMAL"
    else:
        status = "TRIGGERED"

    return {
        "business_score": business_score,
        "status": status,
        "rules": rules,
        "summary": {
            "normal": sum(
                result["status"] == "NORMAL"
                for result in rules.values()
            ),
            "triggered": sum(
                result["status"] == "TRIGGERED"
                for result in rules.values()
            ),
            "unavailable": sum(
                result["status"] == "UNAVAILABLE"
                for result in rules.values()
            ),
        },
    }


def print_report(
    results: dict[str, dict[str, Any]],
) -> None:
    print(
        "\n===== TEST RÈGLES MÉTIER — JSON_CORRECTED ====="
    )

    for dossier_id, result in results.items():
        print(
            f"\nDossier : {dossier_id} | "
            f"score={result['business_score']} | "
            f"statut={result['status']}"
        )

        for rule_id in ALL_RULES:
            rule = result["rules"][rule_id]
            usage = (
                "SCORE"
                if rule_id in SCORABLE_RULES
                else "DIAGNOSTIC"
            )
            print(
                f"- {rule_id} | "
                f"{rule['status']} | "
                f"valeur={rule['score']} | "
                f"{usage}"
            )

    counts = {
        "NORMAL": sum(
            result["status"] == "NORMAL"
            for result in results.values()
        ),
        "TRIGGERED": sum(
            result["status"] == "TRIGGERED"
            for result in results.values()
        ),
        "UNAVAILABLE": sum(
            result["status"] == "UNAVAILABLE"
            for result in results.values()
        ),
    }

    print("\n===== RÉSUMÉ GLOBAL =====")
    print(f"Dossiers analysés : {len(results)}")
    print(f"Normaux : {counts['NORMAL']}")
    print(f"Déclenchés : {counts['TRIGGERED']}")
    print(f"Indisponibles : {counts['UNAVAILABLE']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Teste les règles métier sur tous les JSON corrigés, "
            "sans Neo4j et sans modifier les données."
        )
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Racine contenant les JSON corrigés.",
    )
    parser.add_argument(
        "--dossiers",
        nargs="*",
        default=None,
        help="Exemple : --dossiers D001 D002",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {root}"
        )

    grouped, load_errors = load_corrected_documents(root)

    if args.dossiers:
        requested = {
            dossier_id.upper()
            for dossier_id in args.dossiers
        }
        grouped = {
            dossier_id: documents
            for dossier_id, documents in grouped.items()
            if dossier_id in requested
        }

    results = {
        dossier_id: evaluate_dossier(documents)
        for dossier_id, documents in grouped.items()
    }

    report = {
        "audit_type":
            "BUSINESS_RULES_FROM_CORRECTED_JSON",
        "source_root": str(root),
        "uses_neo4j": False,
        "mutates_data": False,
        "additional_credits_count_removed": True,
        "scorable_rules": SCORABLE_RULES,
        "diagnostic_rules": DIAGNOSTIC_RULES,
        "dossiers_analyzed": list(results),
        "load_errors": load_errors,
        "results": results,
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

    print_report(results)
    print(f"\nRapport : {output_path}")
    print("Neo4j n'a pas été utilisé.")
    print("Aucune donnée n'a été modifiée.")
    print(
        "ADDITIONAL_CREDITS_COUNT a été supprimée."
    )


if __name__ == "__main__":
    main()
