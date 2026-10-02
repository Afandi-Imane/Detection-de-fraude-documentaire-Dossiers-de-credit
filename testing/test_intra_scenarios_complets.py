from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.dossier_graph_loader import (  # noqa: E402
    _delete_existing_dossier,
    load_dossier_json,
)
from graph.neo4j_client import Neo4jClient  # noqa: E402
from signals.intra_dossier_signal import (  # noqa: E402
    normalize_address,
    normalize_text,
    run_intra_dossier_signal,
    text_anomaly_score,
    text_similarity,
)


TEST_PREFIX = "TEST_INTRA_FULL_"
DEFAULT_SOURCE = Path("data/json_validated/D147_D_TEST/dossier.json")
DEFAULT_WORK_ROOT = Path("data/testing/intra_scenarios_complets")
DEFAULT_SIGNAL_ROOT = Path("data/signals")


class ScenarioError(RuntimeError):
    """Erreur explicite lors de la préparation ou de la vérification d'un scénario."""


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    mutation: Callable[[dict[str, Any]], None] | None
    verifier: Callable[[dict[str, Any]], None]
    expected_score: float
    description: str


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dossier source introuvable : {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
        raise ValueError(f"Structure dossier.json invalide : {path}")
    return data


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def find_document(
    dossier: dict[str, Any],
    doc_type: str,
    month: str | None = None,
) -> dict[str, Any]:
    candidates = [
        document
        for document in dossier["documents"]
        if str(document.get("doc_type", "")).upper() == doc_type.upper()
    ]
    if month is not None:
        pattern = re.compile(rf"_M{re.escape(month[-1])}(?:_|$)", re.IGNORECASE)
        candidates = [
            document
            for document in candidates
            if pattern.search(str(document.get("doc_id", "")))
        ]
    if len(candidates) != 1:
        raise ScenarioError(
            f"Document attendu introuvable ou ambigu : type={doc_type}, "
            f"mois={month}, nombre={len(candidates)}"
        )
    return candidates[0]


def find_field(document: dict[str, Any], *field_names: str) -> dict[str, Any]:
    accepted = {name.lower() for name in field_names}
    matches = [
        field
        for field in document.get("fields", [])
        if str(field.get("field_name", "")).lower() in accepted
    ]
    if len(matches) != 1:
        raise ScenarioError(
            f"Champ attendu introuvable ou ambigu dans {document.get('doc_id')} : "
            f"{sorted(accepted)}, nombre={len(matches)}"
        )
    return matches[0]


def clone_with_new_id(source: dict[str, Any], new_dossier_id: str) -> dict[str, Any]:
    if not new_dossier_id.startswith(TEST_PREFIX):
        raise ScenarioError(f"Identifiant de test non sécurisé : {new_dossier_id}")

    dossier = copy.deepcopy(source)
    old_dossier_id = str(dossier.get("dossier_id", ""))
    dossier["dossier_id"] = new_dossier_id

    documents_by_type: dict[str, list[str]] = defaultdict(list)
    document_fields: dict[str, dict[str, dict[str, Any]]] = {}

    for document in dossier["documents"]:
        old_doc_id = str(document.get("doc_id", "DOC_UNKNOWN"))
        if old_dossier_id and old_doc_id.startswith(old_dossier_id):
            suffix = old_doc_id[len(old_dossier_id):]
            new_doc_id = f"{new_dossier_id}{suffix}"
        else:
            new_doc_id = f"{new_dossier_id}__{old_doc_id}"

        document["doc_id"] = new_doc_id
        document["dossier_id"] = new_dossier_id

        doc_type = str(document.get("doc_type", "UNKNOWN")).upper()
        documents_by_type[doc_type].append(new_doc_id)
        document_fields[new_doc_id] = {
            str(field.get("field_name")): {
                "raw_value": field.get("raw_value"),
                "normalized_value": field.get("normalized_value"),
                "confidence": field.get("confidence"),
                "validation_status": field.get("validation_status"),
            }
            for field in document.get("fields", [])
            if isinstance(field, dict) and field.get("field_name")
        }

    dossier["documents_by_type"] = dict(documents_by_type)
    dossier["document_fields"] = document_fields
    return dossier


def set_field_value(field: dict[str, Any], value: Any) -> None:
    field["raw_value"] = value
    field["normalized_value"] = value


def append_field(
    document: dict[str, Any],
    field_name: str,
    raw_value: Any,
    normalized_value: Any,
) -> None:
    existing = [
        field for field in document.get("fields", [])
        if str(field.get("field_name", "")).lower() == field_name.lower()
    ]
    if existing:
        set_field_value(existing[0], normalized_value)
        existing[0]["raw_value"] = raw_value
        return

    document.setdefault("fields", []).append({
        "field_name": field_name,
        "raw_value": raw_value,
        "normalized_value": normalized_value,
        "confidence": None,
        "validation_status": "VALID",
        "evidence": {"texts": [str(raw_value)]},
    })


def mutate_identifier(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").upper())
    if not text:
        raise ScenarioError("Impossible de modifier un identifiant vide.")

    last = text[-1]
    if last.isdigit():
        replacement = str((int(last) + 1) % 10)
    elif "A" <= last <= "Z":
        replacement = chr(((ord(last) - ord("A") + 1) % 26) + ord("A"))
    else:
        replacement = "1"
    return f"{text[:-1]}{replacement}"


def inject_cin_mismatch(dossier: dict[str, Any]) -> None:
    attestation = find_document(dossier, "ATTESTATION_TRAVAIL")
    field = find_field(attestation, "cin", "cin_numero")
    original = field.get("normalized_value") or field.get("raw_value")
    set_field_value(field, mutate_identifier(original))


def inject_rib_mismatch(dossier: dict[str, Any]) -> None:
    statement = find_document(dossier, "RELEVE_BANCAIRE", "M1")
    field = find_field(statement, "rib")
    original = field.get("normalized_value") or field.get("raw_value")
    set_field_value(field, mutate_identifier(original))


def inject_iban_mismatch(dossier: dict[str, Any]) -> None:
    rib_document = find_document(dossier, "RIB")
    iban_field = find_field(rib_document, "iban")
    original = iban_field.get("normalized_value") or iban_field.get("raw_value")
    fraud_value = mutate_identifier(original)

    statement = find_document(dossier, "RELEVE_BANCAIRE", "M1")
    append_field(statement, "iban", fraud_value, fraud_value)


def inject_bank_mismatch(dossier: dict[str, Any]) -> None:
    statement = find_document(dossier, "RELEVE_BANCAIRE", "M1")
    field = find_field(statement, "banque")
    set_field_value(field, "BANQUE_FICTIVE")


def _candidate_text_values(original: Any, *, address: bool) -> list[str]:
    normalizer = normalize_address if address else normalize_text
    base = normalizer(original)
    tokens = base.split()
    candidates: list[str] = []

    if not tokens:
        return ["VALEUR TOTALEMENT DIFFERENTE"]

    # Variantes par troncature du dernier mot : utiles pour les paliers 0,25 et 0,60.
    last = tokens[-1]
    for keep in range(len(last) - 1, 0, -1):
        candidates.append(" ".join([*tokens[:-1], last[:keep]]))

    # Variantes par remplacement d'un ou plusieurs mots.
    replacements = ["RUE", "CENTRE", "NOUVELLE", "ALI", "TEST", "AUTRE"]
    for index, token in enumerate(tokens):
        if token.isdigit():
            continue
        for replacement in replacements:
            changed = tokens.copy()
            changed[index] = replacement
            candidates.append(" ".join(changed))

    non_numeric_indexes = [i for i, token in enumerate(tokens) if not token.isdigit()]
    if len(non_numeric_indexes) >= 2:
        for first_index in non_numeric_indexes:
            for second_index in non_numeric_indexes:
                if first_index >= second_index:
                    continue
                changed = tokens.copy()
                changed[first_index] = "RUE"
                changed[second_index] = "NOUVELLE"
                candidates.append(" ".join(changed))

    # Variantes caractère par caractère.
    compact = base
    for index, char in enumerate(compact):
        if char == " ":
            continue
        replacement = "X" if char != "X" else "Z"
        candidates.append(f"{compact[:index]}{replacement}{compact[index + 1:]}")

    if address:
        numbers = [token for token in tokens if token.isdigit()]
        if numbers:
            changed = tokens.copy()
            number_index = changed.index(numbers[0])
            changed[number_index] = str(int(numbers[0]) + 794)
            candidates.append(" ".join(changed))
        candidates.extend([
            "999 AVENUE FRAUDULEUSE VILLE TEST",
            "999 RUE NOUVELLE AUTRE VILLE",
        ])
    else:
        candidates.extend([
            "PERSONNE TOTALEMENT DIFFERENTE",
            "SOCIETE FRAUDULEUSE",
            "DIRECTEUR GENERAL",
            "VALEUR TOTALEMENT DIFFERENTE",
        ])

    # Conserver l'ordre tout en supprimant les doublons et la valeur originale.
    unique: list[str] = []
    seen: set[str] = {base}
    for candidate in candidates:
        normalized = normalizer(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)
    return unique


def choose_text_candidate(
    original: Any,
    expected_score: float,
    *,
    address: bool = False,
) -> tuple[str, float]:
    for candidate in _candidate_text_values(original, address=address):
        similarity = text_similarity(original, candidate, address=address)
        score = text_anomaly_score(similarity)
        if abs(score - expected_score) <= 1e-9:
            return candidate, similarity
    raise ScenarioError(
        f"Impossible de générer automatiquement un texte avec score={expected_score} "
        f"à partir de {original!r}."
    )


def inject_text_difference(
    dossier: dict[str, Any],
    *,
    doc_type: str,
    field_name: str,
    expected_score: float,
    month: str | None = None,
    address: bool = False,
) -> None:
    document = find_document(dossier, doc_type, month)
    field = find_field(document, field_name)
    original = field.get("normalized_value") or field.get("raw_value")
    candidate, similarity = choose_text_candidate(
        original,
        expected_score,
        address=address,
    )
    set_field_value(field, candidate)
    field["test_expected_similarity"] = round(similarity, 6)
    field["test_expected_score"] = expected_score


def inject_name_difference(dossier: dict[str, Any], expected_score: float) -> None:
    inject_text_difference(
        dossier,
        doc_type="ATTESTATION_TRAVAIL",
        field_name="nom",
        expected_score=expected_score,
    )


def inject_address_difference(dossier: dict[str, Any], expected_score: float) -> None:
    inject_text_difference(
        dossier,
        doc_type="QUITTANCE",
        field_name="adresse",
        expected_score=expected_score,
        address=True,
    )


def inject_employer_difference(dossier: dict[str, Any], expected_score: float) -> None:
    inject_text_difference(
        dossier,
        doc_type="BULLETIN_SALAIRE",
        month="M1",
        field_name="employeur",
        expected_score=expected_score,
    )


def inject_job_difference(dossier: dict[str, Any], expected_score: float) -> None:
    inject_text_difference(
        dossier,
        doc_type="BULLETIN_SALAIRE",
        month="M1",
        field_name="poste",
        expected_score=expected_score,
    )


def _parse_date(value: Any) -> datetime:
    text = str(value or "").strip()
    for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue
    raise ScenarioError(f"Date non reconnue : {value!r}")


def inject_hire_date_difference(dossier: dict[str, Any], difference_days: int) -> None:
    attestation = find_document(dossier, "ATTESTATION_TRAVAIL")
    source_field = find_field(attestation, "date_embauche")
    original = source_field.get("normalized_value") or source_field.get("raw_value")
    target_date = _parse_date(original) + timedelta(days=difference_days)

    bulletin = find_document(dossier, "BULLETIN_SALAIRE", "M1")
    append_field(
        bulletin,
        "date_embauche",
        target_date.strftime("%d/%m/%Y"),
        target_date.strftime("%Y-%m-%d"),
    )


def _to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().replace("\u00a0", "").replace(" ", "")
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    return float(text)


def _is_salary_transaction(transaction: dict[str, Any]) -> bool:
    label = " ".join(str(transaction.get("libelle", "")).upper().split())
    direction = str(transaction.get("sens", "")).upper().strip()
    return direction == "C" and re.search(r"\b(?:SALAIRE|PAIE|PAYE)\b", label) is not None


def inject_salary_difference(
    dossier: dict[str, Any],
    relative_difference: float,
) -> None:
    if not 0 < relative_difference < 1:
        raise ScenarioError("La différence salariale doit être comprise entre 0 et 1.")

    bulletin = find_document(dossier, "BULLETIN_SALAIRE", "M1")
    net_field = find_field(bulletin, "net_a_payer")
    bulletin_amount = _to_float(net_field.get("normalized_value"))

    statement = find_document(dossier, "RELEVE_BANCAIRE", "M1")
    transactions_field = find_field(statement, "transactions")
    transactions = transactions_field.get("normalized_value")
    if not isinstance(transactions, list):
        raise ScenarioError("La liste normalisée des transactions M1 est absente.")

    salary_transactions = [
        transaction for transaction in transactions
        if isinstance(transaction, dict) and _is_salary_transaction(transaction)
    ]
    if len(salary_transactions) != 1:
        raise ScenarioError(
            "Le relevé M1 doit contenir exactement un virement salaire pour ce test; "
            f"nombre={len(salary_transactions)}"
        )

    fraud_amount = round(bulletin_amount * (1.0 - relative_difference), 2)
    salary_transactions[0]["montant"] = fraud_amount

    raw_transactions = transactions_field.get("raw_value")
    if isinstance(raw_transactions, list):
        raw_salary_transactions = [
            transaction for transaction in raw_transactions
            if isinstance(transaction, dict) and _is_salary_transaction(transaction)
        ]
        if len(raw_salary_transactions) == 1:
            raw_salary_transactions[0]["montant"] = fraud_amount


def inject_missing_cin_comparison(dossier: dict[str, Any]) -> None:
    attestation = find_document(dossier, "ATTESTATION_TRAVAIL")
    field = find_field(attestation, "cin", "cin_numero")
    field["raw_value"] = None
    field["normalized_value"] = None


def combine_mutations(*mutations: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
    def combined(dossier: dict[str, Any]) -> None:
        for mutation in mutations:
            mutation(dossier)
    return combined


def get_check(result: dict[str, Any], check_id: str) -> dict[str, Any]:
    for check in result.get("checks", []):
        if check.get("check_id") == check_id:
            return check
    raise ScenarioError(f"Contrôle absent du résultat : {check_id}")


def assert_score(actual: Any, expected: float, label: str) -> None:
    if actual is None or abs(float(actual) - expected) > 1e-9:
        raise AssertionError(f"{label} : attendu={expected}, obtenu={actual}")


def make_score_verifier(
    check_id: str,
    expected_check_score: float,
    *,
    expected_global_score: float | None = None,
    critical: bool | None = None,
    require_suspect: bool = True,
    expected_relative_difference: float | None = None,
    expected_difference_days: int | None = None,
) -> Callable[[dict[str, Any]], None]:
    global_score = (
        expected_check_score
        if expected_global_score is None
        else expected_global_score
    )

    def verifier(result: dict[str, Any]) -> None:
        assert_score(result.get("score"), global_score, "score global")
        check = get_check(result, check_id)
        if check.get("available") is not True:
            raise AssertionError(f"Le contrôle {check_id} doit être disponible.")
        assert_score(check.get("score"), expected_check_score, f"score {check_id}")

        if critical is not None and check.get("critical") is not critical:
            raise AssertionError(
                f"{check_id}.critical : attendu={critical}, obtenu={check.get('critical')}"
            )
        if require_suspect and expected_check_score > 0 and not check.get("suspect_fields"):
            raise AssertionError(f"{check_id} doit conserver le champ suspect.")
        if expected_check_score == 0 and check.get("suspect_fields"):
            raise AssertionError(f"{check_id} ne doit pas contenir de champ suspect.")

        if expected_relative_difference is not None:
            actual = check.get("relative_difference")
            if actual is None or abs(float(actual) - expected_relative_difference) > 1e-4:
                raise AssertionError(
                    f"{check_id}.relative_difference : "
                    f"attendu={expected_relative_difference}, obtenu={actual}"
                )
        if expected_difference_days is not None:
            comparisons = check.get("comparisons", [])
            if not comparisons:
                raise AssertionError(f"{check_id} doit contenir une comparaison de date.")
            actual = comparisons[0].get("difference_days")
            if actual != expected_difference_days:
                raise AssertionError(
                    f"{check_id}.difference_days : "
                    f"attendu={expected_difference_days}, obtenu={actual}"
                )

    return verifier


def verify_coherent(result: dict[str, Any]) -> None:
    assert_score(result.get("score"), 0.0, "score global")
    if result.get("summary", {}).get("anomalies_count") != 0:
        raise AssertionError("Le scénario cohérent ne doit produire aucune anomalie.")


def verify_missing_cin(result: dict[str, Any]) -> None:
    assert_score(result.get("score"), 0.0, "score global")
    check = get_check(result, "CIN")
    if check.get("available") is not False:
        raise AssertionError("CIN doit être indisponible lorsqu'une seule valeur subsiste.")
    if check.get("score") is not None:
        raise AssertionError("Un contrôle indisponible doit avoir score=null, pas score=0.")
    if check.get("status") != "UNAVAILABLE":
        raise AssertionError("Le statut CIN attendu est UNAVAILABLE.")


def verify_combined(
    result: dict[str, Any],
    *,
    expected_global: float,
    expected_checks: dict[str, float],
) -> None:
    assert_score(result.get("score"), expected_global, "score global max")
    for check_id, expected in expected_checks.items():
        assert_score(get_check(result, check_id).get("score"), expected, f"score {check_id}")


SCENARIOS: list[Scenario] = [
    Scenario(
        "TEST_INTRA_FULL_COHERENT",
        None,
        verify_coherent,
        0.0,
        "Dossier cohérent sans modification.",
    ),
    Scenario(
        "TEST_INTRA_FULL_CIN",
        inject_cin_mismatch,
        make_score_verifier("CIN", 1.0, critical=True),
        1.0,
        "CIN différente : contrôle exact critique.",
    ),
    Scenario(
        "TEST_INTRA_FULL_RIB",
        inject_rib_mismatch,
        make_score_verifier("RIB", 1.0, critical=True),
        1.0,
        "RIB différent : contrôle exact critique.",
    ),
    Scenario(
        "TEST_INTRA_FULL_IBAN",
        inject_iban_mismatch,
        make_score_verifier("IBAN", 1.0, critical=True),
        1.0,
        "IBAN différent après ajout contrôlé d'une deuxième valeur.",
    ),
    Scenario(
        "TEST_INTRA_FULL_BANQUE",
        inject_bank_mismatch,
        make_score_verifier("BANQUE", 1.0, critical=True),
        1.0,
        "Banque différente : contrôle exact.",
    ),
    Scenario(
        "TEST_INTRA_FULL_NOM_025",
        lambda dossier: inject_name_difference(dossier, 0.25),
        make_score_verifier("NOM_COMPLET", 0.25, critical=True),
        0.25,
        "Nom avec similarité dans le palier 85-95 %.",
    ),
    Scenario(
        "TEST_INTRA_FULL_NOM_060",
        lambda dossier: inject_name_difference(dossier, 0.60),
        make_score_verifier("NOM_COMPLET", 0.60, critical=True),
        0.60,
        "Nom avec similarité dans le palier 70-85 %.",
    ),
    Scenario(
        "TEST_INTRA_FULL_NOM_100",
        lambda dossier: inject_name_difference(dossier, 1.0),
        make_score_verifier("NOM_COMPLET", 1.0, critical=True),
        1.0,
        "Nom fortement différent.",
    ),
    Scenario(
        "TEST_INTRA_FULL_ADRESSE_025",
        lambda dossier: inject_address_difference(dossier, 0.25),
        make_score_verifier("ADRESSE", 0.25, critical=False),
        0.25,
        "Adresse avec anomalie légère.",
    ),
    Scenario(
        "TEST_INTRA_FULL_ADRESSE_060",
        lambda dossier: inject_address_difference(dossier, 0.60),
        make_score_verifier("ADRESSE", 0.60, critical=False),
        0.60,
        "Adresse avec anomalie moyenne.",
    ),
    Scenario(
        "TEST_INTRA_FULL_ADRESSE_100",
        lambda dossier: inject_address_difference(dossier, 1.0),
        make_score_verifier("ADRESSE", 1.0, critical=False),
        1.0,
        "Adresse fortement différente.",
    ),
    Scenario(
        "TEST_INTRA_FULL_EMPLOYEUR_025",
        lambda dossier: inject_employer_difference(dossier, 0.25),
        make_score_verifier("EMPLOYEUR", 0.25, critical=False),
        0.25,
        "Employeur légèrement différent.",
    ),
    Scenario(
        "TEST_INTRA_FULL_EMPLOYEUR_100",
        lambda dossier: inject_employer_difference(dossier, 1.0),
        make_score_verifier("EMPLOYEUR", 1.0, critical=False),
        1.0,
        "Employeur fortement différent.",
    ),
    Scenario(
        "TEST_INTRA_FULL_POSTE_060",
        lambda dossier: inject_job_difference(dossier, 0.60),
        make_score_verifier("POSTE", 0.60, critical=False),
        0.60,
        "Poste avec anomalie moyenne.",
    ),
    Scenario(
        "TEST_INTRA_FULL_POSTE_100",
        lambda dossier: inject_job_difference(dossier, 1.0),
        make_score_verifier("POSTE", 1.0, critical=False),
        1.0,
        "Poste fortement différent.",
    ),
    Scenario(
        "TEST_INTRA_FULL_DATE_000",
        lambda dossier: inject_hire_date_difference(dossier, 0),
        make_score_verifier(
            "DATE_EMBAUCHE",
            0.0,
            critical=False,
            require_suspect=False,
            expected_difference_days=0,
        ),
        0.0,
        "Date d'embauche identique dans deux documents.",
    ),
    Scenario(
        "TEST_INTRA_FULL_DATE_025",
        lambda dossier: inject_hire_date_difference(dossier, 15),
        make_score_verifier(
            "DATE_EMBAUCHE",
            0.25,
            critical=False,
            expected_difference_days=15,
        ),
        0.25,
        "Écart de date de 15 jours.",
    ),
    Scenario(
        "TEST_INTRA_FULL_DATE_060",
        lambda dossier: inject_hire_date_difference(dossier, 120),
        make_score_verifier(
            "DATE_EMBAUCHE",
            0.60,
            critical=False,
            expected_difference_days=120,
        ),
        0.60,
        "Écart de date de 120 jours.",
    ),
    Scenario(
        "TEST_INTRA_FULL_DATE_100",
        lambda dossier: inject_hire_date_difference(dossier, 400),
        make_score_verifier(
            "DATE_EMBAUCHE",
            1.0,
            critical=False,
            expected_difference_days=400,
        ),
        1.0,
        "Écart de date supérieur à un an.",
    ),
    Scenario(
        "TEST_INTRA_FULL_SALAIRE_03",
        lambda dossier: inject_salary_difference(dossier, 0.03),
        make_score_verifier(
            "SALAIRE_M1",
            0.25,
            expected_relative_difference=0.03,
        ),
        0.25,
        "Écart salarial de 3 %.",
    ),
    Scenario(
        "TEST_INTRA_FULL_SALAIRE_08",
        lambda dossier: inject_salary_difference(dossier, 0.08),
        make_score_verifier(
            "SALAIRE_M1",
            0.50,
            expected_relative_difference=0.08,
        ),
        0.50,
        "Écart salarial de 8 %.",
    ),
    Scenario(
        "TEST_INTRA_FULL_SALAIRE_15",
        lambda dossier: inject_salary_difference(dossier, 0.15),
        make_score_verifier(
            "SALAIRE_M1",
            0.75,
            expected_relative_difference=0.15,
        ),
        0.75,
        "Écart salarial de 15 %.",
    ),
    Scenario(
        "TEST_INTRA_FULL_SALAIRE_25",
        lambda dossier: inject_salary_difference(dossier, 0.25),
        make_score_verifier(
            "SALAIRE_M1",
            1.0,
            expected_relative_difference=0.25,
        ),
        1.0,
        "Écart salarial de 25 %.",
    ),
    Scenario(
        "TEST_INTRA_FULL_CIN_ABSENTE",
        inject_missing_cin_comparison,
        verify_missing_cin,
        0.0,
        "Une seule CIN exploitable : contrôle indisponible, score null.",
    ),
    Scenario(
        "TEST_INTRA_FULL_ADRESSE025_SALAIRE08",
        combine_mutations(
            lambda dossier: inject_address_difference(dossier, 0.25),
            lambda dossier: inject_salary_difference(dossier, 0.08),
        ),
        lambda result: verify_combined(
            result,
            expected_global=0.50,
            expected_checks={"ADRESSE": 0.25, "SALAIRE_M1": 0.50},
        ),
        0.50,
        "Deux anomalies non critiques : max(0,25 ; 0,50) = 0,50.",
    ),
    Scenario(
        "TEST_INTRA_FULL_RIB_SALAIRE15",
        combine_mutations(
            inject_rib_mismatch,
            lambda dossier: inject_salary_difference(dossier, 0.15),
        ),
        lambda result: verify_combined(
            result,
            expected_global=1.0,
            expected_checks={"RIB": 1.0, "SALAIRE_M1": 0.75},
        ),
        1.0,
        "Anomalie critique et salaire : max(1 ; 0,75) = 1.",
    ),
]


def cleanup_test_dossier(client: Neo4jClient, dossier_id: str) -> None:
    if not dossier_id.startswith(TEST_PREFIX):
        raise ScenarioError(f"Suppression refusée pour un dossier non-test : {dossier_id}")
    _delete_existing_dossier(client, dossier_id)


def run_scenarios(
    source_path: Path,
    work_root: Path,
    signal_root: Path,
    keep_neo4j: bool,
) -> dict[str, Any]:
    source = load_json(source_path)
    client = Neo4jClient()
    created_ids: list[str] = []
    report_rows: list[dict[str, Any]] = []

    try:
        client.verifier_connexion()

        for scenario in SCENARIOS:
            scenario_dossier = clone_with_new_id(source, scenario.scenario_id)
            if scenario.mutation is not None:
                scenario.mutation(scenario_dossier)

            scenario_json = work_root / scenario.scenario_id / "dossier.json"
            save_json(scenario_dossier, scenario_json)

            load_result = load_dossier_json(scenario_json, client=client)
            created_ids.append(scenario.scenario_id)
            result = run_intra_dossier_signal(
                scenario.scenario_id,
                output_root=signal_root,
                client=client,
            )

            try:
                scenario.verifier(result)
                status = "OK"
                error = None
                print(
                    f"[OK] {scenario.scenario_id} | "
                    f"attendu={scenario.expected_score} | obtenu={result.get('score')} | "
                    f"anomalies={result.get('summary', {}).get('anomalies_count')}"
                )
            except Exception as exc:
                status = "ERROR"
                error = str(exc)
                print(f"[ERREUR] {scenario.scenario_id} | {error}")

            report_rows.append({
                "scenario_id": scenario.scenario_id,
                "description": scenario.description,
                "status": status,
                "error": error,
                "expected_score": scenario.expected_score,
                "observed_score": result.get("score"),
                "anomalies_count": result.get("summary", {}).get("anomalies_count"),
                "checks_unavailable": result.get("summary", {}).get("checks_unavailable"),
                "dossier_json": str(scenario_json),
                "signal_json": str(
                    signal_root / scenario.scenario_id / "intra_dossier.json"
                ),
                "neo4j_counts": {
                    "documents": load_result.get("documents"),
                    "fields": load_result.get("fields"),
                    "transactions": load_result.get("transactions"),
                },
            })

        failed = [row for row in report_rows if row["status"] != "OK"]
        report = {
            "source_dossier": str(source_path),
            "scenarios_total": len(report_rows),
            "scenarios_ok": len(report_rows) - len(failed),
            "scenarios_failed": len(failed),
            "neo4j_test_dossiers_kept": keep_neo4j,
            "aggregation_tested": "MAX_AVAILABLE_SUBSCORES",
            "ocr_confidence_used": False,
            "results": report_rows,
        }
        save_json(report, work_root / "test_report.json")

        if failed:
            raise AssertionError(
                f"{len(failed)} scénario(s) ont échoué. "
                f"Voir {work_root / 'test_report.json'}"
            )
        return report

    finally:
        if not keep_neo4j:
            for dossier_id in reversed(created_ids):
                cleanup_test_dossier(client, dossier_id)
        client.fermer()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Teste les paliers complets du signal intra-dossier sur des copies "
            "contrôlées et sécurisées d'un dossier légitime."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--signal-root", type=Path, default=DEFAULT_SIGNAL_ROOT)
    parser.add_argument(
        "--keep-neo4j",
        action="store_true",
        help="Conserve les dossiers TEST_INTRA_FULL_* dans Neo4j après les tests.",
    )
    args = parser.parse_args()

    report = run_scenarios(
        source_path=args.source,
        work_root=args.work_root,
        signal_root=args.signal_root,
        keep_neo4j=args.keep_neo4j,
    )

    print("\n===== RÉSUMÉ TESTS INTRA-DOSSIER COMPLETS =====")
    print(f"Scénarios : {report['scenarios_total']}")
    print(f"Réussis : {report['scenarios_ok']}")
    print(f"Échoués : {report['scenarios_failed']}")
    print(f"Rapport : {args.work_root / 'test_report.json'}")
    if args.keep_neo4j:
        print("Les dossiers TEST_INTRA_FULL_* sont conservés dans Neo4j.")
    else:
        print("Les dossiers TEST_INTRA_FULL_* ont été supprimés de Neo4j après le test.")


if __name__ == "__main__":
    main()
