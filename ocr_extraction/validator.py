from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

LOW_CONFIDENCE_THRESHOLD = 0.70

REQUIRED_FIELDS_BY_DOC_TYPE: dict[str, set[str]] = {
    "CIN": {"cin_numero", "nom", "prenom", "date_naissance", "adresse"},
    "QUITTANCE": {"nom_abonne", "adresse", "ville"},
    "RIB": {"banque", "titulaire", "rib"},
    "RELEVE_BANCAIRE": {
        "banque", "titulaire", "adresse", "rib",
        "solde_ouverture", "solde_cloture", "total_credits", "total_debits", "transactions",
    },
    "BULLETIN_SALAIRE": {
        "nom_employe", "prenom_employe", "employeur", "poste", "periode", "salaire_brut", "net_a_payer",
    },
    "ATTESTATION_TRAVAIL": {"nom", "cin", "employeur", "poste", "date_embauche"},
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Le JSON doit contenir un objet.")
    return data


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return output


def validate_cin_number(value: Any) -> tuple[bool, str]:
    if value is None or not re.fullmatch(r"[A-Z]{1,2}\d{5,8}", str(value)):
        return False, "Format CIN non exploitable."
    return True, ""


def validate_identifier(value: Any) -> tuple[bool, str]:
    if value is None or not re.fullmatch(r"[A-Z0-9]+", str(value)):
        return False, "Identifiant non exploitable."
    return True, ""


def validate_rib(value: Any) -> tuple[bool, str]:
    if value is None or not str(value).isdigit():
        return False, "RIB non numérique après normalisation."
    return True, ""


def validate_person_name(value: Any) -> tuple[bool, str]:
    text = str(value or "").strip()
    if len(text) < 2:
        return False, "Nom absent ou trop court."
    if any(character.isdigit() for character in text):
        return False, "Nom contenant un chiffre."
    return True, ""


def validate_text(value: Any, minimum_length: int = 2) -> tuple[bool, str]:
    if value is None or len(str(value).strip()) < minimum_length:
        return False, f"Valeur absente ou plus courte que {minimum_length} caractères."
    return True, ""


def validate_iso_date(value: Any) -> tuple[bool, str]:
    if value is None:
        return False, "Date absente."
    try:
        datetime.strptime(str(value), "%Y-%m-%d")
    except ValueError:
        return False, "Date non normalisée au format YYYY-MM-DD."
    return True, ""


def validate_period(value: Any) -> tuple[bool, str]:
    if value is None or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", str(value)):
        return False, "Période non normalisée au format YYYY-MM."
    return True, ""


def validate_numeric(value: Any) -> tuple[bool, str]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False, "Valeur numérique attendue."
    return True, ""


def validate_percentage(value: Any) -> tuple[bool, str]:
    return validate_numeric(value)


def validate_boolean(value: Any) -> tuple[bool, str]:
    if value is None or isinstance(value, bool):
        return True, ""
    return False, "Valeur booléenne attendue."


def validate_sex(value: Any) -> tuple[bool, str]:
    if value is None or value in {"M", "F"}:
        return True, ""
    return False, "Sexe non normalisé."


def validate_transactions(value: Any) -> tuple[bool, str]:
    if not isinstance(value, list) or not value:
        return False, "Liste de transactions absente ou vide."
    required = {"date", "libelle", "montant", "sens"}
    for index, transaction in enumerate(value, start=1):
        if not isinstance(transaction, dict):
            return False, f"Transaction {index} non structurée."
        if not required.issubset(transaction):
            return False, f"Transaction {index} incomplète."
        if transaction.get("montant") is None:
            return False, f"Montant non normalisé pour la transaction {index}."
        if transaction.get("sens") not in {"D", "C"}:
            return False, f"Sens invalide pour la transaction {index}."
    return True, ""


AMOUNT_FIELDS = {
    "net_a_payer", "salaire_mensuel", "salaire_base_mensuel", "salaire_brut",
    "cotisation_cnss_base", "cotisation_cnss_montant", "cotisation_amo_base",
    "cotisation_amo_montant", "salaire_net_imposable", "solde_ouverture",
    "solde_cloture", "total_credits", "total_debits",
}
PERCENT_FIELDS = {"cotisation_cnss_taux", "cotisation_amo_taux"}
DATE_FIELDS = {"date_naissance", "date_embauche", "date_facture", "date_solde_ouverture", "date_solde_cloture"}
NAME_FIELDS = {"nom", "prenom", "nom_abonne", "titulaire", "nom_employe", "prenom_employe"}
TEXT_FIELDS = {"adresse", "ville", "banque", "employeur", "poste", "anciennete", "lieu_naissance", "nationalite"}


def validator_for(field_name: str) -> Callable[[Any], tuple[bool, str]] | None:
    if field_name in {"cin_numero", "cin"}:
        return validate_cin_number
    if field_name == "rib":
        return validate_rib
    if field_name == "iban":
        return validate_identifier
    if field_name in NAME_FIELDS:
        return validate_person_name
    if field_name in DATE_FIELDS:
        return validate_iso_date
    if field_name == "periode":
        return validate_period
    if field_name in AMOUNT_FIELDS:
        return validate_numeric
    if field_name in PERCENT_FIELDS:
        return validate_percentage
    if field_name == "transactions":
        return validate_transactions
    if field_name == "cachet_present":
        return validate_boolean
    if field_name == "sexe":
        return validate_sex
    if field_name in TEXT_FIELDS:
        return lambda value: validate_text(value, 2)
    return None


def validate_field(
    field: dict[str, Any],
    required_fields: set[str],
    low_confidence_threshold: float,
) -> dict[str, Any]:
    result = copy.deepcopy(field)
    name = str(field.get("field_name", ""))
    value = field.get("normalized_value")
    confidence = float(field.get("confidence", 0.0) or 0.0)
    messages: list[str] = []

    if field.get("status") == "ABSENT_DU_TEMPLATE":
        result["validation_status"] = "ABSENT_DU_TEMPLATE"
        result["validation_messages"] = []
        return result

    if value is None:
        if name in required_fields:
            status = "MISSING" if field.get("raw_value") is None else "INVALID_FORMAT"
            messages.append("Champ obligatoire absent." if status == "MISSING" else "Normalisation impossible à partir de raw_value.")
        elif field.get("raw_value") is not None:
            status = "INVALID_FORMAT"
            messages.append("Normalisation impossible à partir de raw_value.")
        else:
            status = "OPTIONAL_MISSING"
        result["validation_status"] = status
        result["validation_messages"] = messages
        return result

    validator = validator_for(name)
    if validator is not None:
        valid, message = validator(value)
        if not valid:
            result["validation_status"] = "INVALID_FORMAT"
            result["validation_messages"] = [message]
            return result

    if field.get("status") == "TO_REVIEW":
        status = "TO_REVIEW"
        messages.append("Champ marqué à vérifier pendant l'extraction.")
    elif confidence < low_confidence_threshold:
        status = "LOW_CONFIDENCE"
        messages.append(f"Confiance OCR {confidence:.2f} inférieure au seuil {low_confidence_threshold:.2f}.")
    else:
        status = "VALID"

    result["validation_status"] = status
    result["validation_messages"] = messages
    return result


def validate_document(
    document: dict[str, Any],
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    doc_type = str(document.get("doc_type", ""))
    required = REQUIRED_FIELDS_BY_DOC_TYPE.get(doc_type)
    result = copy.deepcopy(document)
    if required is None:
        result["validation_status"] = "NOT_IMPLEMENTED"
        result["validation"] = {
            "scope": "DATA_QUALITY_ONLY",
            "status": "NOT_IMPLEMENTED",
            "message": f"Aucune validation de qualité pour {doc_type}.",
        }
        return result

    fields = document.get("fields", [])
    if not isinstance(fields, list):
        raise ValueError("Le document doit contenir une liste 'fields'.")

    validated = [
        validate_field(field, required, low_confidence_threshold)
        for field in fields
        if isinstance(field, dict)
    ]
    required_missing = [
        field["field_name"] for field in validated
        if field["field_name"] in required and field["validation_status"] == "MISSING"
    ]
    required_invalid = [
        field["field_name"] for field in validated
        if field["field_name"] in required and field["validation_status"] == "INVALID_FORMAT"
    ]
    required_low = [
        field["field_name"] for field in validated
        if field["field_name"] in required and field["validation_status"] == "LOW_CONFIDENCE"
    ]
    required_review = [
        field["field_name"] for field in validated
        if field["field_name"] in required and field["validation_status"] == "TO_REVIEW"
    ]
    optional_invalid = [
        field["field_name"] for field in validated
        if field["field_name"] not in required and field["validation_status"] == "INVALID_FORMAT"
    ]

    if required_missing:
        status = "INCOMPLETE"
    elif required_invalid:
        status = "INVALID"
    elif required_low or required_review or optional_invalid:
        status = "TO_REVIEW"
    else:
        status = "VALID"

    result["fields"] = validated
    result["validation"] = {
        "scope": "DATA_QUALITY_ONLY_NOT_FRAUD_DECISION",
        "status": status,
        "low_confidence_threshold": low_confidence_threshold,
        "required_fields": sorted(required),
        "missing_required_fields": required_missing,
        "invalid_required_fields": required_invalid,
        "low_confidence_required_fields": required_low,
        "required_fields_to_review": required_review,
        "invalid_optional_fields": optional_invalid,
        "raw_values_preserved": True,
    }
    result["validation_status"] = status
    return result


def process_file(input_path: str | Path, output_path: str | Path, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> dict[str, Any]:
    result = validate_document(load_json(input_path), threshold)
    save_json(result, output_path)
    return result


def process_folder(input_dir: str | Path, output_dir: str | Path, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> dict[str, int]:
    source, destination = Path(input_dir), Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    counters = {"processed": 0, "valid": 0, "to_review": 0, "incomplete": 0, "invalid": 0, "errors": 0}
    for input_path in sorted(source.glob("*.json")):
        try:
            result = process_file(input_path, destination / input_path.name, threshold)
            counters["processed"] += 1
            key = result.get("validation_status", "").lower()
            if key in counters:
                counters[key] += 1
            print(f"[VALIDÉ QUALITÉ] {input_path.name} : {result.get('validation_status')}")
        except Exception as error:
            counters["errors"] += 1
            print(f"[ERREUR] {input_path.name} : {error}")
    return counters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation de qualité des données, sans détection de fraude.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--low-confidence-threshold", type=float, default=LOW_CONFIDENCE_THRESHOLD)
    args = parser.parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    summary = process_folder(input_path, output_path, args.low_confidence_threshold) if input_path.is_dir() else {"processed": 1, "errors": 0} if process_file(input_path, output_path, args.low_confidence_threshold) else {}
    print("\n===== RÉSUMÉ VALIDATION QUALITÉ =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
