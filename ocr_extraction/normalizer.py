from __future__ import annotations

import argparse
import copy
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


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


def remove_accents(value: str) -> str:
    value = unicodedata.normalize("NFD", value)
    return "".join(character for character in value if unicodedata.category(character) != "Mn")


def clean_spaces(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).replace("\u00a0", " ").strip().split())
    return text or None


def normalize_identifier(value: Any) -> str | None:
    text = clean_spaces(value)
    if not text:
        return None
    return re.sub(r"[^A-Z0-9]", "", remove_accents(text).upper()) or None


def normalize_rib(value: Any) -> str | None:
    text = clean_spaces(value)
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    return digits or None


def normalize_person_name(value: Any) -> str | None:
    text = clean_spaces(value)
    if not text:
        return None
    text = re.sub(r"[^A-Z'\-\s]", " ", remove_accents(text).upper())
    return " ".join(text.split()) or None


def normalize_free_text(value: Any) -> str | None:
    text = clean_spaces(value)
    return remove_accents(text).upper() if text else None

BANK_ALIASES = {
    "ATTIJARIWAFABANK": "ATTIJARIWAFA_BANK",
    "CIHBANK": "CIH_BANK",
    "SOCIETEGENERALEMAROC": "SOCIETE_GENERALE_MAROC",
}


def normalize_bank(value: Any) -> str | None:
    text = clean_spaces(value)
    if not text:
        return None

    key = re.sub(
        r"[^A-Z0-9]",
        "",
        remove_accents(text).upper(),
    )

    if "ATTIJARI" in key:
        return "ATTIJARIWAFA_BANK"

    if "CIH" in key:
        return "CIH_BANK"

    if "SOCIETEGENERALE" in key:
        return "SOCIETE_GENERALE_MAROC"

    return normalize_free_text(text)

def normalize_address(value: Any) -> str | None:
    text = clean_spaces(value)
    if not text:
        return None
    text = re.sub(r"[^A-Z0-9'\-./,\s]", " ", remove_accents(text).upper())
    return " ".join(text.split()) or None


def normalize_date(value: Any) -> str | None:
    text = clean_spaces(value)
    if not text:
        return None
    text = re.sub(r"\s+", "", text.replace("_", ""))
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    match = re.search(r"\b(\d{2})[\/\-.](\d{2})[\/\-.](\d{4})\b", text)
    if not match:
        return None
    try:
        return datetime.strptime("/".join(match.groups()), "%d/%m/%Y").date().isoformat()
    except ValueError:
        return None


def normalize_partial_date(value: Any) -> str | None:
    text = clean_spaces(value)
    if not text:
        return None
    text = text.replace(" ", "")
    if re.fullmatch(r"\d{2}/\d{2}", text):
        return text
    return normalize_date(text)


def normalize_period(value: Any) -> str | None:
    text = clean_spaces(value)
    if not text:
        return None
    match = re.fullmatch(r"(\d{1,2})[\-/](\d{4})", text.replace(" ", ""))
    if not match:
        return None
    month, year = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        return None
    return f"{year:04d}-{month:02d}"


def normalize_amount(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 2)
    text = clean_spaces(value)
    if not text:
        return None
    text = re.sub(r"(?i)\b(?:MAD|DH|DHS|DIRHAMS?|CREDIT|DEBIT)\b", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[^\d,.\-\s]", "", text).strip().replace(" ", "")
    if not text:
        return None

    if "," in text and "." in text:
        if text.rfind(".") > text.rfind(","):
            text = text.replace(",", "")
        else:
            text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        parts = text.split(",")
        text = "".join(parts[:-1]) + "." + parts[-1] if len(parts[-1]) == 2 else text.replace(",", "")
    elif text.count(".") > 1:
        parts = text.split(".")
        text = "".join(parts[:-1]) + "." + parts[-1]

    try:
        return round(float(text), 2)
    except ValueError:
        return None


def normalize_percentage(value: Any) -> float | None:
    return normalize_amount(value)


def normalize_boolean(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = remove_accents(str(value)).upper().strip()
    if text in {"TRUE", "VRAI", "OUI", "1", "PRESENT"}:
        return True
    if text in {"FALSE", "FAUX", "NON", "0", "ABSENT"}:
        return False
    return None


def normalize_sex(value: Any) -> str | None:
    text = clean_spaces(value)
    if not text:
        return None
    key = re.sub(r"[^A-Z]", "", remove_accents(text).upper())
    if key in {"F", "FEMME", "FEMININ", "FEMALE"}:
        return "F"
    if key in {"M", "H", "HOMME", "MASCULIN", "MALE"}:
        return "M"
    return None


def normalize_transactions(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        normalized.append({
            "date": normalize_partial_date(item.get("date")),
            "libelle": normalize_free_text(item.get("libelle")),
            "montant": normalize_amount(item.get("montant")),
            "sens": normalize_free_text(item.get("sens")),
            "ocr_confidence": item.get("ocr_confidence"),
        })
    return normalized


AMOUNT_FIELDS = {
    "net_a_payer",
    "salaire_mensuel",
    "salaire_base_mensuel",
    "salaire_brut",
    "cotisation_cnss_base",
    "cotisation_cnss_montant",
    "cotisation_amo_base",
    "cotisation_amo_montant",
    "salaire_net_imposable",
    "solde_ouverture",
    "solde_cloture",
    "total_credits",
    "total_debits",
}
PERCENT_FIELDS = {"cotisation_cnss_taux", "cotisation_amo_taux"}
DATE_FIELDS = {
    "date_naissance",
    "date_embauche",
    "date_facture",
    "date_solde_ouverture",
    "date_solde_cloture",
}
NAME_FIELDS = {"nom", "prenom", "nom_abonne", "titulaire", "nom_employe", "prenom_employe"}

FREE_TEXT_FIELDS = {
    "lieu_naissance",
    "nationalite",
    "ville",
    "employeur",
    "poste",
    "anciennete",
}


def normalizer_for(field_name: str) -> Callable[[Any], Any]:
    if field_name in {"cin_numero", "cin", "iban"}:
        return normalize_identifier
    if field_name == "rib":
        return normalize_rib
    if field_name in NAME_FIELDS:
        return normalize_person_name
    if field_name == "adresse":
        return normalize_address
    if field_name in DATE_FIELDS:
        return normalize_date
    if field_name == "periode":
        return normalize_period
    if field_name in AMOUNT_FIELDS:
        return normalize_amount
    if field_name in PERCENT_FIELDS:
        return normalize_percentage
    if field_name == "cachet_present":
        return normalize_boolean
    if field_name == "sexe":
        return normalize_sex
    if field_name == "transactions":
        return normalize_transactions
    if field_name == "banque":
        return normalize_bank
    if field_name in FREE_TEXT_FIELDS:
        return normalize_free_text
    return normalize_free_text


def normalize_field(field: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(field)
    name = str(field.get("field_name", ""))
    raw = field.get("raw_value")

    if field.get("status") == "ABSENT_DU_TEMPLATE":
        result["normalized_value"] = None
        result["normalization_status"] = "ABSENT_DU_TEMPLATE"
        return result

    if raw is None:
        result["normalized_value"] = None
        result["normalization_status"] = "MISSING"
        return result

    value = normalizer_for(name)(raw)
    result["normalized_value"] = value
    result["normalization_status"] = "NORMALIZED" if value is not None else "NORMALIZATION_FAILED"
    return result


def normalize_document(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    fields = document.get("fields", [])
    if not isinstance(fields, list):
        raise ValueError("Le document doit contenir une liste 'fields'.")

    normalized_fields = [normalize_field(field) for field in fields if isinstance(field, dict)]
    result["fields"] = normalized_fields
    result["normalization"] = {
        "scope": "FORMAT_ONLY_RAW_VALUE_PRESERVED",
        "status": "TO_REVIEW" if any(
            field["normalization_status"] == "NORMALIZATION_FAILED"
            for field in normalized_fields
        ) else "OK",
        "normalized_fields": sum(field["normalization_status"] == "NORMALIZED" for field in normalized_fields),
        "missing_fields": [field["field_name"] for field in normalized_fields if field["normalization_status"] == "MISSING"],
        "absent_from_template_fields": [
            field["field_name"] for field in normalized_fields
            if field["normalization_status"] == "ABSENT_DU_TEMPLATE"
        ],
        "failed_fields": [
            field["field_name"] for field in normalized_fields
            if field["normalization_status"] == "NORMALIZATION_FAILED"
        ],
        "raw_values_preserved": True,
    }
    return result


def process_file(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    result = normalize_document(load_json(input_path))
    save_json(result, output_path)
    return result


def process_folder(input_dir: str | Path, output_dir: str | Path) -> dict[str, int]:
    source, destination = Path(input_dir), Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    processed = errors = 0
    for input_path in sorted(source.glob("*.json")):
        try:
            process_file(input_path, destination / input_path.name)
            processed += 1
            print(f"[NORMALISÉ] {input_path.name}")
        except Exception as error:
            errors += 1
            print(f"[ERREUR] {input_path.name} : {error}")
    return {"processed": processed, "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalisation de format ; raw_value reste intact.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    summary = process_folder(input_path, output_path) if input_path.is_dir() else {"processed": 1, "errors": 0} if process_file(input_path, output_path) else {}
    print("\n===== RÉSUMÉ NORMALISATION =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
