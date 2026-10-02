from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    choose_first_result,
    find_regex,
    find_value_near_label,
    get_blocks,
    load_json,
    make_field,
)

REQUIRED_FIELDS = {"cin_numero", "nom", "prenom", "date_naissance", "adresse"}


def _field(field_name: str, result: dict[str, Any], status: str | None = None) -> dict[str, Any]:
    if result.get("value") is None and status is None:
        status = "MISSING"
    return make_field(
        field_name,
        result.get("value"),
        result.get("blocks", []),
        result.get("method", "not_found"),
        status,
    )


def extract_cin(
    ocr_json: dict[str, Any],
    classification: dict[str, Any],
    dossier_id: str,
) -> dict[str, Any]:
    if classification.get("doc_type") != "CIN" and classification.get("template_id") != "CIN_V1":
        raise ValueError("extract_cin a reçu un document qui n'est pas une CIN.")

    blocks = get_blocks(ocr_json)
    cin_numero = choose_first_result(
        find_value_near_label(blocks, ["numéro d'identifiant", "numero d'identifiant", "n° d'identifiant"]),
        find_regex(blocks, r"\b[A-Z]{1,2}\s?\d{5,8}\b"),
    )
    nom = find_value_near_label(blocks, ["nom"])
    prenom = find_value_near_label(blocks, ["prénom", "prenom"])
    date_naissance = choose_first_result(
        find_value_near_label(blocks, ["date de naissance", "né le", "nee le"]),
        find_regex(blocks, r"\b\d{2}[\/\-.]\d{2}[\/\-.]\d{4}\b"),
    )
    lieu_naissance = find_value_near_label(blocks, ["lieu de naissance", "né à", "nee a"])
    adresse = find_value_near_label(
        blocks, ["adresse"], directions=("right", "below"), y_tolerance=28.0, x_tolerance=500.0
    )
    nationalite = find_value_near_label(blocks, ["nationalité", "nationalite"])
    sexe = find_value_near_label(blocks, ["sexe"])

    date_status = "TO_REVIEW" if date_naissance.get("method", "").startswith("regex") else None
    fields = [
        _field("cin_numero", cin_numero),
        _field("nom", nom),
        _field("prenom", prenom),
        _field("date_naissance", date_naissance, date_status),
        _field("lieu_naissance", lieu_naissance),
        _field("adresse", adresse),
        _field("nationalite", nationalite),
        _field("sexe", sexe),
    ]

    missing = [f["field_name"] for f in fields if f["field_name"] in REQUIRED_FIELDS and f["raw_value"] is None]
    review = [f["field_name"] for f in fields if f["field_name"] in REQUIRED_FIELDS and f["status"] == "TO_REVIEW"]
    status = "INCOMPLETE" if missing else ("TO_REVIEW" if review else "OK")

    source_file = str(classification.get("file") or ocr_json.get("file") or "cin")
    return {
        "dossier_id": dossier_id,
        "doc_id": str(classification.get("doc_id") or f"{dossier_id}_{Path(source_file).stem}"),
        "doc_type": "CIN",
        "template_id": "CIN_V1",
        "original_path": classification.get("original_path", ocr_json.get("original_path")),
        "ocr_path": classification.get("ocr_path"),
        "classification": {
            "status": classification.get("status"),
            "doc_type_confidence": classification.get("doc_type_confidence"),
            "template_confidence": classification.get("template_confidence"),
        },
        "fields": fields,
        "extraction_validation": {
            "required_fields": sorted(REQUIRED_FIELDS),
            "missing_required_fields": missing,
            "fields_to_review": review,
        },
        "extraction_status": status,
    }


def extract_cin_from_files(ocr_path: str | Path, classification: dict[str, Any], dossier_id: str) -> dict[str, Any]:
    return extract_cin(load_json(ocr_path), classification, dossier_id)
