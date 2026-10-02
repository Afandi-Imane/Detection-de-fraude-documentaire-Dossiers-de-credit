from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .common import (
    choose_first_result,
    find_regex,
    find_value_near_label,
    get_blocks,
    load_json,
    make_field,
    normalize_text,
)

REQUIRED_FIELDS = {"nom", "cin", "employeur", "poste", "date_embauche"}


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


def _extract_employeur(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    pattern = re.compile(
        r"nous\s+soussign[eé]s?\s*,?\s*(.+?)(?:,\s*repr[eé]sent[eé]e?|,\s*attestons?|,\s*certifions?)",
        re.IGNORECASE,
    )
    for block in blocks:
        match = pattern.search(block["text"])
        if match:
            value = match.group(1).strip(" ,;:-")
            if value:
                return {"value": value, "blocks": [block], "method": "employeur_phrase_nous_soussignes"}

    excluded = {"adresse", "tel", "telephone", "email", "ice", "rc", "if", "cnss", "fait a", "attestation de travail", "nous soussignes"}
    for block in blocks[:12]:
        normalized = normalize_text(block["text"])
        if not normalized or any(keyword in normalized for keyword in excluded):
            continue
        if len(re.findall(r"[A-Za-zÀ-ÿ]", block["text"])) >= 2 and len(block["text"].strip()) <= 100:
            return {"value": block["text"].strip(), "blocks": [block], "method": "employeur_entete"}
    return {"value": None, "blocks": [], "method": "not_found"}


def _detect_cachet_indicator(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    direct = [block for block in blocks if "cachet" in normalize_text(block["text"])]
    if direct:
        return {"value": True, "blocks": direct, "method": "mot_cachet_detecte"}

    signature_blocks = [block for block in blocks if "signature" in normalize_text(block["text"])]
    if not signature_blocks:
        return {"value": None, "blocks": [], "method": "preuve_cachet_insuffisante"}

    signature_y = min(float(block.get("y", 0.0)) for block in signature_blocks)
    stamp_blocks = []
    for block in blocks:
        if float(block.get("y", 0.0)) <= signature_y:
            continue
        normalized = normalize_text(block["text"])
        if any(keyword in normalized for keyword in ("societe", "r c", "rc", "ice", "if")):
            stamp_blocks.append(block)

    unique = {normalize_text(block["text"]) for block in stamp_blocks}
    if len(unique) >= 2:
        return {"value": True, "blocks": signature_blocks + stamp_blocks, "method": "marqueurs_cachet_apres_signature"}
    return {"value": None, "blocks": signature_blocks + stamp_blocks, "method": "preuve_cachet_insuffisante"}


def extract_attestation_travail(
    ocr_json: dict[str, Any],
    classification: dict[str, Any],
    dossier_id: str,
) -> dict[str, Any]:
    if classification.get("doc_type") != "ATTESTATION_TRAVAIL" and classification.get("template_id") != "ATTESTATION_TRAVAIL_V1":
        raise ValueError("L'extracteur d'attestation a reçu un autre type de document.")

    blocks = get_blocks(ocr_json)
    nom = find_value_near_label(
        blocks, ["nom et prénom", "nom et prenom", "nom complet"],
        directions=("right", "below"), y_tolerance=28.0, x_tolerance=500.0,
    )
    cin = choose_first_result(
        find_value_near_label(blocks, ["cin", "n° cin", "numero cin", "numéro cin"]),
        find_regex(blocks, r"\b[A-Z]{1,2}\s?\d{5,8}\b"),
    )
    employeur = _extract_employeur(blocks)
    poste = find_value_near_label(
        blocks, ["poste", "fonction", "qualité", "qualite"],
        directions=("right", "below"), y_tolerance=28.0, x_tolerance=500.0,
    )
    date_embauche = find_value_near_label(
        blocks, ["date d'embauche", "date d embauche", "embauché le", "embauche le"],
        directions=("right", "below"), y_tolerance=28.0, x_tolerance=500.0,
    )
    salaire = find_value_near_label(
        blocks, ["salaire mensuel", "salaire net mensuel", "rémunération mensuelle", "remuneration mensuelle"],
        directions=("right", "below"), y_tolerance=28.0, x_tolerance=500.0,
    )
    cachet = _detect_cachet_indicator(blocks)

    fields = [
        _field("nom", nom),
        _field("cin", cin),
        _field("employeur", employeur),
        _field("poste", poste),
        _field("date_embauche", date_embauche),
        _field("salaire_mensuel", salaire, None if salaire.get("value") is not None else "ABSENT_DU_TEMPLATE"),
        make_field(
            "cachet_present", cachet.get("value"), cachet.get("blocks", []), cachet.get("method", "preuve_cachet_insuffisante"),
            "EXTRACTED" if cachet.get("value") is True else "TO_REVIEW",
        ),
    ]

    missing = [f["field_name"] for f in fields if f["field_name"] in REQUIRED_FIELDS and f["raw_value"] is None]
    review = [f["field_name"] for f in fields if f["field_name"] in REQUIRED_FIELDS and f["status"] == "TO_REVIEW"]
    status = "INCOMPLETE" if missing else ("TO_REVIEW" if review else "OK")
    source_file = str(classification.get("file") or ocr_json.get("file") or "attestation")

    return {
        "dossier_id": dossier_id,
        "doc_id": str(classification.get("doc_id") or f"{dossier_id}_{Path(source_file).stem}"),
        "doc_type": "ATTESTATION_TRAVAIL",
        "template_id": "ATTESTATION_TRAVAIL_V1",
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
            "notes": [
                "cachet_present est seulement un indicateur OCR ; le moteur visuel doit confirmer le cachet.",
                "salaire_mensuel peut être ABSENT_DU_TEMPLATE.",
            ],
        },
        "extraction_status": status,
    }


def extract_attestation_from_files(ocr_path: str | Path, classification: dict[str, Any], dossier_id: str) -> dict[str, Any]:
    return extract_attestation_travail(load_json(ocr_path), classification, dossier_id)
