from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    build_document_result,
    find_line_index,
    find_regex,
    find_value_near_label,
    get_blocks,
    load_json,
    make_field,
    normalize_text,
)

REQUIRED_FIELDS = {"banque", "titulaire", "rib"}
SUPPORTED_TEMPLATES = {"RIB_CIH_V1", "RIB_ATTIJARI_V1"}


def _bank_from_template(template_id: str) -> str:
    return "CIH" if template_id == "RIB_CIH_V1" else "ATTIJARI"


def _bank_result(blocks: list[dict[str, Any]], bank: str) -> dict[str, Any]:
    keywords = ("cih",) if bank == "CIH" else ("attijariwafa", "attijari")
    for block in blocks:
        normalized = normalize_text(block["text"])
        if any(keyword in normalized for keyword in keywords):
            return {"value": block["text"], "blocks": [block], "method": "template_and_ocr_header"}
    fallback = "CIH BANK" if bank == "CIH" else "Attijariwafa bank"
    return {"value": fallback, "blocks": [], "method": "template_id"}


def _extract_cih_account_number(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    index = find_line_index(blocks, "clé")
    if index is None:
        index = find_line_index(blocks, "cle")
    if index is None:
        return {"value": None, "blocks": [], "method": "not_found"}

    values: list[str] = []
    evidence: list[dict[str, Any]] = [blocks[index]]
    for block in blocks[index + 1:index + 8]:
        text = block["text"].strip().replace(" ", "")
        if text.isdigit():
            values.append(text)
            evidence.append(block)
        if len(values) == 4:
            break
    return {
        "value": "".join(values) if len(values) == 4 else None,
        "blocks": evidence if len(values) == 4 else [],
        "method": "audited_cih_four_components" if len(values) == 4 else "not_found",
    }


def extract_rib(
    ocr_json: dict[str, Any],
    classification: dict[str, Any],
    dossier_id: str,
) -> dict[str, Any]:
    template_id = str(classification.get("template_id"))
    if classification.get("doc_type") != "RIB" and template_id not in SUPPORTED_TEMPLATES:
        raise ValueError("L'extracteur RIB a reçu un autre type de document.")

    blocks = get_blocks(ocr_json)
    bank = _bank_from_template(template_id)
    banque = _bank_result(blocks, bank)

    if bank == "CIH":
        titulaire = find_value_near_label(
            blocks,
            ["intitulé du compte", "intitule du compte"],
            directions=("right", "below"),
            y_tolerance=30.0,
        )
        rib = _extract_cih_account_number(blocks)
    else:
        titulaire = find_value_near_label(
            blocks,
            ["titulaire du compte"],
            directions=("right",),
            y_tolerance=30.0,
        )
        rib = find_value_near_label(
            blocks,
            ["références bancaires", "references bancaires"],
            directions=("right",),
            y_tolerance=30.0,
        )
        if rib.get("value") is None:
            rib = find_regex(blocks, r"\b\d(?:[\d ]{18,30})\d\b")

    iban = find_regex(blocks, r"\bMA\d{2}(?:\s?\d{4}){4,6}\b")
    iban_status = None if iban.get("value") is not None else "ABSENT_DU_TEMPLATE"

    fields = [
        make_field("banque", banque["value"], banque["blocks"], banque["method"]),
        make_field("titulaire", titulaire["value"], titulaire["blocks"], titulaire["method"]),
        make_field("rib", rib["value"], rib["blocks"], rib["method"]),
        make_field("iban", iban["value"], iban["blocks"], iban["method"], iban_status),
    ]

    return build_document_result(
        ocr_json=ocr_json,
        classification=classification,
        dossier_id=dossier_id,
        doc_type="RIB",
        template_id=template_id,
        fields=fields,
        required_fields=REQUIRED_FIELDS,
        notes=[
            "La banque est routée par template_id, puis reliée au texte d'en-tête OCR.",
            "IBAN peut être ABSENT_DU_TEMPLATE ; aucune valeur n'est inventée.",
        ],
    )


def extract_rib_from_files(
    ocr_path: str | Path,
    classification: dict[str, Any],
    dossier_id: str,
) -> dict[str, Any]:
    return extract_rib(load_json(ocr_path), classification, dossier_id)
