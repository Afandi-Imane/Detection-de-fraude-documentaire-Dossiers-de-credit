from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .common import (
    build_document_result,
    find_regex,
    find_value_near_label,
    get_blocks,
    load_json,
    make_field,
)

REQUIRED_FIELDS = {"nom_abonne", "adresse", "ville"}
SUPPORTED_TEMPLATES = {
    "QUITTANCE_LYDEC_V1",
    "QUITTANCE_AMENDIS_V1",
    "QUITTANCE_MAROC_TELECOM_V1",
}

# Expressions reprises de l'extracteur OCR audité.
NAME_RE = re.compile(r"^([A-ZÀ-Ü]+(?:[\s\-][A-ZÀ-Ü]+)*)\s+([A-Z][a-zà-ÿ]+)$")
P_CODE_RE = re.compile(r"^P\s?\d{4,}$", re.IGNORECASE)


def _result(value: Any, blocks: list[dict[str, Any]], method: str) -> dict[str, Any]:
    return {"value": value, "blocks": blocks, "method": method}


def extract_quittance(
    ocr_json: dict[str, Any],
    classification: dict[str, Any],
    dossier_id: str,
) -> dict[str, Any]:
    template_id = str(classification.get("template_id"))
    if classification.get("doc_type") != "QUITTANCE" and template_id not in SUPPORTED_TEMPLATES:
        raise ValueError("L'extracteur de quittance a reçu un autre type de document.")

    blocks = get_blocks(ocr_json)
    texts = [block["text"].strip() for block in blocks]

    name_index = next((index for index, text in enumerate(texts) if NAME_RE.fullmatch(text)), None)
    nom = _result(None, [], "not_found")
    adresse = _result(None, [], "not_found")
    ville = _result(None, [], "not_found")

    if name_index is not None:
        nom = _result(texts[name_index], [blocks[name_index]], "audited_name_pattern")
        address_index: int | None = None
        for index in range(name_index + 1, min(name_index + 5, len(texts))):
            text = texts[index]
            if not text or P_CODE_RE.fullmatch(text):
                continue
            adresse = _result(text, [blocks[index]], "audited_position_after_name")
            address_index = index
            break

        if address_index is not None:
            for index in range(address_index + 1, min(address_index + 4, len(texts))):
                text = texts[index]
                if not text or P_CODE_RE.fullmatch(text):
                    continue
                if text.isupper() and text.replace(" ", "").isalpha() and len(text) > 2:
                    ville = _result(text, [blocks[index]], "audited_city_after_address")
                    break

    date_facture = find_regex(blocks, r"Date\s*:\s*(\d{2}/\d{2}/\d{4})", group=1)
    net_a_payer = find_value_near_label(
        blocks,
        ["net à payer", "net a payer"],
        directions=("right",),
        y_tolerance=28.0,
    )

    fields = [
        make_field("nom_abonne", nom["value"], nom["blocks"], nom["method"]),
        make_field("adresse", adresse["value"], adresse["blocks"], adresse["method"]),
        make_field("ville", ville["value"], ville["blocks"], ville["method"]),
        make_field("date_facture", date_facture["value"], date_facture["blocks"], date_facture["method"]),
        make_field("net_a_payer", net_a_payer["value"], net_a_payer["blocks"], net_a_payer["method"]),
    ]

    return build_document_result(
        ocr_json=ocr_json,
        classification=classification,
        dossier_id=dossier_id,
        doc_type="QUITTANCE",
        template_id=template_id,
        fields=fields,
        required_fields=REQUIRED_FIELDS,
        notes=["Règles de localisation reprises du code OCR audité."],
    )


def extract_quittance_from_files(
    ocr_path: str | Path,
    classification: dict[str, Any],
    dossier_id: str,
) -> dict[str, Any]:
    return extract_quittance(load_json(ocr_path), classification, dossier_id)
