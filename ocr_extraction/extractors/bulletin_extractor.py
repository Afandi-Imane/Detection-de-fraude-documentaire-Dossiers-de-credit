from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .common import (
    build_document_result,
    find_line_index,
    find_value_in_row_near_x,
    find_value_near_label,
    get_blocks,
    get_column_x,
    load_json,
    make_field,
)

REQUIRED_FIELDS = {
    "nom_employe",
    "prenom_employe",
    "employeur",
    "poste",
    "periode",
    "salaire_brut",
    "net_a_payer",
}

COMPANY_NAME_RE = re.compile(r"^[A-Z]{3,}(?:\s[A-Z]{2,})*$")
EXCLUDE_COMPANY_WORDS = {
    "SIEGE", "DIRECTION", "NOMBRE", "PERIODE", "FONCTION", "SITUATION",
    "FAMILIALE", "CELIBATAIRE", "MARIE", "ANCIENNETE", "MATRICULE",
    "CNSS", "BULLETIN", "PAIE", "RUBRIQUE", "GAINS", "RETENUES",
    "PART", "PATRONAL", "BASE", "TAUX", "RESULTAT", "SALAIRE",
    "BRUT", "NET", "IMPOSABLE", "COTISATION", "SOCIETE",
}


def _extract_employeur_from_societe(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    index = find_line_index(blocks, "societe")
    if index is None:
        index = find_line_index(blocks, "société")
    if index is None:
        return {"value": None, "blocks": [], "method": "not_found"}

    for block in blocks[index + 1:index + 12]:
        text = block["text"].strip()
        if COMPANY_NAME_RE.fullmatch(text) and text.upper() not in EXCLUDE_COMPANY_WORDS:
            return {
                "value": text,
                "blocks": [blocks[index], block],
                "method": "audited_company_after_societe",
            }
    return {"value": None, "blocks": [], "method": "not_found"}


def extract_bulletin_salaire(
    ocr_json: dict[str, Any],
    classification: dict[str, Any],
    dossier_id: str,
) -> dict[str, Any]:
    template_id = str(classification.get("template_id"))
    if classification.get("doc_type") != "BULLETIN_SALAIRE" and template_id != "BULLETIN_SALAIRE_V1":
        raise ValueError("L'extracteur de bulletin a reçu un autre type de document.")

    blocks = get_blocks(ocr_json)
    gains_x = get_column_x(blocks, "GAINS")
    retenues_x = get_column_x(blocks, "RETENUES")

    nom = find_value_near_label(blocks, ["nom"], directions=("right",), y_tolerance=24.0)
    prenom = find_value_near_label(blocks, ["prénom", "prenom"], directions=("right",), y_tolerance=24.0)
    employeur = _extract_employeur_from_societe(blocks)
    poste = find_value_near_label(blocks, ["fonction"], directions=("right",), y_tolerance=24.0)
    periode = find_value_near_label(blocks, ["période", "periode"], directions=("right",), y_tolerance=24.0)
    anciennete = find_value_near_label(blocks, ["ancienneté", "anciennete"], directions=("right",), y_tolerance=24.0)

    fields = [
        make_field("nom_employe", nom["value"], nom["blocks"], nom["method"]),
        make_field("prenom_employe", prenom["value"], prenom["blocks"], prenom["method"]),
        make_field("employeur", employeur["value"], employeur["blocks"], employeur["method"]),
        make_field("poste", poste["value"], poste["blocks"], poste["method"]),
        make_field("periode", periode["value"], periode["blocks"], periode["method"]),
        make_field("anciennete", anciennete["value"], anciennete["blocks"], anciennete["method"]),
    ]

    amount_specs = [
        ("salaire_base_mensuel", ["salaire de base mensuel"], gains_x, False),
        ("salaire_brut", ["salaire brut"], gains_x, False),
        ("cotisation_cnss_base", ["cotisation cnss"], None, False),
        ("cotisation_cnss_taux", ["cotisation cnss"], None, True),
        ("cotisation_cnss_montant", ["cotisation cnss"], retenues_x, False),
        ("cotisation_amo_base", ["cotisation amo"], None, False),
        ("cotisation_amo_taux", ["cotisation amo"], None, True),
        ("cotisation_amo_montant", ["cotisation amo"], retenues_x, False),
        ("salaire_net_imposable", ["salaire net imposable"], gains_x, False),
        ("net_a_payer", ["net à payer", "net a payer"], None, False),
    ]
    for field_name, labels, target_x, want_percent in amount_specs:
        extracted = find_value_in_row_near_x(
            blocks,
            labels,
            target_x,
            want_percent=want_percent,
            y_tolerance=20.0,
        )
        fields.append(make_field(field_name, extracted["value"], extracted["blocks"], extracted["method"]))

    return build_document_result(
        ocr_json=ocr_json,
        classification=classification,
        dossier_id=dossier_id,
        doc_type="BULLETIN_SALAIRE",
        template_id="BULLETIN_SALAIRE_V1",
        fields=fields,
        required_fields=REQUIRED_FIELDS,
        notes=[
            "Règles de lignes/colonnes reprises de l'extracteur OCR audité.",
            "Les montants restent en raw_value ; leur conversion est faite séparément par normalizer.py.",
        ],
    )


def extract_bulletin_from_files(
    ocr_path: str | Path,
    classification: dict[str, Any],
    dossier_id: str,
) -> dict[str, Any]:
    return extract_bulletin_salaire(load_json(ocr_path), classification, dossier_id)
