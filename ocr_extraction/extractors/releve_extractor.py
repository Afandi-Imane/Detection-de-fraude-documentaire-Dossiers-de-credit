from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .common import (
    AMOUNT_RE,
    AMOUNT_SEARCH_RE,
    DATE_RE,
    average_confidence,
    build_document_result,
    clean_value,
    cluster_rows,
    find_line_index,
    find_regex,
    find_row_index,
    find_value_near_label,
    find_value_near_label_audited,
    get_blocks,
    load_json,
    make_field,
    normalize_text,
)

REQUIRED_FIELDS = {
    "banque",
    "titulaire",
    "adresse",
    "rib",
    "solde_ouverture",
    "solde_cloture",
    "total_credits",
    "total_debits",
    "transactions",
}
SUPPORTED_TEMPLATES = {"RELEVE_CIH_V1", "RELEVE_ATTIJARI_V1"}

NAME_LOOSE_RE = re.compile(r"^[A-ZÀ-Ü][A-Za-zà-ÿ.'\-]*(?:\s+[A-Za-zà-ÿ.'\-]+){1,3}$")
EXCLUDE_TITULAIRE_WORDS = {
    "RELEVE", "AGENCE", "BANQUE", "DEVISE", "CONSEILLER",
    "IDENTITE", "COMPTE", "BANCAIRE", "CASABLANCA", "MAROC",
}
ADDRESS_STOPWORDS = {
    "MAROC", "BANQUE", "VILLE", "CLE", "CLÉ", "DEVISE", "DH",
    "N° DE COMPTE", "COMPTE", "CAPITAUX",
}


def _bank_from_template(template_id: str) -> str:
    return "CIH" if template_id == "RELEVE_CIH_V1" else "ATTIJARI"


def _extract_bank(blocks: list[dict[str, Any]], bank: str) -> dict[str, Any]:
    keywords = ("cih",) if bank == "CIH" else ("attijariwafa", "attijari")
    for block in blocks[:20]:
        text = normalize_text(block["text"])
        if any(keyword in text for keyword in keywords):
            return {"value": block["text"], "blocks": [block], "method": "template_and_ocr_header"}
    fallback = "CIH BANK" if bank == "CIH" else "Attijariwafa bank"
    return {"value": fallback, "blocks": [], "method": "template_id"}


def _extract_titulaire(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    index = find_line_index(blocks, "relev")
    if index is None:
        return {"value": None, "blocks": [], "method": "not_found"}
    for block in blocks[index:index + 8]:
        text = block["text"].strip()
        if not text or not NAME_LOOSE_RE.fullmatch(text):
            continue
        if any(keyword in text.upper() for keyword in EXCLUDE_TITULAIRE_WORDS):
            continue
        return {"value": text, "blocks": [block], "method": "audited_name_after_releve_header"}
    return {"value": None, "blocks": [], "method": "not_found"}


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


def _find_column_threshold(blocks: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    rows = cluster_rows(blocks, y_gap=9)
    amount_xs: list[float] = []
    for row in rows:
        dates = [block for block in row if DATE_RE.fullmatch(block["text"].strip())]
        amounts = [block for block in row if AMOUNT_RE.fullmatch(block["text"].strip())]
        if dates and amounts:
            row_text = " ".join(block["text"] for block in row).upper()
            if not any(keyword in row_text for keyword in ("SOLDE", "TOTAL")):
                amount_xs.append(float(amounts[0]["x"]))

    if len(amount_xs) < 2:
        return None, None
    xs = sorted(amount_xs)
    gaps = [(xs[index + 1] - xs[index], (xs[index] + xs[index + 1]) / 2) for index in range(len(xs) - 1)]
    gaps.sort(key=lambda item: -item[0])
    best_gap, threshold = gaps[0]
    if best_gap < 5:
        return None, None
    left_count = sum(1 for x in amount_xs if x < threshold)
    right_count = len(amount_xs) - left_count
    return threshold, "left" if left_count >= right_count else "right"


def _classify_amount_x(x: float, threshold: float | None, debit_side: str | None) -> str:
    if threshold is None or debit_side is None:
        return "D"
    side = "left" if x < threshold else "right"
    return "D" if side == debit_side else "C"


def _extract_transactions(
    blocks: list[dict[str, Any]],
    threshold: float | None,
    debit_side: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = cluster_rows(blocks, y_gap=9)
    transactions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for row in rows:
        date_tokens = [block for block in row if DATE_RE.fullmatch(block["text"].strip())]
        amount_tokens = [block for block in row if AMOUNT_RE.fullmatch(block["text"].strip())]
        if not date_tokens or not amount_tokens:
            continue
        row_text = " ".join(block["text"] for block in row).upper()
        if any(keyword in row_text for keyword in ("SOLDE", "TOTAL")):
            continue

        date = date_tokens[0]["text"].strip()
        libelle_tokens = [block["text"] for block in row if block not in date_tokens and block not in amount_tokens]
        libelle = " ".join(text for text in libelle_tokens if text.strip()).strip()
        amount_block = amount_tokens[0]
        transactions.append({
            "date": date,
            "libelle": libelle,
            "montant": amount_block["text"],
            "sens": _classify_amount_x(float(amount_block["x"]), threshold, debit_side),
            "ocr_confidence": average_confidence(row),
        })
        evidence.extend(row)
    return transactions, evidence


def _extract_solde(
    rows: list[list[dict[str, Any]]],
    label_keywords: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    index: int | None = None
    for keyword in label_keywords:
        index = find_row_index(rows, keyword)
        if index is not None:
            break
    empty = {"value": None, "blocks": [], "method": "not_found"}
    if index is None:
        return empty, empty

    candidate_rows = [rows[index]]
    if index + 1 < len(rows):
        candidate_rows.append(rows[index + 1])

    date_result = dict(empty)
    amount_result = dict(empty)
    for row in candidate_rows:
        row_text = " ".join(block["text"] for block in row).upper()
        if "TOTAL" in row_text:
            continue
        for block in row:
            if date_result["value"] is None:
                compact = block["text"].replace("_", "").replace(" ", "")
                match = re.search(r"\d{2}/\d{2}/\d{4}", compact)
                if match:
                    date_result = {"value": match.group(0), "blocks": [block], "method": "audited_solde_row_date"}
            if amount_result["value"] is None:
                match = AMOUNT_SEARCH_RE.search(block["text"])
                if match:
                    amount_result = {"value": match.group(0), "blocks": [block], "method": "audited_solde_row_amount"}
    return date_result, amount_result


def _extract_totaux(
    rows: list[list[dict[str, Any]]],
    threshold: float | None,
    debit_side: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    index = find_row_index(rows, "total mouvements")
    if index is None:
        index = find_row_index(rows, "total des mouvements")
    empty = {"value": None, "blocks": [], "method": "not_found"}
    if index is None:
        return empty, empty

    candidate_rows = [rows[index]]
    if index - 1 >= 0:
        candidate_rows.append(rows[index - 1])
    if index + 1 < len(rows):
        candidate_rows.append(rows[index + 1])

    debit_result = dict(empty)
    credit_result = dict(empty)
    for row in candidate_rows:
        row_text = " ".join(block["text"] for block in row).upper()
        if "SOLDE" in row_text:
            continue
        if any(re.fullmatch(r"\d{3}", block["text"].strip()) for block in row):
            continue
        for block in row:
            if DATE_RE.fullmatch(block["text"].strip()):
                continue
            match = AMOUNT_SEARCH_RE.search(block["text"])
            if not match:
                continue
            role = _classify_amount_x(float(block["x"]), threshold, debit_side)
            result = {"value": match.group(0), "blocks": [block], "method": "audited_total_adjacent_row"}
            if role == "D" and debit_result["value"] is None:
                debit_result = result
            elif role == "C" and credit_result["value"] is None:
                credit_result = result
    return debit_result, credit_result


def _extract_adresse(
    blocks: list[dict[str, Any]],
    rows: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    result = find_value_near_label_audited(
        blocks, ["adresse"], direction="right", y_tolerance=20.0
    )
    if result.get("value"):
        return result
    result = find_value_near_label_audited(
        blocks, ["adresse"], direction="below", y_tolerance=20.0
    )
    if result.get("value"):
        return result

    name_row_index: int | None = None
    for index, row in enumerate(rows):
        for block in row:
            text = block["text"].strip()
            if NAME_LOOSE_RE.fullmatch(text) and not any(keyword in text.upper() for keyword in EXCLUDE_TITULAIRE_WORDS):
                name_row_index = index
                break
        if name_row_index is not None:
            break
    if name_row_index is None:
        return {"value": None, "blocks": [], "method": "not_found"}

    fragments: list[str] = []
    evidence: list[dict[str, Any]] = []
    for row in rows[name_row_index + 1:name_row_index + 4]:
        rightmost = max(row, key=lambda block: float(block["x"]))
        text = rightmost["text"].strip()

          # Ignorer les petits codes parasites produits par l'OCR :
           # exemples : U1S9, A12B, etc.
        compact = re.sub(r"\s+", "", text.upper())

        is_short_alphanumeric_noise = (
             len(compact) <= 8
             and re.search(r"[A-Z]", compact) is not None
             and re.search(r"\d", compact) is not None
        )

        if is_short_alphanumeric_noise:
                continue

        if (
            text
            and text.upper() not in ADDRESS_STOPWORDS
            and not DATE_RE.fullmatch(text)
        ):
           fragments.append(text)
           evidence.append(rightmost)

        if len(fragments) >= 2:
             break
    return {
        "value": ", ".join(fragments) if fragments else None,
        "blocks": evidence,
        "method": "audited_cih_address_after_name" if fragments else "not_found",
    }


def extract_releve_bancaire(
    ocr_json: dict[str, Any],
    classification: dict[str, Any],
    dossier_id: str,
) -> dict[str, Any]:
    template_id = str(classification.get("template_id"))
    if classification.get("doc_type") != "RELEVE_BANCAIRE" and template_id not in SUPPORTED_TEMPLATES:
        raise ValueError("L'extracteur de relevé a reçu un autre type de document.")

    blocks = get_blocks(ocr_json)
    bank = _bank_from_template(template_id)
    banque = _extract_bank(blocks, bank)
    titulaire = _extract_titulaire(blocks)

    rib = _extract_cih_account_number(blocks)
    if rib.get("value") is None:
        rib = find_regex(blocks, r"\d{20,24}")

    threshold, debit_side = _find_column_threshold(blocks)
    rows = cluster_rows(blocks, y_gap=9)
    adresse = _extract_adresse(blocks, rows)
    date_ouverture, solde_ouverture = _extract_solde(rows, ["solde depart"])
    date_cloture, solde_cloture = _extract_solde(rows, ["nouveau solde", "solde final"])
    total_debits, total_credits = _extract_totaux(rows, threshold, debit_side)
    transactions, transaction_blocks = _extract_transactions(blocks, threshold, debit_side)

    fields = [
        make_field("banque", banque["value"], banque["blocks"], banque["method"]),
        make_field("titulaire", titulaire["value"], titulaire["blocks"], titulaire["method"]),
        make_field("adresse", adresse["value"], adresse["blocks"], adresse["method"]),
        make_field("rib", rib["value"], rib["blocks"], rib["method"]),
        make_field("solde_ouverture", solde_ouverture["value"], solde_ouverture["blocks"], solde_ouverture["method"]),
        make_field("date_solde_ouverture", date_ouverture["value"], date_ouverture["blocks"], date_ouverture["method"]),
        make_field("solde_cloture", solde_cloture["value"], solde_cloture["blocks"], solde_cloture["method"]),
        make_field("date_solde_cloture", date_cloture["value"], date_cloture["blocks"], date_cloture["method"]),
        make_field("total_credits", total_credits["value"], total_credits["blocks"], total_credits["method"]),
        make_field("total_debits", total_debits["value"], total_debits["blocks"], total_debits["method"]),
        make_field(
            "transactions",
            transactions,
            transaction_blocks,
            "audited_transaction_rows",
            "EXTRACTED" if transactions else "MISSING",
        ),
    ]

    return build_document_result(
        ocr_json=ocr_json,
        classification=classification,
        dossier_id=dossier_id,
        doc_type="RELEVE_BANCAIRE",
        template_id=template_id,
        fields=fields,
        required_fields=REQUIRED_FIELDS,
        notes=[
            "Localisation des soldes, totaux et transactions reprise du code OCR audité.",
            "Aucun calcul de fraude ni correction arithmétique n'est exécuté ici.",
        ],
    )


def extract_releve_from_files(
    ocr_path: str | Path,
    classification: dict[str, Any],
    dossier_id: str,
) -> dict[str, Any]:
    return extract_releve_bancaire(load_json(ocr_path), classification, dossier_id)
