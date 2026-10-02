from __future__ import annotations

import copy
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON introuvable : {json_path}")
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Le JSON doit contenir un objet : {json_path}")
    return data


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return output_path


def normalize_text(value: Any) -> str:
    """Normalisation uniquement pour comparer/rechercher, jamais pour remplacer raw_value."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFD", str(value))
    text = "".join(
        character for character in text
        if unicodedata.category(character) != "Mn"
    )
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def clean_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    text = re.sub(r"^[\s:;.,_\-–—|]+", "", text).strip()
    return text or None


def get_blocks(ocr_json: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = ocr_json.get("blocks", [])
    if not isinstance(blocks, list):
        raise ValueError("ocr_json['blocks'] doit être une liste.")

    valid_blocks: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = str(block.get("text", "")).strip()
        if not text:
            continue

        item = dict(block)
        item["text"] = text
        item["score"] = float(item.get("score", 0.0) or 0.0)
        item.setdefault("x", (float(item.get("x0", 0.0)) + float(item.get("x1", 0.0))) / 2)
        item.setdefault("y", (float(item.get("y0", 0.0)) + float(item.get("y1", 0.0))) / 2)
        item.setdefault("x0", float(item.get("x", 0.0)))
        item.setdefault("x1", float(item.get("x", 0.0)))
        item.setdefault("y0", float(item.get("y", 0.0)))
        item.setdefault("y1", float(item.get("y", 0.0)))
        valid_blocks.append(item)

    return valid_blocks


def average_confidence(blocks: Iterable[dict[str, Any]]) -> float:
    scores = [float(block.get("score", 0.0) or 0.0) for block in blocks]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def union_bbox(blocks: Iterable[dict[str, Any]]) -> list[float] | None:
    items = list(blocks)
    if not items:
        return None
    return [
        min(float(block.get("x0", 0.0)) for block in items),
        min(float(block.get("y0", 0.0)) for block in items),
        max(float(block.get("x1", 0.0)) for block in items),
        max(float(block.get("y1", 0.0)) for block in items),
    ]


def _preserve_raw_value(raw_value: Any) -> Any:
    """Conserve la valeur extraite. Aucun nettoyage destructif pour listes/dicts/nombres."""
    if isinstance(raw_value, (list, dict, int, float)) and not isinstance(raw_value, bool):
        return copy.deepcopy(raw_value)
    if isinstance(raw_value, bool):
        return raw_value
    return clean_value(raw_value)


def make_field(
    field_name: str,
    raw_value: Any,
    evidence_blocks: Iterable[dict[str, Any]] | None = None,
    method: str = "not_found",
    status: str | None = None,
) -> dict[str, Any]:
    blocks = list(evidence_blocks or [])
    preserved = _preserve_raw_value(raw_value)
    if status is None:
        status = "EXTRACTED" if preserved is not None else "MISSING"
    return {
        "field_name": field_name,
        "raw_value": preserved,
        "normalized_value": None,
        "confidence": average_confidence(blocks) if preserved is not None else 0.0,
        "status": status,
        "extraction_method": method,
        "evidence": {
            "texts": [block.get("text", "") for block in blocks],
            "bbox": union_bbox(blocks),
        },
    }


def _inline_value(text: str, labels: Iterable[str]) -> str | None:
    normalized_text = normalize_text(text)
    for label in labels:
        normalized_label = normalize_text(label)
        if not normalized_label or normalized_label not in normalized_text:
            continue

        separator = re.search(r"[:;|]", text)
        if separator:
            value = clean_value(text[separator.end():])
            if value:
                return value

        original_tokens = text.strip().split()
        label_token_count = len(normalized_label.split())
        if len(original_tokens) > label_token_count:
            value = clean_value(" ".join(original_tokens[label_token_count:]))
            if value:
                return value
    return None


def find_value_near_label(
    blocks: list[dict[str, Any]],
    label_variants: Iterable[str],
    directions: tuple[str, ...] = ("right", "below"),
    y_tolerance: float = 22.0,
    x_tolerance: float = 420.0,
) -> dict[str, Any]:
    variants = [normalize_text(label) for label in label_variants if normalize_text(label)]

    for label_block in blocks:
        label_text = normalize_text(label_block["text"])
        matched = any(
            re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", label_text)
            for variant in variants
        )
        if not matched:
            continue

        inline = _inline_value(label_block["text"], label_variants)
        if inline:
            return {"value": inline, "blocks": [label_block], "method": "inline_label"}

        for direction in directions:
            candidates: list[tuple[float, dict[str, Any]]] = []
            for other in blocks:
                if other is label_block:
                    continue

                if direction == "right":
                    same_row = abs(float(other["y"]) - float(label_block["y"])) <= y_tolerance
                    is_right = float(other["x0"]) >= float(label_block["x1"]) - 3
                    if same_row and is_right:
                        distance = max(0.0, float(other["x0"]) - float(label_block["x1"]))
                        candidates.append((distance, other))

                elif direction == "below":
                    vertical = float(other["y0"]) - float(label_block["y1"])
                    horizontal = abs(float(other["x"]) - float(label_block["x"]))
                    if vertical >= -3 and horizontal <= x_tolerance:
                        candidates.append((max(vertical, 0.0) + 0.15 * horizontal, other))

            if candidates:
                candidates.sort(key=lambda item: item[0])
                value_block = candidates[0][1]
                return {
                    "value": clean_value(value_block["text"]),
                    "blocks": [label_block, value_block],
                    "method": f"{direction}_of_label",
                }

    return {"value": None, "blocks": [], "method": "not_found"}


def find_regex(
    blocks: list[dict[str, Any]],
    pattern: str,
    flags: int = re.IGNORECASE,
    group: int = 0,
) -> dict[str, Any]:
    regex = re.compile(pattern, flags)
    for block in blocks:
        match = regex.search(block["text"])
        if match:
            return {"value": clean_value(match.group(group)), "blocks": [block], "method": "regex_block"}

    match = regex.search("\n".join(block["text"] for block in blocks))
    if match:
        return {"value": clean_value(match.group(group)), "blocks": [], "method": "regex_full_text"}
    return {"value": None, "blocks": [], "method": "not_found"}


def choose_first_result(*results: dict[str, Any]) -> dict[str, Any]:
    for result in results:
        if result.get("value") is not None:
            return result
    return {"value": None, "blocks": [], "method": "not_found"}


# ---------------------------------------------------------------------------
# Helpers repris du code OCR audité. Ils servent uniquement à localiser les
# valeurs. Les valeurs brutes sont conservées et la normalisation est séparée.
# ---------------------------------------------------------------------------
AMOUNT_RE = re.compile(r"^\d[\d\s]*[.,]\d{2}$")
AMOUNT_SEARCH_RE = re.compile(r"\d[\d\s]*[.,]\d{2}")
PERCENT_RE = re.compile(r"^\d+(?:[.,]\d+)?%$")
DATE_RE = re.compile(r"^\d{2}/\d{2}(?:/\d{2,4})?$")


def cluster_rows(blocks: list[dict[str, Any]], y_gap: float = 10.0) -> list[list[dict[str, Any]]]:
    if not blocks:
        return []
    sorted_blocks = sorted(blocks, key=lambda item: float(item["y"]))
    rows: list[list[dict[str, Any]]] = []
    current = [sorted_blocks[0]]
    for block in sorted_blocks[1:]:
        if abs(float(block["y"]) - float(current[-1]["y"])) <= y_gap:
            current.append(block)
        else:
            rows.append(current)
            current = [block]
    rows.append(current)
    for row in rows:
        row.sort(key=lambda item: float(item["x"]))
    rows.sort(key=lambda row: sum(float(item["y"]) for item in row) / len(row))
    return rows


def find_line_index(blocks: list[dict[str, Any]], keyword: str) -> int | None:
    target = normalize_text(keyword)
    for index, block in enumerate(blocks):
        if target in normalize_text(block["text"]):
            return index
    return None


def find_exact_text_block(blocks: list[dict[str, Any]], text: str) -> dict[str, Any] | None:
    target = normalize_text(text)
    for block in blocks:
        if normalize_text(block["text"]) == target:
            return block
    return None


def get_column_x(blocks: list[dict[str, Any]], header_text: str) -> float | None:
    block = find_exact_text_block(blocks, header_text)
    return float(block["x"]) if block is not None else None


def find_value_in_row_near_x(
    blocks: list[dict[str, Any]],
    row_label_variants: Iterable[str],
    target_x: float | None,
    *,
    want_percent: bool = False,
    y_tolerance: float = 18.0,
) -> dict[str, Any]:
    """Reprise de value_in_row_near_x du code audité, avec preuves OCR."""
    normalized_labels = [normalize_text(label) for label in row_label_variants]
    for label_block in blocks:
        label_text = normalize_text(label_block["text"])
        if not any(label in label_text for label in normalized_labels):
            continue

        candidates: list[dict[str, Any]] = []
        for other in blocks:
            if abs(float(other["y"]) - float(label_block["y"])) >= y_tolerance:
                continue
            text = other["text"].strip()
            is_percent = bool(PERCENT_RE.fullmatch(text))
            is_amount = bool(AMOUNT_RE.fullmatch(text))
            if want_percent and is_percent:
                candidates.append(other)
            elif not want_percent and is_amount:
                candidates.append(other)

        if not candidates:
            continue
        if target_x is not None:
            candidates.sort(key=lambda item: abs(float(item["x"]) - target_x))
        value_block = candidates[0]
        return {
            "value": value_block["text"],
            "blocks": [label_block, value_block],
            "method": "audited_row_column",
        }
    return {"value": None, "blocks": [], "method": "not_found"}


def extract_date_flexible(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).replace("_", "").replace(" ", "")
    match = re.search(r"\d{2}/\d{2}/\d{4}", cleaned)
    return match.group(0) if match else None


def find_row_index(rows: list[list[dict[str, Any]]], keyword: str) -> int | None:
    target = normalize_text(keyword)
    for index, row in enumerate(rows):
        if any(target in normalize_text(block["text"]) for block in row):
            return index
    return None


def build_document_result(
    *,
    ocr_json: dict[str, Any],
    classification: dict[str, Any],
    dossier_id: str,
    doc_type: str,
    template_id: str,
    fields: list[dict[str, Any]],
    required_fields: set[str],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    missing = [
        field["field_name"]
        for field in fields
        if field["field_name"] in required_fields
        and field.get("raw_value") is None
        and field.get("status") != "ABSENT_DU_TEMPLATE"
    ]
    review = [
        field["field_name"]
        for field in fields
        if field["field_name"] in required_fields
        and field.get("status") == "TO_REVIEW"
    ]
    status = "INCOMPLETE" if missing else ("TO_REVIEW" if review else "OK")
    source_file = str(classification.get("file") or ocr_json.get("file") or doc_type.lower())

    validation: dict[str, Any] = {
        "required_fields": sorted(required_fields),
        "missing_required_fields": missing,
        "fields_to_review": review,
    }
    if notes:
        validation["notes"] = notes

    return {
        "dossier_id": dossier_id,
        "doc_id": str(classification.get("doc_id") or f"{dossier_id}_{Path(source_file).stem}"),
        "doc_type": doc_type,
        "template_id": template_id,
        "original_path": classification.get("original_path", ocr_json.get("original_path")),
        "ocr_path": classification.get("ocr_path"),
        "classification": {
            "status": classification.get("status"),
            "doc_type_confidence": classification.get("doc_type_confidence"),
            "template_confidence": classification.get("template_confidence"),
        },
        "fields": fields,
        "extraction_validation": validation,
        "extraction_status": status,
    }


def find_value_near_label_audited(
    blocks: list[dict[str, Any]],
    label_variants: Iterable[str],
    direction: str = "right",
    y_tolerance: float = 20.0,
) -> dict[str, Any]:
    """Version exacte de la recherche spatiale utilisée dans le code OCR audité."""
    for label_block in blocks:
        normalized_text = normalize_text(label_block["text"])
        for label in label_variants:
            normalized_label = normalize_text(label)
            if not re.search(
                r"(?<![a-z])" + re.escape(normalized_label) + r"(?![a-z])",
                normalized_text,
            ):
                continue

            index = normalized_text.find(normalized_label)
            remainder = clean_value(label_block["text"][index + len(normalized_label):])
            if remainder and len(remainder) > 1:
                return {"value": remainder, "blocks": [label_block], "method": "audited_inline_label"}

            candidates: list[tuple[float, dict[str, Any]]] = []
            for other in blocks:
                if other is label_block:
                    continue
                if (
                    direction == "right"
                    and abs(float(other["y"]) - float(label_block["y"])) < y_tolerance
                    and float(other["x"]) > float(label_block["x"])
                ):
                    candidates.append((float(other["x"]) - float(label_block["x"]), other))
                elif (
                    direction == "below"
                    and float(other["y"]) > float(label_block["y"])
                    and abs(float(other["x"]) - float(label_block["x"])) < 400
                ):
                    candidates.append((float(other["y"]) - float(label_block["y"]), other))

            if candidates:
                candidates.sort(key=lambda item: item[0])
                value_block = candidates[0][1]
                return {
                    "value": clean_value(value_block["text"]),
                    "blocks": [label_block, value_block],
                    "method": f"audited_{direction}_of_label",
                }
    return {"value": None, "blocks": [], "method": "not_found"}
