from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_DOCUMENT_COUNTS = {
    "CIN": 1,
    "QUITTANCE": 1,
    "RIB": 1,
    "RELEVE_BANCAIRE": 3,
    "BULLETIN_SALAIRE": 3,
    "ATTESTATION_TRAVAIL": 1,
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON invalide : {path}")
    return data


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return output


def _field_index(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for field in document.get("fields", []):
        if not isinstance(field, dict) or not field.get("field_name"):
            continue
        index[str(field["field_name"])] = {
            "raw_value": field.get("raw_value"),
            "normalized_value": field.get("normalized_value"),
            "confidence": field.get("confidence"),
            "validation_status": field.get("validation_status"),
        }
    return index


def build_dossier(documents: list[dict[str, Any]], dossier_id: str | None = None) -> dict[str, Any]:
    if not documents:
        raise ValueError("Aucun document validé à assembler.")

    resolved_dossier_id = dossier_id or str(documents[0].get("dossier_id") or "DOSSIER_UNKNOWN")
    counts = Counter(str(document.get("doc_type", "UNKNOWN")) for document in documents)
    by_type: dict[str, list[str]] = defaultdict(list)
    document_fields: dict[str, dict[str, dict[str, Any]]] = {}

    for document in documents:
        doc_id = str(document.get("doc_id") or "DOC_UNKNOWN")
        by_type[str(document.get("doc_type", "UNKNOWN"))].append(doc_id)
        document_fields[doc_id] = _field_index(document)

    missing_documents = {
        doc_type: expected - counts.get(doc_type, 0)
        for doc_type, expected in EXPECTED_DOCUMENT_COUNTS.items()
        if counts.get(doc_type, 0) < expected
    }
    extra_documents = {
        doc_type: counts.get(doc_type, 0) - expected
        for doc_type, expected in EXPECTED_DOCUMENT_COUNTS.items()
        if counts.get(doc_type, 0) > expected
    }

    quality_counts = Counter(str(document.get("validation_status", "UNKNOWN")) for document in documents)
    ready = (
        not missing_documents
        and quality_counts.get("INCOMPLETE", 0) == 0
        and quality_counts.get("INVALID", 0) == 0
        and quality_counts.get("NOT_IMPLEMENTED", 0) == 0
    )

    return {
        "dossier_id": resolved_dossier_id,
        "documents": documents,
        "documents_by_type": dict(by_type),
        "document_fields": document_fields,
        "data_quality": {
            "scope": "INPUT_FOR_SIGNALS_NOT_FRAUD_DECISION",
            "status": "READY_FOR_SIGNALS" if ready else "PARTIAL",
            "ready_for_signals": ready,
            "document_count": len(documents),
            "expected_document_counts": EXPECTED_DOCUMENT_COUNTS,
            "observed_document_counts": dict(counts),
            "missing_documents": missing_documents,
            "extra_documents": extra_documents,
            "document_validation_counts": dict(quality_counts),
            "raw_values_preserved": True,
        },
    }


def build_dossier_from_folder(
    validated_dir: str | Path,
    output_path: str | Path | None = None,
    dossier_id: str | None = None,
) -> dict[str, Any]:
    source = Path(validated_dir)
    paths = [path for path in sorted(source.glob("*.json")) if path.name != "dossier.json"]
    documents = [load_json(path) for path in paths]
    dossier = build_dossier(documents, dossier_id=dossier_id)
    destination = Path(output_path) if output_path else source / "dossier.json"
    save_json(dossier, destination)
    return dossier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble les documents validés sans calculer les signaux de fraude.")
    parser.add_argument("--input", required=True, help="Dossier data/json_validated/Dxxx")
    parser.add_argument("--output")
    parser.add_argument("--dossier-id")
    args = parser.parse_args()
    result = build_dossier_from_folder(args.input, args.output, args.dossier_id)
    print(f"Dossier construit : {result['dossier_id']}")
    print(f"Documents : {result['data_quality']['document_count']}")
    print(f"Prêt pour les signaux : {result['data_quality']['ready_for_signals']}")
