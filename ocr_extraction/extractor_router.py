from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from ocr_extraction.extractors.attestation_extractor import extract_attestation_travail
from ocr_extraction.extractors.bulletin_extractor import extract_bulletin_salaire
from ocr_extraction.extractors.cin_extractor import extract_cin
from ocr_extraction.extractors.quittance_extractor import extract_quittance
from ocr_extraction.extractors.releve_extractor import extract_releve_bancaire
from ocr_extraction.extractors.rib_extractor import extract_rib
from ocr_extraction.extractors.common import load_json, save_json

Extractor = Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]]
EXTRACTOR_BY_TEMPLATE: dict[str, Extractor] = {
    "CIN_V1": extract_cin,
    "QUITTANCE_LYDEC_V1": extract_quittance,
    "QUITTANCE_AMENDIS_V1": extract_quittance,
    "QUITTANCE_MAROC_TELECOM_V1": extract_quittance,
    "RIB_CIH_V1": extract_rib,
    "RIB_ATTIJARI_V1": extract_rib,
    "RELEVE_CIH_V1": extract_releve_bancaire,
    "RELEVE_ATTIJARI_V1": extract_releve_bancaire,
    "BULLETIN_SALAIRE_V1": extract_bulletin_salaire,
    "ATTESTATION_TRAVAIL_V1": extract_attestation_travail,
}


def extract_document(classification: dict[str, Any], dossier_id: str) -> dict[str, Any]:
    template_id = classification.get("template_id")
    ocr_path = classification.get("ocr_path")
    if not ocr_path:
        return {"dossier_id": dossier_id, "doc_type": classification.get("doc_type"), "template_id": template_id, "fields": [], "extraction_status": "ERROR", "error": "ocr_path absent du manifeste."}

    extractor = EXTRACTOR_BY_TEMPLATE.get(template_id)
    if extractor is None:
        return {
            "dossier_id": dossier_id,
            "doc_id": classification.get("doc_id"),
            "doc_type": classification.get("doc_type"),
            "template_id": template_id,
            "original_path": classification.get("original_path"),
            "ocr_path": ocr_path,
            "fields": [],
            "extraction_status": "NOT_IMPLEMENTED",
            "error": f"Aucun extracteur enregistré pour {template_id}.",
        }
    return extractor(load_json(ocr_path), classification, dossier_id)


def process_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    only_template: str | None = None,
    only_implemented: bool = False,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    dossier_id = str(manifest.get("dossier_id", "DOSSIER_UNKNOWN"))
    documents = manifest.get("documents", [])
    if not isinstance(documents, list):
        raise ValueError("Le manifeste doit contenir une liste 'documents'.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    skipped = 0

    for index, classification in enumerate(documents, start=1):
        if not isinstance(classification, dict):
            continue
        template_id = classification.get("template_id")
        if only_template and template_id != only_template:
            continue
        if only_implemented and template_id not in EXTRACTOR_BY_TEMPLATE:
            skipped += 1
            continue

        source_file = str(classification.get("file") or f"document_{index:02d}")
        source_stem = Path(source_file).stem
        classification = dict(classification)
        classification.setdefault("doc_id", f"{dossier_id}_{source_stem}")
        print(f"[EXTRACTION] {source_file} ({template_id})")
        try:
            result = extract_document(classification, dossier_id)
        except Exception as error:
            result = {
                "dossier_id": dossier_id,
                "doc_id": classification["doc_id"],
                "doc_type": classification.get("doc_type"),
                "template_id": template_id,
                "original_path": classification.get("original_path"),
                "ocr_path": classification.get("ocr_path"),
                "fields": [],
                "extraction_status": "ERROR",
                "error": str(error),
            }
        save_json(result, destination / f"{source_stem}.json")
        results.append(result)

    summary = {
        "total_processed": len(results),
        "ok": sum(item.get("extraction_status") == "OK" for item in results),
        "to_review": sum(item.get("extraction_status") == "TO_REVIEW" for item in results),
        "incomplete": sum(item.get("extraction_status") == "INCOMPLETE" for item in results),
        "not_implemented": sum(item.get("extraction_status") == "NOT_IMPLEMENTED" for item in results),
        "skipped_not_implemented": skipped,
        "errors": sum(item.get("extraction_status") == "ERROR" for item in results),
    }
    print("\n===== RÉSUMÉ EXTRACTION =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return {"dossier_id": dossier_id, "results": results, "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Routeur des extracteurs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--only-template")
    parser.add_argument("--only-implemented", action="store_true")
    args = parser.parse_args()
    process_manifest(args.manifest, args.output_dir, args.only_template, args.only_implemented)
