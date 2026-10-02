from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from graph.neo4j_client import Neo4jClient


DIRECT_FIELD_MAP = {
    "cin_numero": "CIN",
    "cin": "CIN",
    "adresse": "ADRESSE",
    "banque": "BANQUE",
    "rib": "RIB",
    "iban": "IBAN",
    "employeur": "EMPLOYEUR",
    "poste": "POSTE",
    "date_embauche": "DATE_EMBAUCHE",
    "periode": "PERIODE",
    "net_a_payer": "NET_A_PAYER",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"dossier.json introuvable : {json_path}")
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not data.get("dossier_id"):
        raise ValueError("Le fichier ne contient pas un dossier_id valide.")
    if not isinstance(data.get("documents"), list):
        raise ValueError("Le fichier ne contient pas une liste documents valide.")
    return data


def _field_map(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(field.get("field_name", "")): field
        for field in document.get("fields", [])
        if isinstance(field, dict) and field.get("field_name")
    }


def _combined_field(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not first and not second:
        return None

    raw_parts = [
        str(item.get("raw_value", "")).strip()
        for item in (first, second)
        if item and item.get("raw_value") not in (None, "")
    ]
    normalized_parts = [
        str(item.get("normalized_value", "")).strip()
        for item in (first, second)
        if item and item.get("normalized_value") not in (None, "")
    ]
    if not normalized_parts:
        return None

    confidences = [
        float(item["confidence"])
        for item in (first, second)
        if item and item.get("confidence") is not None
    ]
    return {
        "raw_value": " ".join(raw_parts) or None,
        "normalized_value": " ".join(normalized_parts),
        "confidence": min(confidences) if confidences else None,
        "validation_status": "VALID",
        "evidence": {
            "texts": raw_parts,
        },
    }


def _canonical_fields(document: dict[str, Any]) -> list[dict[str, Any]]:
    source_fields = _field_map(document)
    doc_type = str(document.get("doc_type", "")).upper()
    canonical: dict[str, dict[str, Any]] = {}

    # Conserver les champs simples, avec les noms canoniques utilisés par Neo4j.
    for source_name, field in source_fields.items():
        if source_name == "transactions":
            continue
        canonical_name = DIRECT_FIELD_MAP.get(source_name, source_name.upper())
        canonical.setdefault(canonical_name, {**field, "source_field": source_name})

    # Construire NOM_COMPLET de la même manière pour tous les templates.
    full_name: dict[str, Any] | None = None
    if doc_type == "CIN":
        full_name = _combined_field(source_fields.get("nom"), source_fields.get("prenom"))
    elif doc_type == "BULLETIN_SALAIRE":
        full_name = _combined_field(
            source_fields.get("nom_employe"),
            source_fields.get("prenom_employe"),
        )
    elif doc_type == "QUITTANCE":
        full_name = source_fields.get("nom_abonne")
    elif doc_type in {"RIB", "RELEVE_BANCAIRE"}:
        full_name = source_fields.get("titulaire")
    elif doc_type == "ATTESTATION_TRAVAIL":
        full_name = source_fields.get("nom")

    if full_name and full_name.get("normalized_value") not in (None, ""):
        canonical["NOM_COMPLET"] = {**full_name, "source_field": "nom_complet_construit"}

    rows: list[dict[str, Any]] = []
    doc_id = str(document["doc_id"])
    for canonical_name, field in canonical.items():
        raw_value = field.get("raw_value")
        normalized_value = field.get("normalized_value")
        if raw_value is None and normalized_value is None:
            continue

        evidence = field.get("evidence") or {}
        texts = evidence.get("texts") if isinstance(evidence, dict) else None
        rows.append({
            "champ_id": f"{doc_id}__{canonical_name}",
            "doc_id": doc_id,
            "nom": canonical_name,
            "nom_source": field.get("source_field"),
            "valeur": raw_value,
            "valeur_normalisee": normalized_value,
            "confidence_ocr": field.get("confidence"),
            "validation_status": field.get("validation_status"),
            "evidence_texts": texts if isinstance(texts, list) else [],
        })
    return rows


def _extract_month_slot(doc_id: str) -> str | None:
    match = re.search(r"_M([123])(?:_|$)", doc_id, flags=re.IGNORECASE)
    return f"M{match.group(1)}" if match else None


def _field_value(document: dict[str, Any], field_name: str) -> Any:
    for field in document.get("fields", []):
        if field.get("field_name") == field_name:
            return field.get("normalized_value")
    return None


def _period_from_closing_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, date_format).strftime("%Y-%m")
        except ValueError:
            continue
    match = re.search(r"(\d{4})[-/](\d{2})", text)
    return f"{match.group(1)}-{match.group(2)}" if match else None


def _salary_transaction_info(label: Any, direction: Any) -> tuple[bool, str | None]:
    text = " ".join(str(label or "").upper().split())
    direction_normalized = str(direction or "").upper().strip()
    salary_pattern = re.compile(r"\b(?:SALAIRE|PAIE|PAYE)\b")
    is_salary = direction_normalized == "C" and salary_pattern.search(text) is not None
    if not is_salary:
        return False, None

    employer = re.sub(
        r"^\s*\d*\s*(?:VIR(?:EMENT)?\s+)?(?:SALAIRE|PAIE|PAYE)\s*",
        "",
        text,
    ).strip()
    return True, employer or None


def _transactions(document: dict[str, Any]) -> list[dict[str, Any]]:
    if str(document.get("doc_type", "")).upper() != "RELEVE_BANCAIRE":
        return []

    source_transactions = _field_value(document, "transactions")
    if not isinstance(source_transactions, list):
        return []

    doc_id = str(document["doc_id"])
    month_slot = _extract_month_slot(doc_id)
    period = _period_from_closing_date(_field_value(document, "date_solde_cloture"))
    rows: list[dict[str, Any]] = []

    for index, transaction in enumerate(source_transactions, start=1):
        if not isinstance(transaction, dict):
            continue
        is_salary, employer = _salary_transaction_info(
            transaction.get("libelle"),
            transaction.get("sens"),
        )
        rows.append({
            "transaction_id": f"{doc_id}__TX{index:03d}",
            "doc_id": doc_id,
            "dossier_id": document.get("dossier_id"),
            "mois_dossier": month_slot,
            "periode": period,
            "date": transaction.get("date"),
            "libelle": transaction.get("libelle"),
            "montant": transaction.get("montant"),
            "sens": transaction.get("sens"),
            "est_salaire": bool(is_salary),
            "employeur_virement": employer,
            "confidence_ocr": transaction.get("ocr_confidence"),
        })
    return rows


def prepare_graph_payload(dossier: dict[str, Any]) -> dict[str, Any]:
    dossier_id = str(dossier["dossier_id"])
    documents: list[dict[str, Any]] = []
    fields: list[dict[str, Any]] = []
    transactions: list[dict[str, Any]] = []

    for document in dossier["documents"]:
        doc_id = str(document["doc_id"])
        documents.append({
            "doc_id": doc_id,
            "dossier_id": dossier_id,
            "doc_type": str(document.get("doc_type", "")).upper(),
            "template_id": document.get("template_id"),
            "original_path": document.get("original_path"),
            "validation_status": document.get("validation_status"),
            "mois_dossier": _extract_month_slot(doc_id),
        })
        document_copy = {**document, "dossier_id": dossier_id}
        fields.extend(_canonical_fields(document_copy))
        transactions.extend(_transactions(document_copy))

    return {
        "dossier": {
            "dossier_id": dossier_id,
            "ready_for_signals": bool(
                dossier.get("data_quality", {}).get("ready_for_signals", False)
            ),
            "validation_status": dossier.get("data_quality", {}).get("status"),
            "document_count": len(documents),
        },
        "documents": documents,
        "fields": fields,
        "transactions": transactions,
    }


def _create_constraints(client: Neo4jClient) -> None:
    statements = [
        "CREATE CONSTRAINT dossier_id_unique IF NOT EXISTS "
        "FOR (d:Dossier) REQUIRE d.dossier_id IS UNIQUE",
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
        "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
        "CREATE CONSTRAINT champ_id_unique IF NOT EXISTS "
        "FOR (c:Champ) REQUIRE c.champ_id IS UNIQUE",
        "CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS "
        "FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE",
        "CREATE CONSTRAINT signal_id_unique IF NOT EXISTS "
        "FOR (s:SignalFraude) REQUIRE s.signal_id IS UNIQUE",
    ]
    for statement in statements:
        client.driver.execute_query(statement, database_=client.database)


def _delete_existing_dossier(client: Neo4jClient, dossier_id: str) -> None:
    queries = [
        "MATCH (d:Dossier {dossier_id: $dossier_id})-[:A_SIGNAL]->(s:SignalFraude) "
        "DETACH DELETE s",
        "MATCH (d:Dossier {dossier_id: $dossier_id})-[:CONTIENT]->(:Document)"
        "-[:POSSEDE]->(c:Champ) DETACH DELETE c",
        "MATCH (d:Dossier {dossier_id: $dossier_id})-[:CONTIENT]->(:Document)"
        "-[:CONTIENT_TRANSACTION]->(t:Transaction) DETACH DELETE t",
        "MATCH (d:Dossier {dossier_id: $dossier_id})-[:CONTIENT]->(doc:Document) "
        "DETACH DELETE doc",
        "MATCH (d:Dossier {dossier_id: $dossier_id}) DETACH DELETE d",
    ]
    for query in queries:
        client.driver.execute_query(
            query,
            dossier_id=dossier_id,
            database_=client.database,
        )


def load_payload(client: Neo4jClient, payload: dict[str, Any]) -> dict[str, int]:
    dossier = payload["dossier"]
    dossier_id = dossier["dossier_id"]
    _create_constraints(client)
    _delete_existing_dossier(client, dossier_id)

    client.driver.execute_query(
        "MERGE (d:Dossier {dossier_id: $dossier_id}) SET d += $properties",
        dossier_id=dossier_id,
        properties=dossier,
        database_=client.database,
    )

    if payload["documents"]:
        client.driver.execute_query(
            """
            MATCH (d:Dossier {dossier_id: $dossier_id})
            UNWIND $rows AS row
            MERGE (doc:Document {doc_id: row.doc_id})
            SET doc += row
            MERGE (d)-[:CONTIENT]->(doc)
            """,
            dossier_id=dossier_id,
            rows=payload["documents"],
            database_=client.database,
        )

    if payload["fields"]:
        client.driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (doc:Document {doc_id: row.doc_id})
            MERGE (c:Champ {champ_id: row.champ_id})
            SET c += row
            MERGE (doc)-[:POSSEDE]->(c)
            """,
            rows=payload["fields"],
            database_=client.database,
        )

    if payload["transactions"]:
        client.driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (doc:Document {doc_id: row.doc_id})
            MERGE (t:Transaction {transaction_id: row.transaction_id})
            SET t += row
            MERGE (doc)-[:CONTIENT_TRANSACTION]->(t)
            """,
            rows=payload["transactions"],
            database_=client.database,
        )

    return {
        "documents": len(payload["documents"]),
        "fields": len(payload["fields"]),
        "transactions": len(payload["transactions"]),
    }


def load_dossier_json(
    dossier_json: str | Path,
    client: Neo4jClient | None = None,
) -> dict[str, Any]:
    dossier = _load_json(dossier_json)
    payload = prepare_graph_payload(dossier)
    owns_client = client is None
    neo4j_client = client or Neo4jClient()
    try:
        if owns_client:
            neo4j_client.verifier_connexion()
        counts = load_payload(neo4j_client, payload)
        return {
            "dossier_id": dossier["dossier_id"],
            **counts,
        }
    finally:
        if owns_client:
            neo4j_client.fermer()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Charge un dossier.json validé dans Neo4j."
    )
    parser.add_argument("--dossier-json", required=True)
    args = parser.parse_args()

    result = load_dossier_json(args.dossier_json)
    print("\n===== CHARGEMENT NEO4J =====")
    print(f"Dossier : {result['dossier_id']}")
    print(f"Documents : {result['documents']}")
    print(f"Champs : {result['fields']}")
    print(f"Transactions : {result['transactions']}")


if __name__ == "__main__":
    main()
