from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from graph.neo4j_client import Neo4jClient


SOURCE = "INTRA_DOSSIER_V2"

FIELD_RULES: dict[str, dict[str, Any]] = {
    "CIN": {
        "type": "exact",
        "critical": True,
        "documents": {"CIN", "ATTESTATION_TRAVAIL"},
        "reference_priority": ["CIN", "ATTESTATION_TRAVAIL"],
    },
    "RIB": {
        "type": "exact",
        "critical": True,
        "documents": {"RIB", "RELEVE_BANCAIRE"},
        "reference_priority": ["RIB", "RELEVE_BANCAIRE"],
    },
    "IBAN": {
        "type": "exact",
        "critical": True,
        "documents": {"RIB", "RELEVE_BANCAIRE"},
        "reference_priority": ["RIB", "RELEVE_BANCAIRE"],
    },
    "BANQUE": {
        "type": "exact",
        "critical": True,
        "documents": {"RIB", "RELEVE_BANCAIRE"},
        "reference_priority": ["RIB", "RELEVE_BANCAIRE"],
    },
    "NOM_COMPLET": {
        "type": "text",
        "critical": True,
        "documents": {
            "CIN", "QUITTANCE", "RIB", "RELEVE_BANCAIRE",
            "BULLETIN_SALAIRE", "ATTESTATION_TRAVAIL",
        },
        "reference_priority": [
            "CIN", "ATTESTATION_TRAVAIL", "RIB", "QUITTANCE",
            "RELEVE_BANCAIRE", "BULLETIN_SALAIRE",
        ],
    },
    "EMPLOYEUR": {
        "type": "text",
        "critical": False,
        "documents": {"BULLETIN_SALAIRE", "ATTESTATION_TRAVAIL"},
        "reference_priority": ["ATTESTATION_TRAVAIL", "BULLETIN_SALAIRE"],
    },
    "POSTE": {
        "type": "text",
        "critical": False,
        "documents": {"BULLETIN_SALAIRE", "ATTESTATION_TRAVAIL"},
        "reference_priority": ["ATTESTATION_TRAVAIL", "BULLETIN_SALAIRE"],
    },
    "DATE_EMBAUCHE": {
        "type": "date",
        "critical": False,
        "documents": {"BULLETIN_SALAIRE", "ATTESTATION_TRAVAIL"},
        "reference_priority": ["ATTESTATION_TRAVAIL", "BULLETIN_SALAIRE"],
    },
    "ADRESSE": {
        "type": "address",
        "critical": False,
        "documents": {"CIN", "QUITTANCE", "RIB", "RELEVE_BANCAIRE"},
        "reference_priority": ["CIN", "QUITTANCE", "RELEVE_BANCAIRE", "RIB"],
    },
}


def _remove_accents(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or ""))
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def normalize_text(value: Any) -> str:
    text = _remove_accents(value).upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_identifier(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", _remove_accents(value).upper())


def normalize_address(value: Any) -> str:
    text = normalize_text(value)
    corrections = [
        (r"\bLOTISSEMENTAL\b", "LOTISSEMENT AL"),
        (r"\bAL\s+LAL\b", "ALLAL"),
        (r"\bAI\b", "AL"),
        (r"\bOODS\b", "QODS"),
        (r"\bMAARIE\b", "MAARIF"),
    ]
    for error, correction in corrections:
        text = re.sub(error, correction, text)
    return " ".join(text.split())


def text_similarity(first: Any, second: Any, *, address: bool = False) -> float:
    normalizer = normalize_address if address else normalize_text
    first_text = normalizer(first)
    second_text = normalizer(second)
    if not first_text or not second_text:
        return 0.0

    first_tokens = set(first_text.split())
    second_tokens = set(second_text.split())
    if address:
        first_numbers = {token for token in first_tokens if token.isdigit()}
        second_numbers = {token for token in second_tokens if token.isdigit()}
        if first_numbers and second_numbers and first_numbers != second_numbers:
            return 0.0

    intersection = first_tokens & second_tokens
    union = first_tokens | second_tokens
    containment = len(intersection) / min(len(first_tokens), len(second_tokens))
    jaccard = len(intersection) / len(union)
    ordered_first = " ".join(sorted(first_tokens))
    ordered_second = " ".join(sorted(second_tokens))
    sequence = SequenceMatcher(None, ordered_first, ordered_second).ratio()
    return round(max(containment, jaccard, sequence), 6)


def text_anomaly_score(similarity: float) -> float:
    if similarity >= 0.95:
        return 0.0
    if similarity >= 0.85:
        return 0.25
    if similarity >= 0.70:
        return 0.60
    return 1.0


def amount_anomaly_score(reference: float, compared: float) -> tuple[float, float]:
    denominator = max(abs(reference), 1.0)
    relative_difference = abs(reference - compared) / denominator
    if relative_difference <= 0.02:
        score = 0.0
    elif relative_difference <= 0.05:
        score = 0.25
    elif relative_difference <= 0.10:
        score = 0.50
    elif relative_difference <= 0.20:
        score = 0.75
    else:
        score = 1.0
    return score, round(relative_difference, 6)


def _parse_date(value: Any) -> datetime | None:
    text = str(value or "").strip()
    for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue
    return None


def date_anomaly_score(reference: Any, compared: Any) -> tuple[float, int | None]:
    reference_date = _parse_date(reference)
    compared_date = _parse_date(compared)
    if reference_date is None or compared_date is None:
        return 1.0, None
    difference_days = abs((reference_date - compared_date).days)
    if difference_days == 0:
        score = 0.0
    elif difference_days <= 31:
        score = 0.25
    elif difference_days <= 365:
        score = 0.60
    else:
        score = 1.0
    return score, difference_days


def _record_to_proof(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "document": record.get("document"),
        "type_document": record.get("type_document"),
        "field_name": record.get("champ"),
        "raw_value": record.get("raw_value"),
        "normalized_value": record.get("normalized_value"),
    }


def _priority_index(doc_type: str, priorities: list[str]) -> int:
    try:
        return priorities.index(doc_type)
    except ValueError:
        return len(priorities)


def _choose_majority_reference(
    records: list[dict[str, Any]],
    rule: dict[str, Any],
) -> dict[str, Any]:
    normalized = [normalize_identifier(record["normalized_value"]) for record in records]
    counts = Counter(value for value in normalized if value)
    most_common_count = counts.most_common(1)[0][1]
    candidates = {
        value for value, count in counts.items() if count == most_common_count
    }
    candidate_records = [
        record for record in records
        if normalize_identifier(record["normalized_value"]) in candidates
    ]
    priorities = rule.get("reference_priority", [])
    return min(
        candidate_records,
        key=lambda record: (
            _priority_index(record["type_document"], priorities),
            str(record["document"]),
        ),
    )


def _choose_text_reference(
    records: list[dict[str, Any]],
    rule: dict[str, Any],
    *,
    address: bool,
) -> dict[str, Any]:
    # Priorité métier d'abord; en cas de plusieurs documents du même type,
    # choisir celui qui ressemble le plus aux autres.
    priorities = rule.get("reference_priority", [])
    best_priority = min(
        _priority_index(record["type_document"], priorities)
        for record in records
    )
    candidates = [
        record for record in records
        if _priority_index(record["type_document"], priorities) == best_priority
    ]
    if len(candidates) == 1:
        return candidates[0]

    def average_similarity(candidate: dict[str, Any]) -> float:
        scores = [
            text_similarity(
                candidate["normalized_value"],
                other["normalized_value"],
                address=address,
            )
            for other in records
            if other["document"] != candidate["document"]
        ]
        return sum(scores) / len(scores) if scores else 1.0

    return max(candidates, key=average_similarity)


def _unavailable_check(rule_id: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "check_id": rule_id,
        "available": False,
        "score": None,
        "status": "UNAVAILABLE",
        "reason": reason,
        **extra,
    }


def _evaluate_field_rule(
    field_name: str,
    records: list[dict[str, Any]],
    rule: dict[str, Any],
) -> dict[str, Any]:
    usable = [
        record for record in records
        if record.get("type_document") in rule["documents"]
        and record.get("normalized_value") not in (None, "")
    ]
    if len({record["document"] for record in usable}) < 2:
        return _unavailable_check(
            field_name,
            "Moins de deux documents contiennent une valeur exploitable.",
            field_name=field_name,
            observed_documents=sorted({record["document"] for record in usable}),
        )

    rule_type = rule["type"]
    address = rule_type == "address"
    if rule_type == "exact":
        reference = _choose_majority_reference(usable, rule)
    else:
        reference = _choose_text_reference(usable, rule, address=address)

    comparisons: list[dict[str, Any]] = []
    suspect_fields: list[dict[str, Any]] = []
    scores: list[float] = []

    for record in usable:
        if record["document"] == reference["document"]:
            continue

        details: dict[str, Any] = {}
        if rule_type == "exact":
            same = normalize_identifier(reference["normalized_value"]) == normalize_identifier(
                record["normalized_value"]
            )
            score = 0.0 if same else 1.0
        elif rule_type in {"text", "address"}:
            similarity = text_similarity(
                reference["normalized_value"],
                record["normalized_value"],
                address=address,
            )
            score = text_anomaly_score(similarity)
            details["similarity"] = round(similarity, 4)
        elif rule_type == "date":
            score, difference_days = date_anomaly_score(
                reference["normalized_value"],
                record["normalized_value"],
            )
            details["difference_days"] = difference_days
        else:
            raise ValueError(f"Type de règle inconnu : {rule_type}")

        comparison = {
            **_record_to_proof(record),
            "score": score,
            **details,
        }
        comparisons.append(comparison)
        scores.append(score)
        if score > 0:
            suspect_fields.append(comparison)

    check_score = max(scores, default=0.0)
    return {
        "check_id": field_name,
        "field_name": field_name,
        "available": True,
        "score": check_score,
        "status": "ANOMALY" if check_score > 0 else "NORMAL",
        "critical": bool(rule.get("critical") and check_score > 0),
        "reference_fields": [_record_to_proof(reference)],
        "suspect_fields": suspect_fields,
        "comparisons": comparisons,
        "message": (
            f"Valeurs incohérentes pour {field_name}."
            if check_score > 0
            else f"Valeurs cohérentes pour {field_name}."
        ),
    }


def _evaluate_salary_checks(
    bulletins: list[dict[str, Any]],
    salary_transactions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bulletin_groups: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for bulletin in bulletins:
        bulletin_groups[bulletin.get("mois_dossier")].append(bulletin)

    transaction_groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for transaction in salary_transactions:
        transaction_groups[
            (transaction.get("mois_dossier"), transaction.get("periode"))
        ].append(transaction)

    checks: list[dict[str, Any]] = []
    for month_slot in ("M1", "M2", "M3"):
        check_id = f"SALAIRE_{month_slot}"
        month_bulletins = bulletin_groups.get(month_slot, [])
        if len(month_bulletins) == 0:
            checks.append(_unavailable_check(
                check_id,
                "Bulletin du mois absent ou net à payer non exploitable.",
                field_name="NET_A_PAYER",
                month_slot=month_slot,
            ))
            continue
        if len(month_bulletins) > 1:
            checks.append(_unavailable_check(
                check_id,
                "Plusieurs bulletins correspondent au même mois; comparaison ambiguë.",
                field_name="NET_A_PAYER",
                month_slot=month_slot,
                documents=[row.get("document") for row in month_bulletins],
            ))
            continue

        bulletin = month_bulletins[0]
        period = bulletin.get("periode")
        if period in (None, ""):
            checks.append(_unavailable_check(
                check_id,
                "Période du bulletin absente ou inexploitable.",
                field_name="NET_A_PAYER",
                month_slot=month_slot,
            ))
            continue

        transactions = transaction_groups.get((month_slot, period), [])
        if bulletin.get("salaire_bulletin") is None:
            checks.append(_unavailable_check(
                check_id,
                "Net à payer absent dans le bulletin.",
                field_name="NET_A_PAYER",
                month_slot=month_slot,
                period=period,
            ))
            continue
        if len(transactions) == 0:
            checks.append(_unavailable_check(
                check_id,
                "Aucun virement salaire correspondant au même mois et à la même période.",
                field_name="NET_A_PAYER",
                month_slot=month_slot,
                period=period,
            ))
            continue
        if len(transactions) > 1:
            checks.append(_unavailable_check(
                check_id,
                "Plusieurs virements salaire correspondent au même mois; comparaison ambiguë.",
                field_name="NET_A_PAYER",
                month_slot=month_slot,
                period=period,
                transaction_ids=[row.get("transaction_id") for row in transactions],
            ))
            continue

        transaction = transactions[0]
        reference_amount = float(bulletin["salaire_bulletin"])
        compared_amount = float(transaction["salaire_releve"])
        score, relative_difference = amount_anomaly_score(
            reference_amount,
            compared_amount,
        )
        bulletin_proof = {
            "document": bulletin.get("document"),
            "type_document": "BULLETIN_SALAIRE",
            "field_name": "NET_A_PAYER",
            "raw_value": bulletin.get("raw_value"),
            "normalized_value": reference_amount,
        }
        transaction_proof = {
            "document": transaction.get("document"),
            "type_document": "RELEVE_BANCAIRE",
            "transaction_id": transaction.get("transaction_id"),
            "field_name": "MONTANT_VIREMENT_SALAIRE",
            "raw_value": transaction.get("salaire_releve"),
            "normalized_value": compared_amount,
            "libelle": transaction.get("libelle"),
        }
        checks.append({
            "check_id": check_id,
            "field_name": "NET_A_PAYER",
            "available": True,
            "score": score,
            "status": "ANOMALY" if score > 0 else "NORMAL",
            "critical": False,
            "month_slot": month_slot,
            "period": period,
            "relative_difference": relative_difference,
            "reference_fields": [bulletin_proof],
            "suspect_fields": [transaction_proof] if score > 0 else [],
            "comparisons": [{**transaction_proof, "score": score}],
            "message": (
                "Le net à payer diffère du virement salaire du même mois."
                if score > 0
                else "Le net à payer correspond au virement salaire du même mois."
            ),
        })
    return checks


def evaluate_intra_records(
    dossier_id: str,
    field_records: Iterable[dict[str, Any]],
    bulletins: Iterable[dict[str, Any]],
    salary_transactions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    grouped_fields: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in field_records:
        grouped_fields[str(record.get("champ"))].append(dict(record))

    checks: list[dict[str, Any]] = []
    for field_name, rule in FIELD_RULES.items():
        checks.append(
            _evaluate_field_rule(
                field_name,
                grouped_fields.get(field_name, []),
                rule,
            )
        )
    checks.extend(_evaluate_salary_checks(list(bulletins), list(salary_transactions)))

    available_checks = [check for check in checks if check["available"]]
    unavailable_checks = [check for check in checks if not check["available"]]
    anomalies = [check for check in available_checks if (check.get("score") or 0) > 0]
    score = max((float(check["score"]) for check in available_checks), default=None)

    return {
        "dossier_id": dossier_id,
        "signal_id": "INTRA_DOSSIER",
        "source": SOURCE,
        "available": bool(available_checks),
        "score": round(score, 4) if score is not None else None,
        "status": (
            "ANOMALY_DETECTED"
            if score is not None and score > 0
            else "NO_ANOMALY_DETECTED"
            if score == 0
            else "UNAVAILABLE"
        ),
        "scoring": {
            "range": [0, 1],
            "aggregation": "MAX_AVAILABLE_SUBSCORES",
            "ocr_confidence_used": False,
            "meaning": "Niveau d'anomalie du signal, pas probabilité de fraude.",
        },
        "checks": checks,
        "anomalies": anomalies,
        "checks_unavailable": unavailable_checks,
        "summary": {
            "checks_total": len(checks),
            "checks_available": len(available_checks),
            "checks_unavailable": len(unavailable_checks),
            "anomalies_count": len(anomalies),
            "critical_anomalies_count": sum(
                1 for anomaly in anomalies if anomaly.get("critical")
            ),
        },
    }


def _fetch_records(client: Neo4jClient, dossier_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    field_rows, _, _ = client.driver.execute_query(
        """
        MATCH (d:Dossier {dossier_id: $dossier_id})-[:CONTIENT]->(doc:Document)
              -[:POSSEDE]->(c:Champ)
        WHERE c.nom IN $fields
          AND c.valeur_normalisee IS NOT NULL
          AND trim(toString(c.valeur_normalisee)) <> ''
        RETURN
            doc.doc_id AS document,
            doc.doc_type AS type_document,
            c.nom AS champ,
            c.valeur AS raw_value,
            c.valeur_normalisee AS normalized_value
        ORDER BY champ, document
        """,
        dossier_id=dossier_id,
        fields=list(FIELD_RULES.keys()),
        database_=client.database,
    )

    bulletin_rows, _, _ = client.driver.execute_query(
        """
        MATCH (d:Dossier {dossier_id: $dossier_id})-[:CONTIENT]->(doc:Document)
              -[:POSSEDE]->(net:Champ)
        MATCH (doc)-[:POSSEDE]->(periode:Champ)
        WHERE doc.doc_type = 'BULLETIN_SALAIRE'
          AND net.nom = 'NET_A_PAYER'
          AND periode.nom = 'PERIODE'
        RETURN
            doc.doc_id AS document,
            doc.mois_dossier AS mois_dossier,
            periode.valeur_normalisee AS periode,
            net.valeur AS raw_value,
            toFloat(net.valeur_normalisee) AS salaire_bulletin
        ORDER BY mois_dossier
        """,
        dossier_id=dossier_id,
        database_=client.database,
    )

    transaction_rows, _, _ = client.driver.execute_query(
        """
        MATCH (d:Dossier {dossier_id: $dossier_id})-[:CONTIENT]->(doc:Document)
              -[:CONTIENT_TRANSACTION]->(t:Transaction)
        WHERE doc.doc_type = 'RELEVE_BANCAIRE'
          AND t.est_salaire = true
        RETURN
            doc.doc_id AS document,
            t.transaction_id AS transaction_id,
            t.mois_dossier AS mois_dossier,
            t.periode AS periode,
            t.libelle AS libelle,
            toFloat(t.montant) AS salaire_releve
        ORDER BY mois_dossier
        """,
        dossier_id=dossier_id,
        database_=client.database,
    )

    return (
        [dict(row) for row in field_rows],
        [dict(row) for row in bulletin_rows],
        [dict(row) for row in transaction_rows],
    )


def _delete_previous_signals(client: Neo4jClient, dossier_id: str) -> None:
    client.driver.execute_query(
        """
        MATCH (d:Dossier {dossier_id: $dossier_id})-[:A_SIGNAL]->(s:SignalFraude)
        WHERE s.source = $source
        DETACH DELETE s
        """,
        dossier_id=dossier_id,
        source=SOURCE,
        database_=client.database,
    )


def _persist_result(client: Neo4jClient, result: dict[str, Any]) -> None:
    dossier_id = result["dossier_id"]
    _delete_previous_signals(client, dossier_id)

    client.driver.execute_query(
        """
        MATCH (d:Dossier {dossier_id: $dossier_id})
        SET
            d.score_intra_dossier = $score,
            d.intra_dossier_available = $available,
            d.intra_dossier_status = $status,
            d.nombre_anomalies_intra = $anomalies_count,
            d.nombre_anomalies_critiques_intra = $critical_count
        """,
        dossier_id=dossier_id,
        score=result["score"],
        available=result["available"],
        status=result["status"],
        anomalies_count=result["summary"]["anomalies_count"],
        critical_count=result["summary"]["critical_anomalies_count"],
        database_=client.database,
    )

    for anomaly in result["anomalies"]:
        signal_id = f"INTRA_{dossier_id}_{anomaly['check_id']}"
        reference_fields = anomaly.get("reference_fields", [])
        suspect_fields = anomaly.get("suspect_fields", [])
        reference_documents = sorted({
            str(field.get("document")) for field in reference_fields if field.get("document")
        })
        suspect_documents = sorted({
            str(field.get("document")) for field in suspect_fields if field.get("document")
        })

        client.driver.execute_query(
            """
            MATCH (d:Dossier {dossier_id: $dossier_id})
            MERGE (s:SignalFraude {signal_id: $signal_id})
            SET
                s.type = $type_signal,
                s.famille = 'COHERENCE_INTRA_DOSSIER',
                s.champ = $champ,
                s.score = $score,
                s.critical = $critical,
                s.description = $description,
                s.documents_reference = $reference_documents,
                s.documents_suspects = $suspect_documents,
                s.preuves_json = $preuves_json,
                s.statut = 'DETECTE',
                s.source = $source
            MERGE (d)-[:A_SIGNAL]->(s)
            """,
            dossier_id=dossier_id,
            signal_id=signal_id,
            type_signal=anomaly["check_id"],
            champ=anomaly.get("field_name"),
            score=float(anomaly["score"]),
            critical=bool(anomaly.get("critical")),
            description=anomaly.get("message"),
            reference_documents=reference_documents,
            suspect_documents=suspect_documents,
            preuves_json=json.dumps(
                {
                    "reference_fields": reference_fields,
                    "suspect_fields": suspect_fields,
                    "comparisons": anomaly.get("comparisons", []),
                },
                ensure_ascii=False,
            ),
            source=SOURCE,
            database_=client.database,
        )

        for document_id in reference_documents:
            client.driver.execute_query(
                """
                MATCH (s:SignalFraude {signal_id: $signal_id})
                MATCH (doc:Document {doc_id: $document_id})
                MERGE (s)-[:DOCUMENT_REFERENCE]->(doc)
                """,
                signal_id=signal_id,
                document_id=document_id,
                database_=client.database,
            )
        for document_id in suspect_documents:
            client.driver.execute_query(
                """
                MATCH (s:SignalFraude {signal_id: $signal_id})
                MATCH (doc:Document {doc_id: $document_id})
                MERGE (s)-[:DOCUMENT_SUSPECT]->(doc)
                """,
                signal_id=signal_id,
                document_id=document_id,
                database_=client.database,
            )

        transaction_ids = sorted({
            str(field.get("transaction_id"))
            for field in suspect_fields
            if field.get("transaction_id")
        })
        for transaction_id in transaction_ids:
            client.driver.execute_query(
                """
                MATCH (s:SignalFraude {signal_id: $signal_id})
                MATCH (t:Transaction {transaction_id: $transaction_id})
                MERGE (s)-[:TRANSACTION_SUSPECTE]->(t)
                """,
                signal_id=signal_id,
                transaction_id=transaction_id,
                database_=client.database,
            )


def run_intra_dossier_signal(
    dossier_id: str,
    output_root: str | Path = "data/signals",
    client: Neo4jClient | None = None,
) -> dict[str, Any]:
    owns_client = client is None
    neo4j_client = client or Neo4jClient()
    try:
        if owns_client:
            neo4j_client.verifier_connexion()
        fields, bulletins, transactions = _fetch_records(neo4j_client, dossier_id)
        result = evaluate_intra_records(
            dossier_id,
            fields,
            bulletins,
            transactions,
        )
        _persist_result(neo4j_client, result)

        output_dir = Path(output_root) / dossier_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "intra_dossier.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        result["output_path"] = str(output_path)
        return result
    finally:
        if owns_client:
            neo4j_client.fermer()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcule le signal de cohérence intra-dossier depuis Neo4j."
    )
    parser.add_argument("--dossier-id", required=True)
    parser.add_argument("--output-root", default="data/signals")
    args = parser.parse_args()

    result = run_intra_dossier_signal(args.dossier_id, args.output_root)
    print("\n===== SIGNAL INTRA-DOSSIER =====")
    print(f"Dossier : {result['dossier_id']}")
    print(f"Disponible : {result['available']}")
    print(f"Score : {result['score']}")
    print(f"Anomalies : {result['summary']['anomalies_count']}")
    print(f"Contrôles indisponibles : {result['summary']['checks_unavailable']}")
    print(f"Résultat : {result['output_path']}")


if __name__ == "__main__":
    main()
