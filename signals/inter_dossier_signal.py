from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graph.neo4j_client import Neo4jClient
from signals.intra_dossier_signal import (
    normalize_identifier,
    normalize_text,
    text_anomaly_score,
    text_similarity,
)


SOURCE = "INTER_DOSSIERS_V1"
KEY_FIELDS = ("CIN", "NOM_COMPLET", "RIB", "IBAN")


def is_test_dossier_id(dossier_id: Any) -> bool:
    value = str(dossier_id or "").upper().strip()
    return (
        value == "D_TEST"
        or value.endswith("_D_TEST")
        or value.startswith("TEST_")
    )


def _normalize_field(field_name: str, value: Any) -> str:
    if field_name == "NOM_COMPLET":
        return normalize_text(value)
    return normalize_identifier(value)


def _empty_profile(dossier_id: str) -> dict[str, Any]:
    return {
        "dossier_id": dossier_id,
        "fields": {field_name: {} for field_name in KEY_FIELDS},
    }


def _build_profiles(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}

    for row in rows:
        dossier_id = str(row.get("dossier_id") or "")
        if not dossier_id:
            continue
        profile = profiles.setdefault(dossier_id, _empty_profile(dossier_id))
        field_name = str(row.get("field_name") or "").upper()
        if field_name not in KEY_FIELDS:
            continue

        normalized_value = _normalize_field(field_name, row.get("normalized_value"))
        if not normalized_value:
            continue

        values = profile["fields"][field_name]
        value_entry = values.setdefault(
            normalized_value,
            {
                "normalized_value": normalized_value,
                "raw_values": [],
                "documents": [],
                "document_types": [],
            },
        )

        raw_value = row.get("raw_value")
        if raw_value not in (None, "") and str(raw_value) not in value_entry["raw_values"]:
            value_entry["raw_values"].append(str(raw_value))

        document = row.get("document")
        if document and str(document) not in value_entry["documents"]:
            value_entry["documents"].append(str(document))

        doc_type = row.get("type_document")
        if doc_type and str(doc_type) not in value_entry["document_types"]:
            value_entry["document_types"].append(str(doc_type))

    return profiles


def fetch_inter_profiles(
    client: Neo4jClient,
    *,
    include_test_dossiers: bool = False,
) -> dict[str, dict[str, Any]]:
    rows, _, _ = client.driver.execute_query(
        """
        MATCH (d:Dossier)
        OPTIONAL MATCH (d)-[:CONTIENT]->(doc:Document)-[:POSSEDE]->(c:Champ)
        WHERE c.nom IN $fields
          AND c.valeur_normalisee IS NOT NULL
          AND trim(toString(c.valeur_normalisee)) <> ''
        RETURN
            d.dossier_id AS dossier_id,
            doc.doc_id AS document,
            doc.doc_type AS type_document,
            c.nom AS field_name,
            c.valeur AS raw_value,
            c.valeur_normalisee AS normalized_value
        ORDER BY dossier_id, field_name, document
        """,
        fields=list(KEY_FIELDS),
        database_=client.database,
    )

    profiles = _build_profiles([dict(row) for row in rows])
    if include_test_dossiers:
        return profiles
    return {
        dossier_id: profile
        for dossier_id, profile in profiles.items()
        if not is_test_dossier_id(dossier_id)
    }


def _values(profile: dict[str, Any], field_name: str) -> set[str]:
    return set(profile.get("fields", {}).get(field_name, {}).keys())


def _evidence(
    profile: dict[str, Any],
    field_name: str,
    selected_values: set[str] | None = None,
) -> list[dict[str, Any]]:
    entries = profile.get("fields", {}).get(field_name, {})
    values = selected_values if selected_values is not None else set(entries.keys())
    return [entries[value] for value in sorted(values) if value in entries]


def _unavailable_check(check_id: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "available": False,
        "score": None,
        "status": "UNAVAILABLE",
        "critical": False,
        "reason": reason,
        **extra,
    }


def _normal_check(check_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "available": True,
        "score": 0.0,
        "status": "NORMAL",
        "critical": False,
        **extra,
    }


def _name_similarity_between_profiles(
    current_profile: dict[str, Any],
    linked_profile: dict[str, Any],
) -> tuple[float | None, dict[str, Any] | None]:
    current_names = sorted(_values(current_profile, "NOM_COMPLET"))
    linked_names = sorted(_values(linked_profile, "NOM_COMPLET"))
    if not current_names or not linked_names:
        return None, None

    candidates: list[tuple[float, str, str]] = []
    for current_name in current_names:
        for linked_name in linked_names:
            candidates.append((text_similarity(current_name, linked_name), current_name, linked_name))

    best_similarity, current_name, linked_name = max(candidates, key=lambda item: item[0])
    return best_similarity, {
        "current_name": current_name,
        "linked_name": linked_name,
    }


def _check_same_cin_multiple_names(
    current_profile: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    check_id = "CIN_MULTIPLE_NAMES"
    current_cins = _values(current_profile, "CIN")
    current_names = _values(current_profile, "NOM_COMPLET")
    if not current_cins:
        return _unavailable_check(check_id, "CIN absente dans le dossier courant.")
    if not current_names:
        return _unavailable_check(check_id, "Nom complet absent dans le dossier courant.")

    current_id = current_profile["dossier_id"]
    comparisons: list[dict[str, Any]] = []
    repeated_identity_dossiers: list[str] = []

    for linked_id, linked_profile in profiles.items():
        if linked_id == current_id:
            continue
        shared_cins = current_cins & _values(linked_profile, "CIN")
        if not shared_cins:
            continue

        similarity, name_pair = _name_similarity_between_profiles(current_profile, linked_profile)
        if similarity is None:
            comparisons.append({
                "linked_dossier_id": linked_id,
                "shared_cin": sorted(shared_cins),
                "score": None,
                "status": "UNRESOLVED_MISSING_LINKED_NAME",
                "linked_cin_evidence": _evidence(linked_profile, "CIN", shared_cins),
            })
            continue

        score = text_anomaly_score(similarity)
        status = "REPEATED_IDENTITY" if score == 0 else "ANOMALY"
        if score == 0:
            repeated_identity_dossiers.append(linked_id)
        comparisons.append({
            "linked_dossier_id": linked_id,
            "shared_cin": sorted(shared_cins),
            "similarity": round(similarity, 4),
            "score": score,
            "status": status,
            **(name_pair or {}),
            "current_cin_evidence": _evidence(current_profile, "CIN", shared_cins),
            "linked_cin_evidence": _evidence(linked_profile, "CIN", shared_cins),
            "current_name_evidence": _evidence(current_profile, "NOM_COMPLET"),
            "linked_name_evidence": _evidence(linked_profile, "NOM_COMPLET"),
        })

    scored = [item for item in comparisons if item.get("score") is not None]
    score = max((float(item["score"]) for item in scored), default=0.0)
    anomalies = [item for item in scored if float(item["score"]) > 0]
    return {
        "check_id": check_id,
        "available": True,
        "score": score,
        "status": "ANOMALY" if score > 0 else "NORMAL",
        "critical": score >= 1.0,
        "comparisons": comparisons,
        "linked_dossiers": sorted({item["linked_dossier_id"] for item in anomalies}),
        "repeated_identity_dossiers": sorted(set(repeated_identity_dossiers)),
        "message": (
            "Une même CIN est associée à des noms incohérents dans plusieurs dossiers."
            if score > 0
            else "Aucune incohérence de nom associée à la CIN."
        ),
    }


def _check_shared_bank_identifier(
    current_profile: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    *,
    field_name: str,
    check_id: str,
) -> dict[str, Any]:
    current_values = _values(current_profile, field_name)
    current_cins = _values(current_profile, "CIN")
    if not current_values:
        return _unavailable_check(
            check_id,
            f"{field_name} absent dans le dossier courant.",
            field_name=field_name,
        )
    if not current_cins:
        return _unavailable_check(
            check_id,
            "CIN absente : impossible de vérifier si l'identité est différente.",
            field_name=field_name,
        )

    current_id = current_profile["dossier_id"]
    comparisons: list[dict[str, Any]] = []

    for linked_id, linked_profile in profiles.items():
        if linked_id == current_id:
            continue
        shared_values = current_values & _values(linked_profile, field_name)
        if not shared_values:
            continue

        linked_cins = _values(linked_profile, "CIN")
        if not linked_cins:
            comparisons.append({
                "linked_dossier_id": linked_id,
                "shared_values": sorted(shared_values),
                "score": None,
                "status": "UNRESOLVED_MISSING_LINKED_CIN",
                "linked_identifier_evidence": _evidence(
                    linked_profile, field_name, shared_values
                ),
            })
            continue

        same_identity = bool(current_cins & linked_cins)
        score = 0.0 if same_identity else 1.0
        comparisons.append({
            "linked_dossier_id": linked_id,
            "shared_values": sorted(shared_values),
            "current_cins": sorted(current_cins),
            "linked_cins": sorted(linked_cins),
            "same_identity": same_identity,
            "score": score,
            "status": "REPEATED_IDENTITY" if same_identity else "ANOMALY",
            "current_identifier_evidence": _evidence(
                current_profile, field_name, shared_values
            ),
            "linked_identifier_evidence": _evidence(
                linked_profile, field_name, shared_values
            ),
            "current_cin_evidence": _evidence(current_profile, "CIN"),
            "linked_cin_evidence": _evidence(linked_profile, "CIN"),
        })

    scored = [item for item in comparisons if item.get("score") is not None]
    score = max((float(item["score"]) for item in scored), default=0.0)
    anomalies = [item for item in scored if float(item["score"]) > 0]
    return {
        "check_id": check_id,
        "field_name": field_name,
        "available": True,
        "score": score,
        "status": "ANOMALY" if score > 0 else "NORMAL",
        "critical": score > 0,
        "comparisons": comparisons,
        "linked_dossiers": sorted({item["linked_dossier_id"] for item in anomalies}),
        "message": (
            f"Le même {field_name} est associé à des CIN différentes."
            if score > 0
            else f"Aucun {field_name} partagé entre des identités différentes."
        ),
    }


def _name_group_score(distinct_cin_count: int) -> float:
    if distinct_cin_count <= 1:
        return 0.0
    if distinct_cin_count == 2:
        return 0.25
    if distinct_cin_count == 3:
        return 0.50
    return 0.60


def _check_same_name_multiple_cin(
    current_profile: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    check_id = "NAME_MULTIPLE_CIN"
    current_names = _values(current_profile, "NOM_COMPLET")
    current_cins = _values(current_profile, "CIN")
    if not current_names:
        return _unavailable_check(check_id, "Nom complet absent dans le dossier courant.")
    if not current_cins:
        return _unavailable_check(check_id, "CIN absente dans le dossier courant.")

    current_id = current_profile["dossier_id"]
    groups: list[dict[str, Any]] = []

    for current_name in sorted(current_names):
        linked_profiles: list[dict[str, Any]] = []
        distinct_cins = set(current_cins)

        for linked_id, linked_profile in profiles.items():
            if linked_id == current_id:
                continue
            if current_name not in _values(linked_profile, "NOM_COMPLET"):
                continue
            linked_cins = _values(linked_profile, "CIN")
            if not linked_cins:
                continue
            distinct_cins.update(linked_cins)
            linked_profiles.append({
                "linked_dossier_id": linked_id,
                "linked_cins": sorted(linked_cins),
                "name_evidence": _evidence(
                    linked_profile, "NOM_COMPLET", {current_name}
                ),
                "cin_evidence": _evidence(linked_profile, "CIN"),
            })

        score = _name_group_score(len(distinct_cins))
        groups.append({
            "shared_name": current_name,
            "distinct_cins": sorted(distinct_cins),
            "distinct_cin_count": len(distinct_cins),
            "score": score,
            "status": "ANOMALY" if score > 0 else "NORMAL",
            "current_name_evidence": _evidence(
                current_profile, "NOM_COMPLET", {current_name}
            ),
            "current_cin_evidence": _evidence(current_profile, "CIN"),
            "linked_profiles": linked_profiles,
        })

    score = max((float(group["score"]) for group in groups), default=0.0)
    anomalous_groups = [group for group in groups if float(group["score"]) > 0]
    linked_dossiers = sorted({
        linked["linked_dossier_id"]
        for group in anomalous_groups
        for linked in group["linked_profiles"]
    })
    return {
        "check_id": check_id,
        "available": True,
        "score": score,
        "status": "ANOMALY" if score > 0 else "NORMAL",
        "critical": False,
        "groups": groups,
        "linked_dossiers": linked_dossiers,
        "message": (
            "Le même nom complet est associé à plusieurs CIN."
            if score > 0
            else "Aucun nom complet associé à plusieurs CIN."
        ),
    }


def evaluate_inter_profiles(
    dossier_id: str,
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    current_profile = profiles.get(dossier_id)
    if current_profile is None:
        return {
            "dossier_id": dossier_id,
            "signal_id": "INTER_DOSSIERS",
            "source": SOURCE,
            "available": False,
            "score": None,
            "status": "DOSSIER_NOT_FOUND",
            "scoring": {
                "range": [0, 1],
                "aggregation": "MAX_AVAILABLE_SUBSCORES",
                "ocr_confidence_used": False,
                "meaning": "Niveau d'anomalie réseau, pas probabilité de fraude.",
            },
            "checks": [],
            "anomalies": [],
            "checks_unavailable": [],
            "summary": {
                "checks_total": 0,
                "checks_available": 0,
                "checks_unavailable": 0,
                "anomalies_count": 0,
                "critical_anomalies_count": 0,
                "linked_dossiers_count": 0,
            },
        }

    checks = [
        _check_same_cin_multiple_names(current_profile, profiles),
        _check_shared_bank_identifier(
            current_profile,
            profiles,
            field_name="RIB",
            check_id="RIB_MULTIPLE_IDENTITIES",
        ),
        _check_shared_bank_identifier(
            current_profile,
            profiles,
            field_name="IBAN",
            check_id="IBAN_MULTIPLE_IDENTITIES",
        ),
        _check_same_name_multiple_cin(current_profile, profiles),
    ]

    available_checks = [check for check in checks if check.get("available")]
    unavailable_checks = [check for check in checks if not check.get("available")]
    anomalies = [
        check
        for check in available_checks
        if check.get("score") is not None and float(check["score"]) > 0
    ]
    score = max(
        (float(check["score"]) for check in available_checks if check.get("score") is not None),
        default=None,
    )
    linked_dossiers = sorted({
        linked_id
        for check in checks
        for linked_id in check.get("linked_dossiers", [])
    })
    repeated_identity_dossiers = sorted({
        linked_id
        for check in checks
        for linked_id in check.get("repeated_identity_dossiers", [])
    })

    critical_count = sum(1 for anomaly in anomalies if anomaly.get("critical"))
    if not available_checks:
        status = "UNAVAILABLE"
    elif critical_count > 0:
        status = "CRITICAL_NETWORK_ANOMALY"
    elif anomalies:
        status = "NETWORK_ANOMALY_DETECTED"
    else:
        status = "NO_NETWORK_ANOMALY"

    return {
        "dossier_id": dossier_id,
        "signal_id": "INTER_DOSSIERS",
        "source": SOURCE,
        "available": bool(available_checks),
        "score": score,
        "status": status,
        "scoring": {
            "range": [0, 1],
            "aggregation": "MAX_AVAILABLE_SUBSCORES",
            "ocr_confidence_used": False,
            "meaning": "Niveau d'anomalie réseau, pas probabilité de fraude.",
            "rules": {
                "same_cin_different_name": {
                    "similarity_gte_0_95": 0.0,
                    "similarity_0_85_to_0_95": 0.25,
                    "similarity_0_70_to_0_85": 0.60,
                    "similarity_lt_0_70": 1.0,
                },
                "same_rib_different_cin": 1.0,
                "same_iban_different_cin": 1.0,
                "same_name_multiple_cin": {
                    "2_cin": 0.25,
                    "3_cin": 0.50,
                    "4_or_more_cin": 0.60,
                },
            },
        },
        "current_profile": {
            field_name: _evidence(current_profile, field_name)
            for field_name in KEY_FIELDS
        },
        "checks": checks,
        "anomalies": anomalies,
        "checks_unavailable": unavailable_checks,
        "linked_dossiers": linked_dossiers,
        "repeated_identity_dossiers": repeated_identity_dossiers,
        "summary": {
            "checks_total": len(checks),
            "checks_available": len(available_checks),
            "checks_unavailable": len(unavailable_checks),
            "anomalies_count": len(anomalies),
            "critical_anomalies_count": critical_count,
            "linked_dossiers_count": len(linked_dossiers),
            "repeated_identity_dossiers_count": len(repeated_identity_dossiers),
        },
    }


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


def _collect_documents(value: Any) -> set[str]:
    documents: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "documents" and isinstance(nested, list):
                documents.update(str(item) for item in nested if item)
            else:
                documents.update(_collect_documents(nested))
    elif isinstance(value, list):
        for item in value:
            documents.update(_collect_documents(item))
    return documents


def _persist_result(client: Neo4jClient, result: dict[str, Any]) -> None:
    dossier_id = result["dossier_id"]
    _delete_previous_signals(client, dossier_id)

    client.driver.execute_query(
        """
        MATCH (d:Dossier {dossier_id: $dossier_id})
        SET
            d.score_inter_dossiers = $score,
            d.inter_dossiers_available = $available,
            d.inter_dossiers_status = $status,
            d.nombre_anomalies_inter = $anomalies_count,
            d.nombre_anomalies_critiques_inter = $critical_count,
            d.nombre_dossiers_lies_inter = $linked_count
        """,
        dossier_id=dossier_id,
        score=result.get("score"),
        available=result.get("available", False),
        status=result.get("status"),
        anomalies_count=result.get("summary", {}).get("anomalies_count", 0),
        critical_count=result.get("summary", {}).get("critical_anomalies_count", 0),
        linked_count=result.get("summary", {}).get("linked_dossiers_count", 0),
        database_=client.database,
    )

    for anomaly in result.get("anomalies", []):
        check_id = str(anomaly["check_id"])
        signal_id = f"INTER_{dossier_id}_{check_id}"
        linked_dossiers = sorted(set(anomaly.get("linked_dossiers", [])))
        all_documents = sorted(_collect_documents(anomaly))

        client.driver.execute_query(
            """
            MATCH (d:Dossier {dossier_id: $dossier_id})
            MERGE (s:SignalFraude {signal_id: $signal_id})
            SET
                s.type = $type_signal,
                s.famille = 'RESEAU_INTER_DOSSIERS',
                s.score = $score,
                s.critical = $critical,
                s.description = $description,
                s.dossiers_lies = $linked_dossiers,
                s.preuves_json = $preuves_json,
                s.statut = 'DETECTE',
                s.source = $source
            MERGE (d)-[:A_SIGNAL]->(s)
            """,
            dossier_id=dossier_id,
            signal_id=signal_id,
            type_signal=check_id,
            score=float(anomaly["score"]),
            critical=bool(anomaly.get("critical")),
            description=anomaly.get("message"),
            linked_dossiers=linked_dossiers,
            preuves_json=json.dumps(anomaly, ensure_ascii=False),
            source=SOURCE,
            database_=client.database,
        )

        for linked_id in linked_dossiers:
            client.driver.execute_query(
                """
                MATCH (s:SignalFraude {signal_id: $signal_id})
                MATCH (linked:Dossier {dossier_id: $linked_id})
                MERGE (s)-[:DOSSIER_LIE]->(linked)
                """,
                signal_id=signal_id,
                linked_id=linked_id,
                database_=client.database,
            )

        for document_id in all_documents:
            client.driver.execute_query(
                """
                MATCH (s:SignalFraude {signal_id: $signal_id})
                MATCH (doc:Document {doc_id: $document_id})
                MERGE (s)-[:DOCUMENT_PREUVE]->(doc)
                """,
                signal_id=signal_id,
                document_id=document_id,
                database_=client.database,
            )


def run_inter_dossier_signal(
    dossier_id: str,
    output_root: str | Path = "data/signals",
    client: Neo4jClient | None = None,
    *,
    include_test_dossiers: bool = False,
) -> dict[str, Any]:
    owns_client = client is None
    neo4j_client = client or Neo4jClient()
    try:
        if owns_client:
            neo4j_client.verifier_connexion()

        profiles = fetch_inter_profiles(
            neo4j_client,
            include_test_dossiers=include_test_dossiers,
        )
        result = evaluate_inter_profiles(dossier_id, profiles)
        if dossier_id in profiles:
            _persist_result(neo4j_client, result)

        output_dir = Path(output_root) / dossier_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "inter_dossiers.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        result["output_path"] = str(output_path)
        return result
    finally:
        if owns_client:
            neo4j_client.fermer()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcule le signal de fraude réseau entre dossiers."
    )
    parser.add_argument("--dossier-id", required=True)
    parser.add_argument("--output-root", default="data/signals")
    parser.add_argument(
        "--include-test-dossiers",
        action="store_true",
        help="Réservé aux tests contrôlés TEST_INTER_*.",
    )
    args = parser.parse_args()

    result = run_inter_dossier_signal(
        args.dossier_id,
        args.output_root,
        include_test_dossiers=args.include_test_dossiers,
    )
    print("\n===== SIGNAL INTER-DOSSIERS =====")
    print(f"Dossier : {result['dossier_id']}")
    print(f"Disponible : {result['available']}")
    print(f"Score : {result['score']}")
    print(f"Anomalies : {result['summary']['anomalies_count']}")
    print(f"Anomalies critiques : {result['summary']['critical_anomalies_count']}")
    print(f"Dossiers liés suspects : {result['summary']['linked_dossiers_count']}")
    print(f"Contrôles indisponibles : {result['summary']['checks_unavailable']}")
    print(f"Résultat : {result['output_path']}")


if __name__ == "__main__":
    main()
