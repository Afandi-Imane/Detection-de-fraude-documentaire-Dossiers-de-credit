from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graph.neo4j_client import Neo4jClient


DEFAULT_FIELDS = [
    "CIN",
    "RIB",
    "IBAN",
    "NOM_COMPLET",
    "ADRESSE",
    "EMPLOYEUR",
]


def _single_value(client: Neo4jClient, query: str, key: str, **parameters: Any) -> int:
    records, _, _ = client.driver.execute_query(
        query,
        **parameters,
        database_=client.database,
    )
    return int(records[0][key]) if records else 0


def run_audit(
    output_path: str | Path = "data/testing/inter_dossier/audit_graph.json",
    duplicate_limit: int = 20,
) -> dict[str, Any]:
    client = Neo4jClient()
    try:
        client.verifier_connexion()

        counts = {
            "dossiers": _single_value(
                client,
                "MATCH (d:Dossier) RETURN count(d) AS n",
                "n",
            ),
            "dossiers_ready_for_signals": _single_value(
                client,
                "MATCH (d:Dossier) WHERE d.ready_for_signals = true RETURN count(d) AS n",
                "n",
            ),
            "documents": _single_value(
                client,
                "MATCH (d:Document) RETURN count(d) AS n",
                "n",
            ),
            "fields": _single_value(
                client,
                "MATCH (c:Champ) RETURN count(c) AS n",
                "n",
            ),
            "transactions": _single_value(
                client,
                "MATCH (t:Transaction) RETURN count(t) AS n",
                "n",
            ),
        }

        structural_checks = {
            "documents_without_dossier": _single_value(
                client,
                "MATCH (doc:Document) WHERE NOT (:Dossier)-[:CONTIENT]->(doc) RETURN count(doc) AS n",
                "n",
            ),
            "fields_without_document": _single_value(
                client,
                "MATCH (c:Champ) WHERE NOT (:Document)-[:POSSEDE]->(c) RETURN count(c) AS n",
                "n",
            ),
            "fields_without_name": _single_value(
                client,
                "MATCH (c:Champ) WHERE c.nom IS NULL OR trim(toString(c.nom)) = '' RETURN count(c) AS n",
                "n",
            ),
            "fields_without_normalized_value": _single_value(
                client,
                "MATCH (c:Champ) WHERE c.valeur_normalisee IS NULL OR trim(toString(c.valeur_normalisee)) = '' RETURN count(c) AS n",
                "n",
            ),
        }

        coverage: dict[str, Any] = {}
        duplicates: dict[str, list[dict[str, Any]]] = {}

        for field_name in DEFAULT_FIELDS:
            dossier_count = _single_value(
                client,
                """
                MATCH (d:Dossier)-[:CONTIENT]->(:Document)-[:POSSEDE]->(c:Champ {nom: $field_name})
                WHERE c.valeur_normalisee IS NOT NULL
                  AND trim(toString(c.valeur_normalisee)) <> ''
                RETURN count(DISTINCT d) AS n
                """,
                "n",
                field_name=field_name,
            )
            coverage[field_name] = {
                "dossiers_with_value": dossier_count,
                "coverage_ratio": round(
                    dossier_count / counts["dossiers"], 4
                ) if counts["dossiers"] else None,
            }

            records, _, _ = client.driver.execute_query(
                """
                MATCH (d:Dossier)-[:CONTIENT]->(doc:Document)-[:POSSEDE]->(c:Champ {nom: $field_name})
                WHERE c.valeur_normalisee IS NOT NULL
                  AND trim(toString(c.valeur_normalisee)) <> ''
                WITH
                    trim(toString(c.valeur_normalisee)) AS normalized_value,
                    collect(DISTINCT d.dossier_id) AS dossiers,
                    collect(DISTINCT doc.doc_id) AS documents
                WHERE size(dossiers) > 1
                RETURN
                    normalized_value,
                    size(dossiers) AS dossier_count,
                    dossiers,
                    documents
                ORDER BY dossier_count DESC, normalized_value
                LIMIT $duplicate_limit
                """,
                field_name=field_name,
                duplicate_limit=int(duplicate_limit),
                database_=client.database,
            )
            duplicates[field_name] = [dict(record) for record in records]

        duplicate_summary = {
            field_name: {
                "groups_returned": len(groups),
                "max_dossier_count": max(
                    (int(group["dossier_count"]) for group in groups),
                    default=0,
                ),
            }
            for field_name, groups in duplicates.items()
        }

        result = {
            "audit_type": "INTER_DOSSIER_GRAPH_READ_ONLY",
            "mutates_neo4j": False,
            "counts": counts,
            "structural_checks": structural_checks,
            "field_coverage": coverage,
            "duplicate_summary": duplicate_summary,
            "duplicate_groups": duplicates,
            "interpretation_note": (
                "Cet audit repere les reutilisations de valeurs entre dossiers. "
                "Une reutilisation n'est pas automatiquement une fraude : le futur signal "
                "tiendra compte du type d'entite et des identites associees."
            ),
        }

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        print("\n===== AUDIT GRAPHE INTER-DOSSIERS =====")
        print(f"Dossiers : {counts['dossiers']}")
        print(f"Dossiers prêts pour signaux : {counts['dossiers_ready_for_signals']}")
        print(f"Documents : {counts['documents']}")
        print(f"Champs : {counts['fields']}")
        print(f"Transactions : {counts['transactions']}")

        print("\nCouverture des entités :")
        for field_name, field_result in coverage.items():
            print(
                f"- {field_name}: {field_result['dossiers_with_value']} dossiers "
                f"({field_result['coverage_ratio']})"
            )

        print("\nGroupes de valeurs partagées trouvés :")
        for field_name, summary in duplicate_summary.items():
            print(
                f"- {field_name}: {summary['groups_returned']} groupe(s), "
                f"maximum {summary['max_dossier_count']} dossiers"
            )

        print(f"\nRapport : {destination}")
        print("Aucune donnée Neo4j n'a été modifiée.")
        return result

    finally:
        client.fermer()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit en lecture seule du graphe avant création du signal inter-dossiers."
        )
    )
    parser.add_argument(
        "--output",
        default="data/testing/inter_dossier/audit_graph.json",
    )
    parser.add_argument("--duplicate-limit", type=int, default=20)
    args = parser.parse_args()
    run_audit(args.output, args.duplicate_limit)


if __name__ == "__main__":
    main()
