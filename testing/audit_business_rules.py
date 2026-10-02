from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.neo4j_client import Neo4jClient


OUTPUT_DIR = Path("data/testing/business_rules")
REPORT_PATH = OUTPUT_DIR / "audit_business_rules.json"
FIELDS_CSV_PATH = OUTPUT_DIR / "field_catalog.csv"
TRANSACTIONS_CSV_PATH = OUTPUT_DIR / "transaction_property_catalog.csv"
RULES_CSV_PATH = OUTPUT_DIR / "rule_readiness.csv"

TEST_FILTER = """
d.dossier_id <> 'D_TEST'
AND NOT d.dossier_id ENDS WITH '_D_TEST'
AND NOT d.dossier_id STARTS WITH 'TEST_'
"""

# Une règle peut demander :
# - un groupe de champs : au moins un alias doit exister ;
# - une propriété de transaction ;
# - une propriété portée par Dossier ou Document ;
# - une table de référence à construire depuis les règles du générateur.
#
# Cet audit ne calcule aucun score et ne conclut jamais à la fraude.
RULE_DEFINITIONS = [
    {
        "rule_id": "CIN_CITY_PREFIX",
        "description": "Préfixe CIN compatible avec la ville du client.",
        "field_groups": [["CIN"], ["VILLE", "CITY", "VILLE_CLIENT"]],
        "transaction_properties": [],
        "dossier_properties": [],
        "document_properties": [],
        "reference_required": "cin_city_prefixes.json",
    },
    {
        "rule_id": "EMPLOYER_POSITION_ALLOWED",
        "description": "Poste autorisé pour l'employeur ou son secteur.",
        "field_groups": [["EMPLOYEUR"], ["POSTE"]],
        "transaction_properties": [],
        "dossier_properties": [],
        "document_properties": [],
        "reference_required": "professional_reference.json",
    },
    {
        "rule_id": "POSITION_SALARY_RANGE",
        "description": "Salaire net compatible avec la fourchette du poste.",
        "field_groups": [
            ["POSTE"],
            ["NET_A_PAYER", "SALAIRE_NET", "NET_PAYE", "NET_IMPOSABLE"],
        ],
        "transaction_properties": [],
        "dossier_properties": [],
        "document_properties": [],
        "reference_required": "professional_reference.json",
    },
    {
        "rule_id": "EMPLOYMENT_DATE_BEFORE_DEPOSIT",
        "description": "Date d'embauche antérieure à la date de dépôt.",
        "field_groups": [["DATE_EMBAUCHE"]],
        "transaction_properties": [],
        "dossier_properties": ["date_depot"],
        "document_properties": [],
        "reference_required": None,
    },
    {
        "rule_id": "DOCUMENT_NOT_AFTER_DEPOSIT",
        "description": "Date du document non postérieure à la date de dépôt.",
        "field_groups": [["DATE_DOCUMENT"]],
        "transaction_properties": [],
        "dossier_properties": ["date_depot"],
        "document_properties": [],
        "reference_required": None,
    },
    {
        "rule_id": "RECENT_QUITTANCE_ATTESTATION",
        "description": "Quittance et attestation âgées de 90 jours au maximum.",
        "field_groups": [["DATE_DOCUMENT"]],
        "transaction_properties": [],
        "dossier_properties": ["date_depot"],
        "document_properties": ["doc_type"],
        "reference_required": None,
    },
    {
        "rule_id": "THREE_CONSECUTIVE_PERIODS",
        "description": "Trois périodes consécutives pour bulletins et relevés.",
        "field_groups": [["PERIODE", "MOIS", "MOIS_DOCUMENT"]],
        "transaction_properties": [],
        "dossier_properties": [],
        "document_properties": ["doc_type"],
        "reference_required": None,
    },
    {
        "rule_id": "ONE_SALARY_TRANSFER_PER_MONTH",
        "description": "Exactement un virement de salaire par mois.",
        "field_groups": [],
        "transaction_properties": ["est_salaire", "mois_dossier"],
        "dossier_properties": [],
        "document_properties": [],
        "reference_required": None,
    },
    {
        "rule_id": "SALARY_AMOUNT_MATCH",
        "description": "Montant du virement salaire égal au net du bulletin.",
        "field_groups": [
            ["NET_A_PAYER", "SALAIRE_NET", "NET_PAYE", "NET_IMPOSABLE"],
            ["PERIODE", "MOIS", "MOIS_DOCUMENT"],
        ],
        "transaction_properties": ["montant", "est_salaire", "mois_dossier"],
        "dossier_properties": [],
        "document_properties": [],
        "reference_required": None,
    },
    {
        "rule_id": "SALARY_LABEL_EMPLOYER",
        "description": "Libellé du virement salaire compatible avec l'employeur.",
        "field_groups": [["EMPLOYEUR"]],
        "transaction_properties": ["libelle", "est_salaire"],
        "dossier_properties": [],
        "document_properties": [],
        "reference_required": None,
    },
    {
        "rule_id": "EMPLOYER_PAYDAY",
        "description": "Date du virement compatible avec la date de paie de l'employeur.",
        "field_groups": [["EMPLOYEUR"]],
        "transaction_properties": ["date", "est_salaire"],
        "dossier_properties": [],
        "document_properties": [],
        "reference_required": "employer_paydays.json",
    },
    {
        "rule_id": "TRANSACTION_TOTALS",
        "description": "Somme des opérations égale aux totaux débit/crédit déclarés.",
        "field_groups": [
            ["TOTAL_CREDITS", "TOTAL_CREDIT"],
            ["TOTAL_DEBITS", "TOTAL_DEBIT"],
        ],
        "transaction_properties": ["montant", "sens"],
        "dossier_properties": [],
        "document_properties": [],
        "reference_required": None,
    },
    {
        "rule_id": "BALANCE_EQUATION",
        "description": "Ouverture + crédits - débits = clôture.",
        "field_groups": [
            ["SOLDE_OUVERTURE", "SOLDE_DEPART"],
            ["TOTAL_CREDITS", "TOTAL_CREDIT"],
            ["TOTAL_DEBITS", "TOTAL_DEBIT"],
            ["SOLDE_CLOTURE", "SOLDE_FINAL"],
        ],
        "transaction_properties": [],
        "dossier_properties": [],
        "document_properties": [],
        "reference_required": None,
    },
    {
        "rule_id": "BALANCE_CHAINING",
        "description": "Clôture du mois N égale à l'ouverture du mois N+1.",
        "field_groups": [
            ["SOLDE_OUVERTURE", "SOLDE_DEPART"],
            ["SOLDE_CLOTURE", "SOLDE_FINAL"],
            ["PERIODE", "MOIS", "MOIS_DOCUMENT"],
        ],
        "transaction_properties": [],
        "dossier_properties": [],
        "document_properties": [],
        "reference_required": None,
    },
    {
        "rule_id": "NON_NEGATIVE_BALANCE",
        "description": "Solde bancaire non négatif.",
        "field_groups": [
            ["SOLDE_OUVERTURE", "SOLDE_DEPART", "SOLDE_CLOTURE", "SOLDE_FINAL"]
        ],
        "transaction_properties": [],
        "dossier_properties": [],
        "document_properties": [],
        "reference_required": None,
    },
    {
        "rule_id": "ADDITIONAL_CREDITS_COUNT",
        "description": "Un salaire accompagné de 1 à 3 crédits additionnels par relevé.",
        "field_groups": [],
        "transaction_properties": ["sens", "est_salaire", "mois_dossier"],
        "dossier_properties": [],
        "document_properties": [],
        "reference_required": None,
    },
]


def execute(client: Neo4jClient, query: str, **parameters: Any) -> list[dict[str, Any]]:
    records, _, _ = client.driver.execute_query(
        query,
        database_=client.database,
        **parameters,
    )
    return [record.data() for record in records]


def normalize_name(value: Any) -> str:
    return str(value or "").strip().upper()


def coverage_ratio(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_counts(client: Neo4jClient) -> dict[str, int]:
    rows = execute(
        client,
        f"""
        MATCH (d:Dossier)
        WHERE {TEST_FILTER}
        OPTIONAL MATCH (d)-[:CONTIENT]->(doc:Document)
        OPTIONAL MATCH (doc)-[:POSSEDE]->(c:Champ)
        OPTIONAL MATCH (doc)-[:CONTIENT_TRANSACTION]->(t:Transaction)
        RETURN
            count(DISTINCT d) AS dossiers,
            count(DISTINCT doc) AS documents,
            count(DISTINCT c) AS fields,
            count(DISTINCT t) AS transactions
        """,
    )
    return rows[0] if rows else {
        "dossiers": 0,
        "documents": 0,
        "fields": 0,
        "transactions": 0,
    }


def audit_fields(client: Neo4jClient, total_dossiers: int) -> list[dict[str, Any]]:
    rows = execute(
        client,
        f"""
        MATCH (d:Dossier)-[:CONTIENT]->(doc:Document)-[:POSSEDE]->(c:Champ)
        WHERE {TEST_FILTER}
        WITH
            toUpper(trim(toString(c.nom))) AS field_name,
            d,
            doc,
            c
        WHERE field_name <> ''
        WITH
            field_name,
            count(c) AS occurrences,
            count(DISTINCT d) AS dossiers_with_field,
            count(
                CASE
                    WHEN c.valeur_normalisee IS NOT NULL
                     AND trim(toString(c.valeur_normalisee)) <> ''
                    THEN 1
                END
            ) AS non_empty_occurrences,
            count(
                DISTINCT CASE
                    WHEN c.valeur_normalisee IS NOT NULL
                     AND trim(toString(c.valeur_normalisee)) <> ''
                    THEN d
                END
            ) AS dossiers_with_non_empty_value,
            collect(DISTINCT doc.doc_type)[0..20] AS document_types
        RETURN
            field_name,
            occurrences,
            dossiers_with_field,
            non_empty_occurrences,
            dossiers_with_non_empty_value,
            document_types
        ORDER BY field_name
        """,
    )

    result = []
    for row in rows:
        row["coverage_ratio"] = coverage_ratio(
            int(row["dossiers_with_non_empty_value"]),
            total_dossiers,
        )
        row["document_types"] = "|".join(
            sorted(str(value) for value in (row.get("document_types") or []) if value)
        )
        result.append(row)
    return result


def audit_node_properties(
    client: Neo4jClient,
    label: str,
    match_pattern: str,
    where_clause: str,
) -> list[dict[str, Any]]:
    # label est uniquement utilisé pour l'affichage ; le pattern est fixé dans le code.
    del label
    return execute(
        client,
        f"""
        MATCH {match_pattern}
        WHERE {where_clause}
        UNWIND keys(n) AS property_name
        WITH property_name, n
        RETURN
            property_name,
            count(n) AS nodes_with_property,
            count(
                CASE
                    WHEN n[property_name] IS NOT NULL
                     AND trim(toString(n[property_name])) <> ''
                    THEN 1
                END
            ) AS non_empty_values
        ORDER BY property_name
        """,
    )


def audit_transaction_properties(client: Neo4jClient) -> list[dict[str, Any]]:
    return execute(
        client,
        f"""
        MATCH (d:Dossier)-[:CONTIENT]->(:Document)-[:CONTIENT_TRANSACTION]->(n:Transaction)
        WHERE {TEST_FILTER}
        UNWIND keys(n) AS property_name
        WITH property_name, n
        RETURN
            property_name,
            count(n) AS nodes_with_property,
            count(
                CASE
                    WHEN n[property_name] IS NOT NULL
                     AND trim(toString(n[property_name])) <> ''
                    THEN 1
                END
            ) AS non_empty_values
        ORDER BY property_name
        """,
    )


def build_rule_readiness(
    fields: list[dict[str, Any]],
    transaction_properties: list[dict[str, Any]],
    dossier_properties: list[dict[str, Any]],
    document_properties: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    available_fields = {
        normalize_name(row["field_name"])
        for row in fields
        if int(row.get("dossiers_with_non_empty_value", 0)) > 0
    }
    available_tx = {
        str(row["property_name"]).strip()
        for row in transaction_properties
        if int(row.get("non_empty_values", 0)) > 0
    }
    available_dossier = {
        str(row["property_name"]).strip()
        for row in dossier_properties
        if int(row.get("non_empty_values", 0)) > 0
    }
    available_document = {
        str(row["property_name"]).strip()
        for row in document_properties
        if int(row.get("non_empty_values", 0)) > 0
    }

    results = []

    for rule in RULE_DEFINITIONS:
        missing_groups = []
        resolved_groups = []

        for aliases in rule["field_groups"]:
            aliases_normalized = [normalize_name(alias) for alias in aliases]
            found = sorted(set(aliases_normalized) & available_fields)
            if found:
                resolved_groups.append("|".join(found))
            else:
                missing_groups.append("|".join(aliases_normalized))

        missing_tx = sorted(
            set(rule["transaction_properties"]) - available_tx
        )
        missing_dossier = sorted(
            set(rule["dossier_properties"]) - available_dossier
        )
        missing_document = sorted(
            set(rule["document_properties"]) - available_document
        )

        data_ready = not any([
            missing_groups,
            missing_tx,
            missing_dossier,
            missing_document,
        ])

        reference_required = rule.get("reference_required")
        status = "READY_DATA"
        if not data_ready:
            status = "MISSING_DATA"
        elif reference_required:
            status = "READY_DATA_REFERENCE_NEEDED"

        results.append({
            "rule_id": rule["rule_id"],
            "description": rule["description"],
            "status": status,
            "resolved_field_groups": ";".join(resolved_groups),
            "missing_field_groups": ";".join(missing_groups),
            "missing_transaction_properties": ";".join(missing_tx),
            "missing_dossier_properties": ";".join(missing_dossier),
            "missing_document_properties": ";".join(missing_document),
            "reference_required": reference_required or "",
        })

    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = Neo4jClient()
    try:
        client.verifier_connexion()

        counts = audit_counts(client)
        fields = audit_fields(client, int(counts["dossiers"]))

        dossier_properties = audit_node_properties(
            client=client,
            label="Dossier",
            match_pattern="(n:Dossier)",
            where_clause=TEST_FILTER.replace("d.", "n."),
        )
        document_properties = audit_node_properties(
            client=client,
            label="Document",
            match_pattern="(d:Dossier)-[:CONTIENT]->(n:Document)",
            where_clause=TEST_FILTER,
        )
        transaction_properties = audit_transaction_properties(client)

        rules = build_rule_readiness(
            fields=fields,
            transaction_properties=transaction_properties,
            dossier_properties=dossier_properties,
            document_properties=document_properties,
        )

        report = {
            "audit_type": "BUSINESS_RULES_INPUT_READ_ONLY",
            "mutates_neo4j": False,
            "excluded": {
                "rib_key": True,
                "iban_key": True,
                "benford": True,
            },
            "counts": counts,
            "field_catalog": fields,
            "dossier_property_catalog": dossier_properties,
            "document_property_catalog": document_properties,
            "transaction_property_catalog": transaction_properties,
            "rule_readiness": rules,
            "notes": [
                "Cet audit vérifie uniquement la présence des entrées nécessaires.",
                "Il ne calcule aucun score de fraude.",
                "Les tables de référence seront construites après validation des champs disponibles.",
            ],
        }

        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        write_csv(
            FIELDS_CSV_PATH,
            fields,
            [
                "field_name",
                "occurrences",
                "dossiers_with_field",
                "non_empty_occurrences",
                "dossiers_with_non_empty_value",
                "coverage_ratio",
                "document_types",
            ],
        )
        write_csv(
            TRANSACTIONS_CSV_PATH,
            transaction_properties,
            [
                "property_name",
                "nodes_with_property",
                "non_empty_values",
            ],
        )
        write_csv(
            RULES_CSV_PATH,
            rules,
            [
                "rule_id",
                "description",
                "status",
                "resolved_field_groups",
                "missing_field_groups",
                "missing_transaction_properties",
                "missing_dossier_properties",
                "missing_document_properties",
                "reference_required",
            ],
        )

        print("\n===== AUDIT ENTRÉES RÈGLES MÉTIER =====")
        print(f"Dossiers : {counts['dossiers']}")
        print(f"Documents : {counts['documents']}")
        print(f"Champs : {counts['fields']}")
        print(f"Transactions : {counts['transactions']}")

        ready = [rule for rule in rules if rule["status"] == "READY_DATA"]
        reference = [
            rule for rule in rules
            if rule["status"] == "READY_DATA_REFERENCE_NEEDED"
        ]
        missing = [rule for rule in rules if rule["status"] == "MISSING_DATA"]

        print("\nPréparation des règles :")
        print(f"- Données disponibles : {len(ready)}")
        print(f"- Données disponibles, référentiel à créer : {len(reference)}")
        print(f"- Données manquantes : {len(missing)}")

        if missing:
            print("\nRègles avec données manquantes :")
            for rule in missing:
                details = []
                if rule["missing_field_groups"]:
                    details.append(
                        f"champs={rule['missing_field_groups']}"
                    )
                if rule["missing_transaction_properties"]:
                    details.append(
                        "transactions="
                        f"{rule['missing_transaction_properties']}"
                    )
                if rule["missing_dossier_properties"]:
                    details.append(
                        f"dossier={rule['missing_dossier_properties']}"
                    )
                if rule["missing_document_properties"]:
                    details.append(
                        f"document={rule['missing_document_properties']}"
                    )
                print(f"- {rule['rule_id']} : {', '.join(details)}")

        print(f"\nRapport JSON : {REPORT_PATH}")
        print(f"Catalogue champs : {FIELDS_CSV_PATH}")
        print(f"Catalogue transactions : {TRANSACTIONS_CSV_PATH}")
        print(f"État des règles : {RULES_CSV_PATH}")
        print("Aucune donnée Neo4j n'a été modifiée.")

    finally:
        client.fermer()


if __name__ == "__main__":
    main()
