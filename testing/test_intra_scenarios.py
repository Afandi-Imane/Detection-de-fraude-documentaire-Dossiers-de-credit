from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.dossier_graph_loader import (  # noqa: E402
    _delete_existing_dossier,
    load_dossier_json,
)
from graph.neo4j_client import Neo4jClient  # noqa: E402
from signals.intra_dossier_signal import run_intra_dossier_signal  # noqa: E402


TEST_PREFIX = "TEST_INTRA_"
DEFAULT_SOURCE = Path("data/json_validated/D147_D_TEST/dossier.json")
DEFAULT_WORK_ROOT = Path("data/testing/intra_scenarios")
DEFAULT_SIGNAL_ROOT = Path("data/signals")


class ScenarioError(RuntimeError):
    """Erreur explicite lors de la préparation ou de la vérification d'un scénario."""


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dossier source introuvable : {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
        raise ValueError(f"Structure dossier.json invalide : {path}")
    return data


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def find_document(dossier: dict[str, Any], doc_type: str, month: str | None = None) -> dict[str, Any]:
    candidates = [
        document
        for document in dossier["documents"]
        if str(document.get("doc_type", "")).upper() == doc_type.upper()
    ]
    if month is not None:
        pattern = re.compile(rf"_M{re.escape(month[-1])}(?:_|$)", re.IGNORECASE)
        candidates = [
            document
            for document in candidates
            if pattern.search(str(document.get("doc_id", "")))
        ]
    if len(candidates) != 1:
        raise ScenarioError(
            f"Document attendu introuvable ou ambigu : type={doc_type}, mois={month}, "
            f"nombre={len(candidates)}"
        )
    return candidates[0]


def find_field(document: dict[str, Any], *field_names: str) -> dict[str, Any]:
    accepted = {name.lower() for name in field_names}
    matches = [
        field
        for field in document.get("fields", [])
        if str(field.get("field_name", "")).lower() in accepted
    ]
    if len(matches) != 1:
        raise ScenarioError(
            f"Champ attendu introuvable ou ambigu dans {document.get('doc_id')} : "
            f"{sorted(accepted)}, nombre={len(matches)}"
        )
    return matches[0]


def clone_with_new_id(source: dict[str, Any], new_dossier_id: str) -> dict[str, Any]:
    if not new_dossier_id.startswith(TEST_PREFIX):
        raise ScenarioError(f"Identifiant de test non sécurisé : {new_dossier_id}")

    dossier = copy.deepcopy(source)
    old_dossier_id = str(dossier.get("dossier_id", ""))
    dossier["dossier_id"] = new_dossier_id

    documents_by_type: dict[str, list[str]] = defaultdict(list)
    document_fields: dict[str, dict[str, dict[str, Any]]] = {}

    for document in dossier["documents"]:
        old_doc_id = str(document.get("doc_id", "DOC_UNKNOWN"))
        if old_dossier_id and old_doc_id.startswith(old_dossier_id):
            suffix = old_doc_id[len(old_dossier_id):]
            new_doc_id = f"{new_dossier_id}{suffix}"
        else:
            new_doc_id = f"{new_dossier_id}__{old_doc_id}"

        document["doc_id"] = new_doc_id
        document["dossier_id"] = new_dossier_id

        doc_type = str(document.get("doc_type", "UNKNOWN")).upper()
        documents_by_type[doc_type].append(new_doc_id)
        document_fields[new_doc_id] = {
            str(field.get("field_name")): {
                "raw_value": field.get("raw_value"),
                "normalized_value": field.get("normalized_value"),
                "confidence": field.get("confidence"),
                "validation_status": field.get("validation_status"),
            }
            for field in document.get("fields", [])
            if isinstance(field, dict) and field.get("field_name")
        }

    dossier["documents_by_type"] = dict(documents_by_type)
    dossier["document_fields"] = document_fields
    return dossier


def mutate_identifier(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").upper())
    if not text:
        raise ScenarioError("Impossible de modifier un identifiant vide.")

    last = text[-1]
    if last.isdigit():
        replacement = str((int(last) + 1) % 10)
    elif "A" <= last <= "Z":
        replacement = chr(((ord(last) - ord("A") + 1) % 26) + ord("A"))
    else:
        replacement = "1"
    return f"{text[:-1]}{replacement}"


def inject_cin_mismatch(dossier: dict[str, Any]) -> None:
    attestation = find_document(dossier, "ATTESTATION_TRAVAIL")
    cin_field = find_field(attestation, "cin", "cin_numero")
    original = cin_field.get("normalized_value") or cin_field.get("raw_value")
    fraud_value = mutate_identifier(original)
    cin_field["raw_value"] = fraud_value
    cin_field["normalized_value"] = fraud_value


def _to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().replace("\u00a0", "").replace(" ", "")
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    return float(text)


def _is_salary_transaction(transaction: dict[str, Any]) -> bool:
    label = " ".join(str(transaction.get("libelle", "")).upper().split())
    direction = str(transaction.get("sens", "")).upper().strip()
    return direction == "C" and re.search(r"\b(?:SALAIRE|PAIE|PAYE)\b", label) is not None


def inject_salary_difference(dossier: dict[str, Any], relative_difference: float = 0.08) -> None:
    if not 0 < relative_difference < 1:
        raise ScenarioError("La différence salariale doit être comprise entre 0 et 1.")

    bulletin = find_document(dossier, "BULLETIN_SALAIRE", "M1")
    net_field = find_field(bulletin, "net_a_payer")
    bulletin_amount = _to_float(net_field.get("normalized_value"))

    statement = find_document(dossier, "RELEVE_BANCAIRE", "M1")
    transactions_field = find_field(statement, "transactions")
    transactions = transactions_field.get("normalized_value")
    if not isinstance(transactions, list):
        raise ScenarioError("La liste normalisée des transactions M1 est absente.")

    salary_transactions = [tx for tx in transactions if isinstance(tx, dict) and _is_salary_transaction(tx)]
    if len(salary_transactions) != 1:
        raise ScenarioError(
            "Le relevé M1 doit contenir exactement un virement salaire pour ce test; "
            f"nombre={len(salary_transactions)}"
        )

    fraud_amount = round(bulletin_amount * (1.0 - relative_difference), 2)
    salary_transactions[0]["montant"] = fraud_amount

    raw_transactions = transactions_field.get("raw_value")
    if isinstance(raw_transactions, list):
        raw_salary_transactions = [
            tx for tx in raw_transactions if isinstance(tx, dict) and _is_salary_transaction(tx)
        ]
        if len(raw_salary_transactions) == 1:
            raw_salary_transactions[0]["montant"] = fraud_amount


def get_check(result: dict[str, Any], check_id: str) -> dict[str, Any]:
    for check in result.get("checks", []):
        if check.get("check_id") == check_id:
            return check
    raise ScenarioError(f"Contrôle absent du résultat : {check_id}")


def assert_score(actual: Any, expected: float, label: str) -> None:
    if actual is None or abs(float(actual) - expected) > 1e-9:
        raise AssertionError(f"{label} : attendu={expected}, obtenu={actual}")


def verify_coherent(result: dict[str, Any]) -> None:
    assert_score(result.get("score"), 0.0, "score global")
    if result.get("summary", {}).get("anomalies_count") != 0:
        raise AssertionError("Le scénario cohérent ne doit produire aucune anomalie.")


def verify_cin(result: dict[str, Any]) -> None:
    assert_score(result.get("score"), 1.0, "score global")
    cin_check = get_check(result, "CIN")
    assert_score(cin_check.get("score"), 1.0, "score CIN")
    if cin_check.get("critical") is not True:
        raise AssertionError("L'incohérence CIN doit être critique.")
    if not cin_check.get("suspect_fields"):
        raise AssertionError("Le champ CIN suspect doit être conservé.")


def verify_salary_8(result: dict[str, Any]) -> None:
    assert_score(result.get("score"), 0.50, "score global")
    salary_check = get_check(result, "SALAIRE_M1")
    assert_score(salary_check.get("score"), 0.50, "score SALAIRE_M1")
    relative_difference = salary_check.get("relative_difference")
    if relative_difference is None or abs(float(relative_difference) - 0.08) > 1e-6:
        raise AssertionError(
            f"Écart salarial relatif attendu=0.08, obtenu={relative_difference}"
        )
    if not salary_check.get("suspect_fields"):
        raise AssertionError("La transaction salaire suspecte doit être conservée.")


def verify_combined(result: dict[str, Any]) -> None:
    assert_score(result.get("score"), 1.0, "score global max")
    assert_score(get_check(result, "CIN").get("score"), 1.0, "score CIN")
    assert_score(get_check(result, "SALAIRE_M1").get("score"), 0.50, "score SALAIRE_M1")


ScenarioMutation = Callable[[dict[str, Any]], None]
ScenarioVerifier = Callable[[dict[str, Any]], None]


SCENARIOS: list[tuple[str, ScenarioMutation | None, ScenarioVerifier]] = [
    ("TEST_INTRA_COHERENT", None, verify_coherent),
    ("TEST_INTRA_CIN", inject_cin_mismatch, verify_cin),
    (
        "TEST_INTRA_SALAIRE_08",
        lambda dossier: inject_salary_difference(dossier, 0.08),
        verify_salary_8,
    ),
    (
        "TEST_INTRA_CIN_SALAIRE_08",
        lambda dossier: (
            inject_cin_mismatch(dossier),
            inject_salary_difference(dossier, 0.08),
        ),
        verify_combined,
    ),
]


def cleanup_test_dossier(client: Neo4jClient, dossier_id: str) -> None:
    if not dossier_id.startswith(TEST_PREFIX):
        raise ScenarioError(f"Suppression refusée pour un dossier non-test : {dossier_id}")
    _delete_existing_dossier(client, dossier_id)


def run_scenarios(
    source_path: Path,
    work_root: Path,
    signal_root: Path,
    keep_neo4j: bool,
) -> dict[str, Any]:
    source = load_json(source_path)
    client = Neo4jClient()
    created_ids: list[str] = []
    report_rows: list[dict[str, Any]] = []

    try:
        client.verifier_connexion()

        for scenario_id, mutation, verifier in SCENARIOS:
            scenario_dossier = clone_with_new_id(source, scenario_id)
            if mutation is not None:
                mutation(scenario_dossier)

            scenario_json = work_root / scenario_id / "dossier.json"
            save_json(scenario_dossier, scenario_json)

            load_result = load_dossier_json(scenario_json, client=client)
            created_ids.append(scenario_id)
            result = run_intra_dossier_signal(
                scenario_id,
                output_root=signal_root,
                client=client,
            )

            try:
                verifier(result)
                status = "OK"
                error = None
                print(
                    f"[OK] {scenario_id} | score={result.get('score')} | "
                    f"anomalies={result.get('summary', {}).get('anomalies_count')}"
                )
            except Exception as exc:
                status = "ERROR"
                error = str(exc)
                print(f"[ERREUR] {scenario_id} | {error}")

            report_rows.append({
                "scenario_id": scenario_id,
                "status": status,
                "error": error,
                "score": result.get("score"),
                "anomalies_count": result.get("summary", {}).get("anomalies_count"),
                "dossier_json": str(scenario_json),
                "signal_json": result.get("output_path"),
                "neo4j_counts": {
                    "documents": load_result.get("documents"),
                    "fields": load_result.get("fields"),
                    "transactions": load_result.get("transactions"),
                },
            })

        failed = [row for row in report_rows if row["status"] != "OK"]
        report = {
            "source_dossier": str(source_path),
            "scenarios_total": len(report_rows),
            "scenarios_ok": len(report_rows) - len(failed),
            "scenarios_failed": len(failed),
            "neo4j_test_dossiers_kept": keep_neo4j,
            "results": report_rows,
        }
        save_json(report, work_root / "test_report.json")

        if failed:
            raise AssertionError(
                f"{len(failed)} scénario(s) ont échoué. Voir {work_root / 'test_report.json'}"
            )
        return report

    finally:
        if not keep_neo4j:
            for dossier_id in reversed(created_ids):
                cleanup_test_dossier(client, dossier_id)
        client.fermer()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Teste les paliers du signal intra-dossier sur des copies contrôlées "
            "et sécurisées d'un dossier légitime."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--signal-root", type=Path, default=DEFAULT_SIGNAL_ROOT)
    parser.add_argument(
        "--keep-neo4j",
        action="store_true",
        help="Conserve les dossiers TEST_INTRA_* dans Neo4j après les tests.",
    )
    args = parser.parse_args()

    report = run_scenarios(
        source_path=args.source,
        work_root=args.work_root,
        signal_root=args.signal_root,
        keep_neo4j=args.keep_neo4j,
    )

    print("\n===== RÉSUMÉ TESTS INTRA-DOSSIER =====")
    print(f"Scénarios : {report['scenarios_total']}")
    print(f"Réussis : {report['scenarios_ok']}")
    print(f"Échoués : {report['scenarios_failed']}")
    print(f"Rapport : {args.work_root / 'test_report.json'}")
    if args.keep_neo4j:
        print("Les dossiers TEST_INTRA_* sont conservés dans Neo4j.")
    else:
        print("Les dossiers TEST_INTRA_* ont été supprimés de Neo4j après le test.")


if __name__ == "__main__":
    main()
