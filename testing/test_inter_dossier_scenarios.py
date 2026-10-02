from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.dossier_graph_loader import _delete_existing_dossier  # noqa: E402
from graph.neo4j_client import Neo4jClient  # noqa: E402
from signals.inter_dossier_signal import (  # noqa: E402
    fetch_inter_profiles,
    run_inter_dossier_signal,
)
from signals.intra_dossier_signal import (  # noqa: E402
    normalize_text,
    text_anomaly_score,
    text_similarity,
)


TEST_PREFIX = "TEST_INTER_"
DEFAULT_BASE_DOSSIER = "D147"
DEFAULT_REPORT = Path("data/testing/inter_dossier_scenarios/test_report.json")
DEFAULT_SIGNAL_ROOT = Path("data/signals")


class ScenarioError(RuntimeError):
    pass


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    expected_score: float
    expected_checks: dict[str, float | None]
    description: str
    prepare: Callable[[Neo4jClient, dict[str, str]], list[str]]
    verifier: Callable[[dict[str, Any]], None] | None = None


def _first_value(profile: dict[str, Any], field_name: str, *, required: bool = True) -> str | None:
    values = sorted(profile.get("fields", {}).get(field_name, {}).keys())
    if not values:
        if required:
            raise ScenarioError(
                f"Le dossier de référence ne contient pas de valeur {field_name}."
            )
        return None
    return values[0]


def _fetch_base_values(client: Neo4jClient, base_dossier_id: str) -> dict[str, str]:
    profiles = fetch_inter_profiles(client, include_test_dossiers=False)
    profile = profiles.get(base_dossier_id)
    if profile is None:
        raise ScenarioError(
            f"Dossier de référence absent de Neo4j : {base_dossier_id}"
        )
    iban = _first_value(profile, "IBAN", required=False)
    if not iban:
        raise ScenarioError(
            "Le dossier de référence doit contenir un IBAN pour tester ce contrôle."
        )
    return {
        "CIN": str(_first_value(profile, "CIN")),
        "NOM_COMPLET": str(_first_value(profile, "NOM_COMPLET")),
        "RIB": str(_first_value(profile, "RIB")),
        "IBAN": str(iban),
    }


def _create_test_dossier(
    client: Neo4jClient,
    dossier_id: str,
    *,
    cin: str | None,
    name: str | None,
    rib: str | None,
    iban: str | None,
) -> None:
    if not dossier_id.startswith(TEST_PREFIX):
        raise ScenarioError(f"Identifiant de test non sécurisé : {dossier_id}")

    _delete_existing_dossier(client, dossier_id)
    doc_id = f"{dossier_id}__PROFILE"
    fields = []
    for field_name, value in (
        ("CIN", cin),
        ("NOM_COMPLET", name),
        ("RIB", rib),
        ("IBAN", iban),
    ):
        if value in (None, ""):
            continue
        fields.append({
            "champ_id": f"{doc_id}__{field_name}",
            "nom": field_name,
            "valeur": value,
            "valeur_normalisee": value,
        })

    client.driver.execute_query(
        """
        MERGE (d:Dossier {dossier_id: $dossier_id})
        SET d.ready_for_signals = true,
            d.validation_status = 'TEST_CONTROLLED'
        MERGE (doc:Document {doc_id: $doc_id})
        SET doc.dossier_id = $dossier_id,
            doc.doc_type = 'TEST_PROFILE',
            doc.validation_status = 'VALID'
        MERGE (d)-[:CONTIENT]->(doc)
        WITH doc
        UNWIND $fields AS field
        MERGE (c:Champ {champ_id: field.champ_id})
        SET c.nom = field.nom,
            c.valeur = field.valeur,
            c.valeur_normalisee = field.valeur_normalisee,
            c.validation_status = 'VALID'
        MERGE (doc)-[:POSSEDE]->(c)
        """,
        dossier_id=dossier_id,
        doc_id=doc_id,
        fields=fields,
        database_=client.database,
    )


def _cleanup_all_test_dossiers(client: Neo4jClient) -> list[str]:
    records, _, _ = client.driver.execute_query(
        """
        MATCH (d:Dossier)
        WHERE d.dossier_id STARTS WITH $prefix
        RETURN d.dossier_id AS dossier_id
        ORDER BY dossier_id
        """,
        prefix=TEST_PREFIX,
        database_=client.database,
    )
    dossier_ids = [str(record["dossier_id"]) for record in records]
    for dossier_id in dossier_ids:
        _delete_existing_dossier(client, dossier_id)
    return dossier_ids


def _unique_values(index: int) -> dict[str, str]:
    return {
        "CIN": f"ZT{900000 + index}",
        "NOM_COMPLET": f"PERSONNE RESEAU TEST {index}",
        "RIB": f"990000000000000000{index:06d}",
        "IBAN": f"MA99{index:024d}",
    }


def _candidate_names(original: str) -> list[str]:
    base = normalize_text(original)
    tokens = base.split()
    candidates: list[str] = []
    if tokens:
        last = tokens[-1]
        for keep in range(len(last) - 1, 0, -1):
            candidates.append(" ".join([*tokens[:-1], last[:keep]]))
        for index in range(len(tokens)):
            changed = tokens.copy()
            changed[index] = "AUTRE"
            candidates.append(" ".join(changed))
    candidates.extend([
        "PERSONNE COMPLETEMENT DIFFERENTE",
        "RESEAU FRAUDULEUX INCONNU",
        "AUTRE IDENTITE TEST",
    ])
    return candidates


def _name_for_score(original: str, expected_score: float) -> str:
    for candidate in _candidate_names(original):
        score = text_anomaly_score(text_similarity(original, candidate))
        if abs(score - expected_score) <= 1e-9:
            return candidate
    raise ScenarioError(
        f"Impossible de produire un nom avec score {expected_score} depuis {original!r}."
    )


def _single_prepare(
    *,
    cin: Callable[[dict[str, str]], str | None],
    name: Callable[[dict[str, str]], str | None],
    rib: Callable[[dict[str, str]], str | None],
    iban: Callable[[dict[str, str]], str | None],
) -> Callable[[Neo4jClient, dict[str, str]], list[str]]:
    def prepare(client: Neo4jClient, base: dict[str, str]) -> list[str]:
        scenario_id = _CURRENT_SCENARIO_ID[0]
        _create_test_dossier(
            client,
            scenario_id,
            cin=cin(base),
            name=name(base),
            rib=rib(base),
            iban=iban(base),
        )
        return [scenario_id]
    return prepare


_CURRENT_SCENARIO_ID = [""]


def _prepare_name_count(total_cins: int) -> Callable[[Neo4jClient, dict[str, str]], list[str]]:
    if total_cins not in {3, 4}:
        raise ValueError("total_cins doit valoir 3 ou 4.")

    def prepare(client: Neo4jClient, base: dict[str, str]) -> list[str]:
        target_id = _CURRENT_SCENARIO_ID[0]
        created: list[str] = []
        anchor_count = total_cins - 2  # dossier historique + cible + ancres
        for anchor_index in range(1, anchor_count + 1):
            values = _unique_values(100 + anchor_index)
            anchor_id = f"{target_id}_ANCHOR_{anchor_index}"
            _create_test_dossier(
                client,
                anchor_id,
                cin=values["CIN"],
                name=base["NOM_COMPLET"],
                rib=values["RIB"],
                iban=values["IBAN"],
            )
            created.append(anchor_id)

        target_values = _unique_values(200 + total_cins)
        _create_test_dossier(
            client,
            target_id,
            cin=target_values["CIN"],
            name=base["NOM_COMPLET"],
            rib=target_values["RIB"],
            iban=target_values["IBAN"],
        )
        created.append(target_id)
        return created

    return prepare


def _verify_repeated_identity(result: dict[str, Any]) -> None:
    repeated = set(result.get("repeated_identity_dossiers", []))
    if not repeated:
        raise ScenarioError(
            "Le dossier répété n'a pas été reconnu comme même identité."
        )


def _verify_iban_unavailable(result: dict[str, Any]) -> None:
    checks = {check["check_id"]: check for check in result.get("checks", [])}
    check = checks.get("IBAN_MULTIPLE_IDENTITIES")
    if not check or check.get("available") is not False or check.get("score") is not None:
        raise ScenarioError(
            "Le contrôle IBAN devait être indisponible avec score null."
        )


def _scenarios(base: dict[str, str]) -> list[Scenario]:
    unique1 = _unique_values(1)
    unique2 = _unique_values(2)
    unique3 = _unique_values(3)
    unique4 = _unique_values(4)
    unique5 = _unique_values(5)
    unique6 = _unique_values(6)

    return [
        Scenario(
            "TEST_INTER_CLEAN_REPEAT",
            0.0,
            {
                "CIN_MULTIPLE_NAMES": 0.0,
                "RIB_MULTIPLE_IDENTITIES": 0.0,
                "IBAN_MULTIPLE_IDENTITIES": 0.0,
                "NAME_MULTIPLE_CIN": 0.0,
            },
            "Même personne redéposant un dossier : aucune fraude réseau.",
            _single_prepare(
                cin=lambda b: b["CIN"],
                name=lambda b: b["NOM_COMPLET"],
                rib=lambda b: b["RIB"],
                iban=lambda b: b["IBAN"],
            ),
            _verify_repeated_identity,
        ),
        Scenario(
            "TEST_INTER_CIN_NAME_025",
            0.25,
            {"CIN_MULTIPLE_NAMES": 0.25},
            "Même CIN avec nom légèrement différent.",
            _single_prepare(
                cin=lambda b: b["CIN"],
                name=lambda b: _name_for_score(b["NOM_COMPLET"], 0.25),
                rib=lambda b: unique1["RIB"],
                iban=lambda b: unique1["IBAN"],
            ),
        ),
        Scenario(
            "TEST_INTER_CIN_NAME_060",
            0.60,
            {"CIN_MULTIPLE_NAMES": 0.60},
            "Même CIN avec nom moyennement différent.",
            _single_prepare(
                cin=lambda b: b["CIN"],
                name=lambda b: _name_for_score(b["NOM_COMPLET"], 0.60),
                rib=lambda b: unique2["RIB"],
                iban=lambda b: unique2["IBAN"],
            ),
        ),
        Scenario(
            "TEST_INTER_CIN_NAME_100",
            1.0,
            {"CIN_MULTIPLE_NAMES": 1.0},
            "Même CIN avec identité fortement différente.",
            _single_prepare(
                cin=lambda b: b["CIN"],
                name=lambda b: _name_for_score(b["NOM_COMPLET"], 1.0),
                rib=lambda b: unique3["RIB"],
                iban=lambda b: unique3["IBAN"],
            ),
        ),
        Scenario(
            "TEST_INTER_RIB_DIFFERENT_CIN",
            1.0,
            {"RIB_MULTIPLE_IDENTITIES": 1.0},
            "Même RIB avec CIN différente.",
            _single_prepare(
                cin=lambda b: unique4["CIN"],
                name=lambda b: unique4["NOM_COMPLET"],
                rib=lambda b: b["RIB"],
                iban=lambda b: unique4["IBAN"],
            ),
        ),
        Scenario(
            "TEST_INTER_IBAN_DIFFERENT_CIN",
            1.0,
            {"IBAN_MULTIPLE_IDENTITIES": 1.0},
            "Même IBAN avec CIN différente.",
            _single_prepare(
                cin=lambda b: unique5["CIN"],
                name=lambda b: unique5["NOM_COMPLET"],
                rib=lambda b: unique5["RIB"],
                iban=lambda b: b["IBAN"],
            ),
        ),
        Scenario(
            "TEST_INTER_NAME_2_CIN",
            0.25,
            {"NAME_MULTIPLE_CIN": 0.25},
            "Même nom complet associé à deux CIN.",
            _single_prepare(
                cin=lambda b: unique6["CIN"],
                name=lambda b: b["NOM_COMPLET"],
                rib=lambda b: unique6["RIB"],
                iban=lambda b: unique6["IBAN"],
            ),
        ),
        Scenario(
            "TEST_INTER_NAME_3_CIN",
            0.50,
            {"NAME_MULTIPLE_CIN": 0.50},
            "Même nom complet associé à trois CIN.",
            _prepare_name_count(3),
        ),
        Scenario(
            "TEST_INTER_NAME_4_CIN",
            0.60,
            {"NAME_MULTIPLE_CIN": 0.60},
            "Même nom complet associé à quatre CIN.",
            _prepare_name_count(4),
        ),
        Scenario(
            "TEST_INTER_IBAN_ABSENT",
            0.0,
            {"IBAN_MULTIPLE_IDENTITIES": None},
            "IBAN absent : contrôle indisponible, pas score zéro artificiel.",
            _single_prepare(
                cin=lambda b: _unique_values(7)["CIN"],
                name=lambda b: _unique_values(7)["NOM_COMPLET"],
                rib=lambda b: _unique_values(7)["RIB"],
                iban=lambda b: None,
            ),
            _verify_iban_unavailable,
        ),
        Scenario(
            "TEST_INTER_COMBINED_RIB_NAME",
            1.0,
            {
                "RIB_MULTIPLE_IDENTITIES": 1.0,
                "NAME_MULTIPLE_CIN": 0.25,
            },
            "RIB critique partagé et même nom avec CIN différente : max = 1.",
            _single_prepare(
                cin=lambda b: _unique_values(8)["CIN"],
                name=lambda b: b["NOM_COMPLET"],
                rib=lambda b: b["RIB"],
                iban=lambda b: _unique_values(8)["IBAN"],
            ),
        ),
    ]


def _check_score(result: dict[str, Any], expected: float) -> None:
    observed = result.get("score")
    if observed is None or abs(float(observed) - expected) > 1e-9:
        raise ScenarioError(f"Score attendu={expected}, obtenu={observed}")


def _check_subscores(
    result: dict[str, Any],
    expected_checks: dict[str, float | None],
) -> None:
    checks = {check["check_id"]: check for check in result.get("checks", [])}
    for check_id, expected in expected_checks.items():
        if check_id not in checks:
            raise ScenarioError(f"Contrôle absent : {check_id}")
        observed = checks[check_id].get("score")
        if expected is None:
            if observed is not None:
                raise ScenarioError(
                    f"{check_id}: score attendu=null, obtenu={observed}"
                )
        elif observed is None or abs(float(observed) - expected) > 1e-9:
            raise ScenarioError(
                f"{check_id}: attendu={expected}, obtenu={observed}"
            )


def run_tests(
    base_dossier_id: str = DEFAULT_BASE_DOSSIER,
    report_path: Path = DEFAULT_REPORT,
    signal_root: Path = DEFAULT_SIGNAL_ROOT,
) -> dict[str, Any]:
    client = Neo4jClient()
    results: list[dict[str, Any]] = []

    try:
        client.verifier_connexion()
        removed_before = _cleanup_all_test_dossiers(client)
        base = _fetch_base_values(client, base_dossier_id)

        for scenario in _scenarios(base):
            _CURRENT_SCENARIO_ID[0] = scenario.scenario_id
            created_ids: list[str] = []
            try:
                created_ids = scenario.prepare(client, base)
                result = run_inter_dossier_signal(
                    scenario.scenario_id,
                    signal_root,
                    client=client,
                    include_test_dossiers=True,
                )
                _check_score(result, scenario.expected_score)
                _check_subscores(result, scenario.expected_checks)
                if scenario.verifier is not None:
                    scenario.verifier(result)

                print(
                    f"[OK] {scenario.scenario_id} | attendu={scenario.expected_score} "
                    f"| obtenu={result['score']} | anomalies="
                    f"{result['summary']['anomalies_count']}"
                )
                results.append({
                    "scenario_id": scenario.scenario_id,
                    "description": scenario.description,
                    "status": "OK",
                    "error": None,
                    "expected_score": scenario.expected_score,
                    "observed_score": result.get("score"),
                    "expected_checks": scenario.expected_checks,
                    "observed_checks": {
                        check["check_id"]: check.get("score")
                        for check in result.get("checks", [])
                    },
                    "anomalies_count": result["summary"]["anomalies_count"],
                    "critical_anomalies_count": result["summary"][
                        "critical_anomalies_count"
                    ],
                    "linked_dossiers": result.get("linked_dossiers", []),
                    "signal_json": result.get("output_path"),
                })
            except Exception as error:
                print(f"[ERREUR] {scenario.scenario_id} : {error}")
                results.append({
                    "scenario_id": scenario.scenario_id,
                    "description": scenario.description,
                    "status": "ERROR",
                    "error": str(error),
                    "expected_score": scenario.expected_score,
                    "observed_score": None,
                    "expected_checks": scenario.expected_checks,
                })
            finally:
                for dossier_id in reversed(created_ids):
                    _delete_existing_dossier(client, dossier_id)

        removed_after = _cleanup_all_test_dossiers(client)
        ok_count = sum(1 for item in results if item["status"] == "OK")
        failed_count = len(results) - ok_count
        report = {
            "base_dossier": base_dossier_id,
            "scenarios_total": len(results),
            "scenarios_ok": ok_count,
            "scenarios_failed": failed_count,
            "aggregation_tested": "MAX_AVAILABLE_SUBSCORES",
            "ocr_confidence_used": False,
            "test_dossiers_removed_before": removed_before,
            "test_dossiers_removed_after": removed_after,
            "neo4j_test_dossiers_kept": False,
            "results": results,
        }

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

        print("\n===== RÉSUMÉ TESTS INTER-DOSSIERS =====")
        print(f"Scénarios : {len(results)}")
        print(f"Réussis : {ok_count}")
        print(f"Échoués : {failed_count}")
        print(f"Rapport : {report_path}")
        print("Les dossiers TEST_INTER_* ont été supprimés de Neo4j après le test.")

        if failed_count:
            raise SystemExit(1)
        return report
    finally:
        client.fermer()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teste les scores du signal de fraude réseau inter-dossiers."
    )
    parser.add_argument("--base-dossier", default=DEFAULT_BASE_DOSSIER)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--signal-root", default=str(DEFAULT_SIGNAL_ROOT))
    args = parser.parse_args()
    run_tests(
        base_dossier_id=args.base_dossier,
        report_path=Path(args.report),
        signal_root=Path(args.signal_root),
    )


if __name__ == "__main__":
    main()
