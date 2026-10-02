import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from signals.intra_dossier_signal import evaluate_intra_records


def field(document, doc_type, champ, value):
    return {
        "document": document,
        "type_document": doc_type,
        "champ": champ,
        "raw_value": value,
        "normalized_value": value,
    }


def test_coherent():
    fields = [
        field("cin", "CIN", "CIN", "BK123456"),
        field("att", "ATTESTATION_TRAVAIL", "CIN", "BK123456"),
        field("rib", "RIB", "RIB", "001122"),
        field("rel", "RELEVE_BANCAIRE", "RIB", "001122"),
        field("cin", "CIN", "NOM_COMPLET", "EL ALAMI NABIL"),
        field("att", "ATTESTATION_TRAVAIL", "NOM_COMPLET", "NABIL EL ALAMI"),
    ]
    bulletins = [{
        "document": "bulletin_M1",
        "mois_dossier": "M1",
        "periode": "2025-01",
        "raw_value": "12400,00",
        "salaire_bulletin": 12400.0,
    }]
    transactions = [{
        "document": "releve_M1",
        "transaction_id": "tx1",
        "mois_dossier": "M1",
        "periode": "2025-01",
        "libelle": "VIR SALAIRE SOCIETE",
        "salaire_releve": 12400.0,
    }]
    result = evaluate_intra_records("D_TEST", fields, bulletins, transactions)
    assert result["score"] == 0.0
    assert result["summary"]["anomalies_count"] == 0


def test_cin_mismatch_is_not_diluted():
    fields = [
        field("cin", "CIN", "CIN", "BK123456"),
        field("att", "ATTESTATION_TRAVAIL", "CIN", "BK123457"),
        field("rib", "RIB", "RIB", "001122"),
        field("rel", "RELEVE_BANCAIRE", "RIB", "001122"),
    ]
    result = evaluate_intra_records("D_TEST", fields, [], [])
    assert result["score"] == 1.0
    cin_check = next(check for check in result["checks"] if check["check_id"] == "CIN")
    assert cin_check["critical"] is True
    assert cin_check["score"] == 1.0


def test_salary_progressive_score():
    bulletins = [{
        "document": "bulletin_M1",
        "mois_dossier": "M1",
        "periode": "2025-01",
        "raw_value": "10000",
        "salaire_bulletin": 10000.0,
    }]
    transactions = [{
        "document": "releve_M1",
        "transaction_id": "tx1",
        "mois_dossier": "M1",
        "periode": "2025-01",
        "libelle": "VIR SALAIRE SOCIETE",
        "salaire_releve": 9200.0,
    }]
    result = evaluate_intra_records("D_TEST", [], bulletins, transactions)
    assert result["score"] == 0.5


if __name__ == "__main__":
    test_coherent()
    test_cin_mismatch_is_not_diluted()
    test_salary_progressive_score()
    print("Tous les tests de logique intra-dossier ont réussi.")
