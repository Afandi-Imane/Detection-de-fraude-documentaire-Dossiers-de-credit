from __future__ import annotations

import argparse
import json
from pathlib import Path

from testing.test_business_rules_from_corrected_json import (
    evaluate_dossier,
    load_corrected_documents,
)


DEFAULT_ROOT = Path(
    "data/testing/business_rules/fraud_cases"
)

EXPECTED = {
    "D9001": "THREE_CONSECUTIVE_PERIODS",
    "D9002": "ONE_SALARY_TRANSFER_PER_MONTH",
    "D9003": "SALARY_AMOUNT_MATCH",
    "D9004": "SALARY_LABEL_EMPLOYER",
    "D9005": "TRANSACTION_TOTALS",
    "D9006": "BALANCE_EQUATION",
    "D9007": "BALANCE_CHAINING",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
    )
    args = parser.parse_args()

    root = Path(args.root)
    grouped, errors = load_corrected_documents(root)

    if errors:
        raise RuntimeError(
            f"Erreurs de chargement : {errors}"
        )

    all_ok = True
    report = {}

    for dossier_id, expected_rule in EXPECTED.items():
        if dossier_id not in grouped:
            print(
                f"[ABSENT] {dossier_id}"
            )
            all_ok = False
            continue

        result = evaluate_dossier(grouped[dossier_id])
        actual = result["rules"][expected_rule]["status"]
        ok = actual == "TRIGGERED"
        all_ok = all_ok and ok

        print(
            f"{dossier_id} | {expected_rule} | "
            f"{actual} | {'OK' if ok else 'ERREUR'}"
        )

        report[dossier_id] = {
            "expected_rule": expected_rule,
            "expected_status": "TRIGGERED",
            "actual_status": actual,
            "business_score": result["business_score"],
            "ok": ok,
            "full_result": result,
        }

    output = (
        root.parent
        / "fraud_cases_verification_report.json"
    )
    output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nRapport : {output}")

    if not all_ok:
        raise SystemExit(1)

    print(
        "Toutes les règles attendues se sont déclenchées."
    )


if __name__ == "__main__":
    main()
