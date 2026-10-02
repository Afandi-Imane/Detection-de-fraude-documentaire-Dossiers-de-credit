from __future__ import annotations

import argparse
from pathlib import Path

from signals.business_rules_signal import (
    run_business_rules_signal,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Teste le signal métier sur un ou plusieurs "
            "dossier.json validés."
        )
    )
    parser.add_argument(
        "--dossier-json",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--expected-score",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--output-root",
        default="data/testing/business_rules/signal_outputs",
    )
    args = parser.parse_args()

    failures = 0

    for value in args.dossier_json:
        source = Path(value)
        result = run_business_rules_signal(
            source,
            args.output_root,
        )

        print(
            f"{result['dossier_id']} | "
            f"score={result['score']} | "
            f"status={result['status']} | "
            f"triggered="
            f"{result['summary']['triggered_scoring_rules']}"
        )

        if (
            args.expected_score is not None
            and result["score"] != args.expected_score
        ):
            failures += 1
            print(
                f"[ERREUR] attendu={args.expected_score}, "
                f"observé={result['score']}"
            )

    if failures:
        raise SystemExit(1)

    print("\nTest terminé avec succès.")


if __name__ == "__main__":
    main()
