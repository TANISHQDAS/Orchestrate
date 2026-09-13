"""Validate output.csv and optionally score against public sample rows."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["request_id", "amount_safe_to_pay", "affordability_status", "recommended_payment_method", "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation"]
STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    requests = {row["request_id"]: row for row in rows(ROOT / "dataset" / "requests.csv")}
    output = rows(ROOT / "output.csv")
    assert len(output) == len(requests), (len(output), len(requests))
    assert output and output[0].keys() == dict.fromkeys(FIELDS).keys()
    assert {row["request_id"] for row in output} == set(requests)
    for row in output:
        request = requests[row["request_id"]]
        amount = float(row["amount_safe_to_pay"])
        assert 0 <= amount <= float(request["requested_amount"]), row
        assert row["affordability_status"] in STATUSES
        assert row["recommended_payment_method"] in METHODS
    print(f"Validated {len(output)} output rows and exact required schema.")

    sample_path = ROOT / "dataset" / "sample_requests.csv"
    if sample_path.exists():
        sample = {row["request_id"]: row for row in rows(sample_path)}
        generated = {row["request_id"]: row for row in rows(ROOT / "sample_output.csv")} if (ROOT / "sample_output.csv").exists() else {}
        if generated:
            fields = ["affordability_status", "recommended_payment_method", "earliest_date_for_full_payment"]
            matches = sum(all(generated[key][field] == sample[key][field] for field in fields) for key in sample)
            print(f"Public sample decision-field matches: {matches}/{len(sample)}")


if __name__ == "__main__":
    main()