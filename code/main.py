"""Deterministic Buy or Wait financial decision agent.

The agent intentionally keeps the financial arithmetic deterministic. Optional
LLM/VLM enrichment is not required to run the submission and cannot override
the ledger or the challenge constraints.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
OUTPUT_COLUMNS = [
	"request_id", "amount_safe_to_pay", "affordability_status",
	"recommended_payment_method", "payment_plan",
	"earliest_date_for_full_payment", "spending_changes_needed",
	"decision_explanation",
]
ACTIVE_STATUSES = {"settled", "pending", "scheduled"}
BAD_STATUSES = {"failed", "cancelled", "unrealized"}
CENT = Decimal("0.01")


def read_csv(name: str) -> list[dict[str, str]]:
	with (DATA / name).open(newline="", encoding="utf-8-sig") as handle:
		return list(csv.DictReader(handle))


def dec(value: str | Decimal | float | int) -> Decimal:
	return Decimal(str(value or "0")).quantize(CENT, rounding=ROUND_HALF_UP)


def fmt(value: Decimal) -> str:
	value = dec(value)
	text = format(value, "f").rstrip("0").rstrip(".")
	return text or "0"


def parse_date(value: str) -> date:
	return date.fromisoformat(value[:10])


def add_months(day: date, months: int) -> date:
	month = day.month - 1 + months
	year, month = day.year + month // 12, month % 12 + 1
	import calendar
	return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def build_rates(rows: list[dict[str, str]]) -> dict[tuple[str, str, date], Decimal]:
	return {(r["from_currency"], r["to_currency"], parse_date(r["rate_date"])): dec(r["rate"]) for r in rows}


def converter(rates: dict[tuple[str, str, date], Decimal], currency: str, home: str, day: date):
	if currency == home:
		return Decimal("1")
	candidates = [(d, rate) for (source, target, d), rate in rates.items()
				  if source == currency and target == home and d <= day]
	if not candidates:
		candidates = [(d, rate) for (source, target, d), rate in rates.items()
					  if source == currency and target == home]
	if candidates:
		return max(candidates, key=lambda pair: pair[0])[1]
	# The supplied data normally includes a direct route. Try a two-leg route.
	for middle in {"USD", "EUR", "ZAR", "IDR", "INR"} - {currency, home}:
		first = converter(rates, currency, middle, day)
		second = converter(rates, middle, home, day)
		if first and second:
			return first * second
	raise ValueError(f"No exchange rate for {currency}->{home} on {day}")


def amount_in(event: dict[str, str], home: str, day: date, rates) -> Decimal:
	return dec(event["amount"]) * converter(rates, event["currency"], home, day)


def image_amounts() -> dict[str, str]:
	"""Read textual amounts from linked images when OCR is available."""
	result: dict[str, str] = {}
	try:
		import pytesseract
		from PIL import Image
	except ImportError:
		return result
	for image in read_csv("images.csv"):
		path = DATA / "media" / "images" / f"{image['image_id']}.png"
		if not path.exists():
			continue
		text = pytesseract.image_to_string(Image.open(path))
		matches = re.findall(r"(?:INR|IDR|ZAR|EUR|USD)\s*([0-9][0-9,.]*)", text, re.I)
		if matches:
			result[image["related_event_id"]] = matches[-1].replace(",", "")
	return result


def normalize_events(events: list[dict[str, str]], rates) -> list[dict[str, str]]:
	image_values = image_amounts()
	for event in events:
		if not event["amount"].strip() and event["event_id"] in image_values:
			event["amount"] = image_values[event["event_id"]]
	return events


def event_is_cash(event: dict[str, str]) -> bool:
	return event["status"] in ACTIVE_STATUSES and event["event_type"] != "investment_valuation"


def future_events(user_events, request_day, home, rates, horizon):
	"""Return known cash movements and inferred recurring movements."""
	known = []
	signatures = defaultdict(list)
	for event in user_events:
		if not event_is_cash(event) or not event["amount"].strip():
			continue
		event_day = parse_date(event["settlement_date"] or event["event_date"])
		if event_day >= request_day - timedelta(days=90):
			signatures[(event["description"], event["category"], event["direction"], event["currency"])].append(event)
		if request_day < event_day <= horizon:
			known.append((event_day, amount_in(event, home, event_day, rates) * (1 if event["direction"] == "credit" else -1), event["event_id"]))

	known_keys = {(day, event_id) for day, _, event_id in known}
	inferred = []
	for signature, history in signatures.items():
		# Two matching records can be a one-off split or duplicate. Require a
		# third observation before projecting an unseen recurring commitment.
		if len(history) < 3:
			continue
		dates = sorted(parse_date(x["settlement_date"] or x["event_date"]) for x in history)
		gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
		gap = round(sum(gaps) / len(gaps))
		if not (5 <= gap <= 35 and max(gaps) - min(gaps) <= 10):
			continue
		last = history[-1]
		next_day = dates[-1] + timedelta(days=gap)
		while next_day <= horizon:
			event_id = f"inferred:{last['event_id']}:{next_day.isoformat()}"
			if next_day >= request_day and (next_day, event_id) not in known_keys:
				value = amount_in(last, home, next_day, rates)
				inferred.append((next_day, value * (1 if last["direction"] == "credit" else -1), event_id))
			next_day += timedelta(days=gap)
	return known + inferred


def simulate(balance, minimum, flows, payments, start, end) -> tuple[bool, Decimal]:
	by_day = defaultdict(Decimal)
	for day, amount, _ in flows:
		if start < day <= end:
			by_day[day] += amount
	for day, amount in payments:
		by_day[day] -= amount
	current = dec(balance)
	lowest = current
	for day in sorted(by_day):
		current += by_day[day]
		lowest = min(lowest, current)
		if current < minimum:
			return False, lowest
	return True, lowest


def plan_text(payments) -> str:
	return "|".join(f"{day.isoformat()}:{fmt(amount)}" for day, amount in payments) if payments else "none"


def main() -> None:
	profiles = {r["user_id"]: r for r in read_csv("financial_profiles.csv")}
	requests_name = os.environ.get("REQUESTS_FILE", "requests.csv")
	output_name = os.environ.get("OUTPUT_FILE", "output.csv")
	requests = read_csv(requests_name)
	events = normalize_events(read_csv("financial_events.csv"), build_rates(read_csv("exchange_rates.csv")))
	rates = build_rates(read_csv("exchange_rates.csv"))
	events_by_user = defaultdict(list)
	for event in events:
		events_by_user[event["user_id"]].append(event)
	options_by_request = defaultdict(list)
	for option in read_csv("request_payment_options.csv"):
		options_by_request[option["request_id"]].append(option)

	results = []
	for request in requests:
		profile = profiles[request["user_id"]]
		home = profile["home_currency"]
		day = parse_date(request["request_date"])
		deadline = parse_date(request["desired_completion_date"])
		horizon = day + timedelta(days=90)
		amount = dec(request["requested_amount"])
		minimum = dec(profile["minimum_balance_to_keep"])
		flows = future_events(events_by_user[request["user_id"]], day, home, rates, horizon)
		safe_today = dec(profile["current_available_balance"]) - minimum
		# Future recurring/confirmed flows are used to find the largest safe debit today.
		ok, _ = simulate(profile["current_available_balance"], minimum, flows, [(day, max(safe_today, Decimal("0")))], day - timedelta(days=1), horizon)
		if not ok:
			low, high = Decimal("0"), max(safe_today, Decimal("0"))
			for _ in range(40):
				mid = (low + high) / 2
				if simulate(profile["current_available_balance"], minimum, flows, [(day, mid)], day - timedelta(days=1), horizon)[0]:
					low = mid
				else:
					high = mid
			safe_today = low
		else:
			safe_today = max(safe_today, Decimal("0"))
		safe_today = min(dec(safe_today), amount)

		def safe_full(on_day):
			return simulate(profile["current_available_balance"], minimum, flows, [(on_day, amount)], day - timedelta(days=1), horizon)[0]

		earliest = day if safe_full(day) else None
		if earliest is None:
			cursor = day + timedelta(days=1)
			while cursor <= min(deadline, horizon):
				if safe_full(cursor):
					earliest = cursor
					break
				cursor += timedelta(days=1)

		spending_change = "none"
		changed_flows = flows
		change_candidates = []
		allowed_changes = set(profile["expense_categories_user_is_willing_to_stop"].split("|")) | set(profile["expense_categories_user_is_willing_to_reduce"].split("|"))
		for event in events_by_user[request["user_id"]]:
			if event["flexibility"] not in {"stoppable", "reducible"} or event["category"] not in allowed_changes:
				continue
			if not event["amount"].strip() or event["category"] in set(profile["expense_categories_to_protect"].split("|")):
				continue
			event_day = parse_date(event["settlement_date"] or event["event_date"])
			if event_day >= day:
				continue
			matching = [flow for flow in flows if flow[2] == event["event_id"] or (event["description"].lower() in next((x["description"].lower() for x in events_by_user[request["user_id"]] if x["event_id"] == flow[2]), ""))]
			if matching:
				candidate_flows = [flow for flow in flows if flow not in matching]
			else:
				candidate_flows = flows
			if simulate(profile["current_available_balance"], minimum, candidate_flows, [(day, amount)], day - timedelta(days=1), horizon)[0]:
				change_candidates.append((event["event_id"], candidate_flows))
		if change_candidates:
			spending_change, changed_flows = f"stop:{change_candidates[0][0]}", change_candidates[0][1]

		method, payments = "not_recommended", []
		status = "not_affordable"
		if earliest == day and "full_payment" in profile["payment_methods_user_will_consider"].split("|"):
			method, payments, status = "full_payment", [(day, amount)], "affordable_now"
		elif spending_change != "none" and "full_payment" in profile["payment_methods_user_will_consider"].split("|") and simulate(profile["current_available_balance"], minimum, changed_flows, [(day, amount)], day - timedelta(days=1), horizon)[0]:
			method, payments, status = "full_payment", [(day, amount)], "affordable_with_plan"
		else:
			for option in sorted(options_by_request[request["request_id"]], key=lambda x: x["payment_option_id"]):
				option_method = option["payment_method"]
				if option_method not in profile["payment_methods_user_will_consider"].split("|"):
					continue
				first = parse_date(option["first_payment_date"])
				count = int(option["number_of_payments"])
				freq = int(option["payment_frequency_days"] or 0)
				option_payments = [(first + timedelta(days=freq * i), dec(option["payment_amount"])) for i in range(count)]
				if option_payments[-1][0] <= deadline and all(d <= horizon for d, _ in option_payments) and simulate(profile["current_available_balance"], minimum, flows, option_payments, day - timedelta(days=1), horizon)[0]:
					method, payments, status = option_method, option_payments, "affordable_with_plan"
					break
			if method == "not_recommended" and earliest and earliest <= deadline and "full_payment" in profile["payment_methods_user_will_consider"].split("|"):
				method, payments, status = "wait", [(earliest, amount)], "affordable_later"
			elif method == "not_recommended" and request["allows_partial_payment"].lower() == "true" and "partial_payment" in profile["payment_methods_user_will_consider"].split("|") and safe_today > 0 and safe_today < amount and earliest and earliest <= deadline:
				method, payments, status = "partial_payment", [(day, safe_today), (earliest, amount - safe_today)], "affordable_with_plan"

		if status == "affordable_now":
			explanation = f"Pay {home} {fmt(amount)} today while keeping at least {home} {fmt(minimum)} available."
		elif status == "affordable_later":
			explanation = f"Wait until {earliest.isoformat()}, when {home} {fmt(amount)} can be paid while keeping the {home} {fmt(minimum)} minimum."
		elif status == "affordable_with_plan":
			explanation = f"Use the recommended {method.replace('_', ' ')} plan; it keeps at least {home} {fmt(minimum)} available."
		else:
			explanation = f"Do not proceed: no eligible plan keeps the {home} {fmt(minimum)} minimum protected through the deadline."
		results.append({
			"request_id": request["request_id"],
			"amount_safe_to_pay": fmt(safe_today),
			"affordability_status": status,
			"recommended_payment_method": method,
			"payment_plan": plan_text(payments),
			"earliest_date_for_full_payment": earliest.isoformat() if earliest else "",
			"spending_changes_needed": spending_change,
			"decision_explanation": explanation,
		})

	with (ROOT / output_name).open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
		writer.writeheader()
		writer.writerows(results)


if __name__ == "__main__":
	main()
