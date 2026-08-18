"""
Generate synthetic service_cases data for testing.
"""

import random
from datetime import date, timedelta
from typing import List, Optional

# ICD-10 codes for common conditions
ICD_CODES = [
    ("A09", "Infectious gastroenteritis", "Gastro", False, 3, 25000),
    ("J18.9", "Pneumonia", "Respiratory", False, 5, 75000),
    ("K35.8", "Appendicitis", "Digestive", False, 4, 80000),
    ("E11.9", "Type 2 Diabetes Mellitus", "Metabolic", True, 5, 50000),
    ("I10", "Hypertension", "Cardiovascular", True, 3, 30000),
    ("I21.9", "Acute Myocardial Infarction", "Cardiovascular", False, 10, 350000),
    ("O80", "Normal Delivery", "Maternity", False, 3, 25000),
    ("O82", "Cesarean Section", "Maternity", False, 5, 50000),
    ("S72.0", "Hip Fracture", "Trauma", False, 7, 150000),
    ("M54.5", "Lower Back Pain", "Musculoskeletal", True, 4, 40000),
    ("N20.0", "Kidney Stone", "Urological", False, 2, 35000),
    ("H25.9", "Cataract", "Ophthalmological", False, 1, 30000),
    ("K80.2", "Gallstones", "Digestive", False, 4, 90000),
    ("C50.9", "Breast Cancer", "Oncology", False, 14, 500000),
    ("J45.9", "Asthma", "Respiratory", True, 3, 25000),
]

CARE_PROVIDERS = [
    ("Northstar Medical Center", "Lakeview", "North District", True),
    ("Willow Community Clinic", "Fairhaven", "West District", True),
    ("Summit Regional Center", "Brookfield", "Central District", True),
    ("Harborview Clinic", "Riverton", "Coastal District", True),
    ("Pinecrest Medical Center", "Oakridge", "East District", True),
    ("Meadow General Clinic", "Clearwater", "South District", False),
    ("Cedar District Center", "Hillcrest", "North District", False),
    ("Juniper Health Center", "Mapleton", "West District", True),
]

CASE_STATUSES = ["OPENED", "IN_REVIEW", "ELIGIBLE", "RESOLVED", "DECLINED"]

DECISION_REASONS = [
    "Eligibility delay has not elapsed",
    "Required background information was not supplied",
    "Service is outside the contract terms",
    "Case was submitted after the stated deadline",
    "Missing documentation",
    "Non-participating provider used without advance review",
    "Service is listed as non-covered",
]


def generate_service_cases(
    contract_id: int,
    participant_ids: List[int],
    count: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[dict]:
    """
    Generate synthetic service_cases data.

    Args:
        contract_id: ID of the policy
        participant_ids: Synthetic participant row IDs to assign service cases to
        count: Number of service_cases to generate
        start_date: Start of date range
        end_date: End of date range
    """
    if not start_date:
        start_date = date.today() - timedelta(days=365)
    if not end_date:
        end_date = date.today()

    service_cases = []
    date_range = (end_date - start_date).days

    for i in range(count):
        # Pick random ICD code
        icd = random.choice(ICD_CODES)
        code, description, category, is_chronic, typical_days, typical_cost = icd

        # Pick random hospital
        provider = random.choice(CARE_PROVIDERS)
        provider_label, city, state, is_network = provider

        # Generate dates
        admission = start_date + timedelta(days=random.randint(0, date_range))
        los = max(1, typical_days + random.randint(-2, 3))  # Length of stay
        discharge = admission + timedelta(days=los)

        # Generate amounts
        base_amount = typical_cost * (1 + random.uniform(-0.3, 0.5))
        requested_amount = round(base_amount, -2)  # Round to nearest 100

        # Determine claim status with weighted probabilities
        status = random.choices(
            CASE_STATUSES,
            weights=[5, 10, 20, 55, 10],  # Most are resolved
            k=1,
        )[0]

        # Calculate the eligible amount when applicable
        eligible_amount = 0
        deductible = 0
        copay = 0
        payable_amount = 0
        decision_reason = None
        resolved_on = None

        if status in ["ELIGIBLE", "RESOLVED"]:
            # Apply deductible
            deductible = 5000
            after_deductible = max(0, requested_amount - deductible)

            # Apply copay (20%)
            copay = round(after_deductible * 0.2, -2)

            # Non-network penalty
            if not is_network:
                copay += round(after_deductible * 0.2, -2)

            eligible_amount = requested_amount
            payable_amount = round(requested_amount - deductible - copay, -2)

            if status == "RESOLVED":
                resolved_on = discharge + timedelta(days=random.randint(7, 30))

        elif status == "DECLINED":
            decision_reason = random.choice(DECISION_REASONS)

        funding_mode = random.choice(["DIRECT_BILLING", "MEMBER_PAID"])
        care_setting = "FACILITY_STAY" if los > 0 else random.choice(["CLINIC", "SAME_DAY"])

        service_cases.append(
            {
                "case_ref": f"CASE-{contract_id:04d}-{i + 1:04d}",
                "contract_id": contract_id,
                "participant_id": random.choice(participant_ids),
                "funding_mode": funding_mode,
                "care_setting": care_setting,
                "condition_code": code,
                "condition_label": description,
                "service_category": category,
                "provider_label": provider_label,
                "provider_city": city,
                "provider_region": state,
                "service_started_on": admission,
                "service_ended_on": discharge,
                "requested_amount": requested_amount,
                "eligible_amount": eligible_amount,
                "fixed_share_applied": deductible,
                "percentage_share_applied": copay,
                "payable_amount": payable_amount,
                "case_status": status,
                "submitted_on": admission,
                "resolved_on": resolved_on,
                "decision_reason": decision_reason,
            }
        )

    return service_cases


def generate_participants(contract_id: int, count: int = 20) -> List[dict]:
    """Generate synthetic member data."""
    participants = []

    first_names = ["Alex", "Morgan", "Casey", "Jordan", "Taylor", "Riley", "Avery", "Quinn"]
    last_names = ["Reed", "Blake", "Hayes", "Parker", "Rowan", "Ellis", "Lane", "Morgan"]
    cities = ["Lakeview", "Fairhaven", "Brookfield", "Riverton", "Oakridge", "Clearwater"]
    states = [
        "North District",
        "West District",
        "Central District",
        "Coastal District",
        "East District",
        "South District",
    ]
    enrolment_roles = ["SELF", "SPOUSE", "CHILD", "PARENT"]

    for i in range(count):
        enrolment_role = random.choices(enrolment_roles, weights=[40, 30, 20, 10], k=1)[0]

        age = (
            random.randint(25, 60)
            if enrolment_role == "SELF"
            else (
                random.randint(25, 55)
                if enrolment_role == "SPOUSE"
                else random.randint(1, 22)
                if enrolment_role == "CHILD"
                else random.randint(55, 80)
            )
        )

        gender = random.choice(["M", "F"])
        city_idx = random.randint(0, len(cities) - 1)

        participants.append(
            {
                "participant_ref": f"PART-{contract_id:04d}-{i + 1:04d}",
                "contract_id": contract_id,
                "participant_label": f"{random.choice(first_names)} {random.choice(last_names)}",
                "enrolment_role": enrolment_role,
                "gender": gender,
                "birth_date": date.today() - timedelta(days=age * 365 + random.randint(0, 365)),
                "age": age,
                "benefit_ceiling": random.choice([300000, 500000, 1000000, 2500000]),
                "status": "ACTIVE",
                "city": cities[city_idx],
                "state": states[city_idx],
            }
        )

    return participants


if __name__ == "__main__":
    # Test generation
    participants = generate_participants(1, 10)
    participant_ids = list(range(1, 11))
    service_cases = generate_service_cases(1, participant_ids, 20)

    print(f"Generated {len(participants)} participants")
    print(f"Generated {len(service_cases)} service_cases")

    # Status breakdown
    from collections import Counter

    status_counts = Counter(c["case_status"] for c in service_cases)
    print("\nCase Status Distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
