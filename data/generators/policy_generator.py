"""
Generate sample policy documents for testing.
"""

import random
from datetime import date, timedelta
from typing import List

POLICY_TEMPLATES = [
    {
        "contract_title": "Northstar Benefits Plus",
        "plan_category": "EMPLOYEE_BENEFITS",
        "sections": [
            {
                "name": "Coverage Summary",
                "content": """
COVERAGE SUMMARY

Benefit Cap Options: CU 300,000, CU 500,000, CU 1,000,000, CU 2,500,000

This policy covers the following:
- Facility stays (24+ hours)
- Day Care Procedures (less than 24 hours)
- Pre- and post-service support (30 days before, 60 days after)
- Domiciliary Treatment
- Organ Donor Expenses
- Approved complementary treatments

Room Rent Limits:
- Standard Plan: CU 5,000 per day or 1% of the benefit cap (whichever is lower)
- Enhanced Plan: CU 10,000 per day or 2% of the benefit cap
- Premium Plan: No sub-limit on room rent

ICU Charges: Up to 2x the applicable room rent limit
""",
            },
            {
                "name": "Maternity Benefits",
                "content": """
MATERNITY BENEFITS

Benefit Amount: Up to CU 50,000 per event

Waiting Period: 36 months from policy inception date

What's Covered:
- Standard service: CU 25,000
- Surgical service: CU 50,000
- Pre-natal and Post-natal expenses (within limits)
- New Born Baby Cover: Covered from day 1 for first 90 days

Conditions:
- Coverage applies only to first 2 children
- Must complete continuous coverage for waiting period
- Complications arising from pregnancy are covered under maternity limit
""",
            },
            {
                "name": "Pre-existing Diseases",
                "content": """
PRE-EXISTING DISEASES (PED)

Waiting Period: 48 months for pre-existing conditions

Definition: Any condition, ailment, or injury that existed prior to the first policy inception date, whether or not diagnosed.

Coverage After Waiting Period:
- Full coverage at par with other service_cases
- No additional deductible or co-payment

Disclosure Requirements:
- All known conditions must be disclosed at enrollment
- Non-disclosure may result in claim rejection or policy cancellation

Chronic Conditions Covered (after waiting):
- Diabetes
- Hypertension
- Heart Disease
- Thyroid Disorders
- Asthma
""",
            },
            {
                "name": "Exclusions",
                "content": """
PERMANENT EXCLUSIONS

The following are NOT covered under this policy:

1. Cosmetic or aesthetic treatments
2. Treatment for obesity/weight control
3. Dental treatments (unless requiring hospitalization)
4. Fertility treatments, IVF, IUI
5. STDs, AIDS/HIV (except through medical procedures)
6. Self-inflicted injuries or suicide attempts
7. War, nuclear contamination
8. Participation in hazardous sports/activities
9. Treatment outside the covered region
10. Experimental or unproven treatments
11. Treatments for addiction (drugs, alcohol)
12. Congenital conditions (after age 16)

TIME-BOUND EXCLUSIONS (Waiting Period):
- Pre-existing diseases: 48 months
- Maternity: 36 months
- Specific diseases (cataract, hernia, etc.): 24 months
- All other conditions: 30 days
""",
            },
            {
                "name": "Claim Process",
                "content": """
SERVICE-CASE PROCESS

DIRECT-BILLING CASES:
1. Ask a participating provider to open a service case
2. Share the participant access code and contract reference
3. The provider sends a benefit-check request to the service partner
4. Planned services receive a response by the next business day
5. Urgent services may be reviewed after stabilization

Required Documents:
- Service-case form
- Provider completion summary
- Medical bills and receipts (originals)
- Investigation reports (X-ray, MRI, blood tests)
- Doctor's prescription and diagnosis
- Participant access-code confirmation

MEMBER-PAID CASES:
1. Pay the provider directly
2. Submit the service case within 45 days of completion
3. Typical review window: 5-12 business days
4. Approved amounts are sent through the configured payout method
""",
            },
            {
                "name": "Copay and Deductibles",
                "content": """
CO-PAYMENT AND DEDUCTIBLES

COPAY (Your Share of Claim):
- Standard Plan: 20% copay for all service_cases
- Enhanced Plan: 10% percentage share for all service cases
- Premium Plan: No copay

DEDUCTIBLE:
- Per-case fixed share: CU 5,000 (Standard), CU 2,500 (Enhanced), none (Premium)
- Applied once per hospitalization

SPECIAL CONDITIONS:
- Cases above CU 1,000,000: additional 10% member share
- Service from a non-participating provider: additional 20% member share
- Room upgrade beyond entitlement: Proportional deduction

Example Calculation:
Requested Amount: CU 100,000
Fixed Share (Standard): CU 5,000
After Fixed Share: CU 95,000
Percentage Share (20%): CU 19,000
Payable Amount: CU 76,000
""",
            },
        ],
    },
    {
        "contract_title": "Cedar Workforce Benefits",
        "plan_category": "EMPLOYEE_BENEFITS",
        "sections": [
            {
                "name": "Overview",
                "content": """
CEDAR WORKFORCE BENEFITS - OVERVIEW

This fictional workforce benefit contract provides medical support to participants and eligible dependants.

Eligible Members:
- Participant (primary enrollee)
- Spouse
- Up to 2 dependent children (up to age 25)
- Dependent parents (optional add-on)

Entry Age:
- Minimum: 18 years
- Maximum: 65 years (new enrollment)
- Continuation: Up to 80 years

Contract Period: Twelve months from the configured effective date
""",
            },
            {
                "name": "Critical Illness",
                "content": """
CRITICAL ILLNESS COVER

Additional lump sum benefit for diagnosis of specified critical illnesses.

Covered Conditions (30 days survival required):
1. Cancer (all stages except early stage)
2. Heart Attack (First occurrence)
3. Stroke with permanent symptoms
4. Kidney Failure requiring dialysis
5. Major Organ Transplant
6. Coronary Artery Bypass Surgery
7. Aorta Surgery
8. Paralysis of Limbs

Benefit Amount: 100% of Critical Illness Sum Insured
Waiting Period: 90 days from policy start

This is a one-time benefit and does not reduce the base health cover.
""",
            },
        ],
    },
]


def generate_contract_document(contract_title: str, template_index: int = 0) -> str:
    """Generate a full policy document from template."""
    template = POLICY_TEMPLATES[template_index % len(POLICY_TEMPLATES)]

    doc_parts = [f"# {contract_title}\n\n"]

    for section in template["sections"]:
        doc_parts.append(f"## {section['name']}\n")
        doc_parts.append(section["content"].strip())
        doc_parts.append("\n\n")

    return "\n".join(doc_parts)


def generate_sample_benefit_contracts(count: int = 5) -> List[dict]:
    """Generate multiple sample benefit_contracts for database seeding."""
    benefit_contracts = []
    sponsors = [
        "Blue Meadow Studio",
        "Northwind Works",
        "Cedar Labs",
        "Juniper Cooperative",
        "Lakehouse Design",
    ]

    for i in range(count):
        template = POLICY_TEMPLATES[i % len(POLICY_TEMPLATES)]
        start_date = date.today() - timedelta(days=random.randint(30, 365))

        benefit_contracts.append(
            {
                "contract_ref": f"BEN-{730000 + i}",
                "contract_title": f"{sponsors[i % len(sponsors)]} - {template['contract_title']}",
                "plan_category": template["plan_category"],
                "sponsor_label": sponsors[i % len(sponsors)],
                "effective_from": start_date,
                "effective_until": start_date + timedelta(days=365),
                "participant_count": random.randint(50, 500),
                "aggregate_benefit_cap": random.randint(5, 50) * 1000000,
                "contribution_amount": random.randint(10, 100) * 100000,
                "source_text": generate_contract_document(template["contract_title"], i),
            }
        )

    return benefit_contracts


if __name__ == "__main__":
    # Generate sample output
    benefit_contracts = generate_sample_benefit_contracts(2)
    for p in benefit_contracts:
        print(f"Policy: {p['contract_title']}")
        print(f"Lives: {p['participant_count']}")
        print("-" * 50)
