"""
Generate sample policy documents for testing.
"""
import json
from typing import List
from datetime import date, timedelta
import random


POLICY_TEMPLATES = [
    {
        "policy_name": "Group Health Shield Premium",
        "product_type": "GROUP_HEALTH",
        "sections": [
            {
                "name": "Coverage Summary",
                "content": """
COVERAGE SUMMARY

Sum Insured Options: ₹3 Lakhs, ₹5 Lakhs, ₹10 Lakhs, ₹25 Lakhs

This policy covers the following:
- Inpatient Hospitalization (24+ hours)
- Day Care Procedures (less than 24 hours)
- Pre and Post Hospitalization (30 days pre, 60 days post)
- Domiciliary Treatment
- Organ Donor Expenses
- Alternative Treatments (Ayurveda, Homeopathy, Unani)

Room Rent Limits:
- Standard Plan: ₹5,000 per day or 1% of Sum Insured (whichever is lower)
- Enhanced Plan: ₹10,000 per day or 2% of Sum Insured
- Premium Plan: No sub-limit on room rent

ICU Charges: Up to 2x the applicable room rent limit
"""
            },
            {
                "name": "Maternity Benefits",
                "content": """
MATERNITY BENEFITS

Coverage Amount: Up to ₹50,000 per pregnancy

Waiting Period: 36 months from policy inception date

What's Covered:
- Normal Delivery: ₹25,000
- Cesarean Section: ₹50,000
- Pre-natal and Post-natal expenses (within limits)
- New Born Baby Cover: Covered from day 1 for first 90 days

Conditions:
- Coverage applies only to first 2 children
- Must complete continuous coverage for waiting period
- Complications arising from pregnancy are covered under maternity limit
"""
            },
            {
                "name": "Pre-existing Diseases",
                "content": """
PRE-EXISTING DISEASES (PED)

Waiting Period: 48 months for pre-existing conditions

Definition: Any condition, ailment, or injury that existed prior to the first policy inception date, whether or not diagnosed.

Coverage After Waiting Period:
- Full coverage at par with other claims
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
"""
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
9. Treatment outside India
10. Experimental or unproven treatments
11. Treatments for addiction (drugs, alcohol)
12. Congenital conditions (after age 16)

TIME-BOUND EXCLUSIONS (Waiting Period):
- Pre-existing diseases: 48 months
- Maternity: 36 months
- Specific diseases (cataract, hernia, etc.): 24 months
- All other conditions: 30 days
"""
            },
            {
                "name": "Claim Process",
                "content": """
CLAIM PROCESS

CASHLESS CLAIMS:
1. Pre-authorization required at network hospital
2. Submit Employee ID and Policy Number
3. Hospital coordinates with TPA
4. Approval within 2-4 hours for planned procedures
5. Emergency: Post-facto approval within 24 hours

Required Documents:
- Claim form (PART A & B completed)
- Hospital discharge summary
- Medical bills and receipts (originals)
- Investigation reports (X-ray, MRI, blood tests)
- Doctor's prescription and diagnosis
- Photo ID proof

REIMBURSEMENT CLAIMS:
1. Settle hospital bill directly
2. Submit claim within 30 days of discharge
3. Process time: 7-15 working days
4. Payment via NEFT to registered bank account

Claim Settlement Ratio: 96.5% (FY 2023-24)
"""
            },
            {
                "name": "Copay and Deductibles",
                "content": """
CO-PAYMENT AND DEDUCTIBLES

COPAY (Your Share of Claim):
- Standard Plan: 20% copay for all claims
- Enhanced Plan: 10% copay for all claims  
- Premium Plan: No copay

DEDUCTIBLE:
- Per claim deductible: ₹5,000 (Standard), ₹2,500 (Enhanced), Nil (Premium)
- Applied once per hospitalization

SPECIAL CONDITIONS:
- Claims above ₹10 Lakhs: Additional 10% copay
- Treatment in non-network hospital: 20% additional copay
- Room upgrade beyond entitlement: Proportional deduction

Example Calculation:
Claim Amount: ₹1,00,000
Deductible (Standard): ₹5,000
After Deductible: ₹95,000
Copay (20%): ₹19,000
Net Payable by Insurer: ₹76,000
"""
            }
        ]
    },
    {
        "policy_name": "Corporate Health Guard",
        "product_type": "GROUP_HEALTH",
        "sections": [
            {
                "name": "Overview",
                "content": """
CORPORATE HEALTH GUARD - OVERVIEW

This Group Health Insurance policy provides comprehensive medical coverage to employees and their dependents.

Eligible Members:
- Employee (Primary Insured)
- Spouse
- Up to 2 dependent children (up to age 25)
- Dependent parents (optional add-on)

Entry Age:
- Minimum: 18 years
- Maximum: 65 years (new enrollment)
- Continuation: Up to 80 years

Policy Period: Annual (April to March or customized)
"""
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
"""
            }
        ]
    }
]


def generate_policy_document(policy_name: str, template_index: int = 0) -> str:
    """Generate a full policy document from template."""
    template = POLICY_TEMPLATES[template_index % len(POLICY_TEMPLATES)]
    
    doc_parts = [f"# {policy_name}\n\n"]
    
    for section in template["sections"]:
        doc_parts.append(f"## {section['name']}\n")
        doc_parts.append(section["content"].strip())
        doc_parts.append("\n\n")
    
    return "\n".join(doc_parts)


def generate_sample_policies(count: int = 5) -> List[dict]:
    """Generate multiple sample policies for database seeding."""
    policies = []
    companies = ["Acme Corp", "Tech Solutions", "Global Industries", "HealthFirst Inc", "Innovate Labs"]
    
    for i in range(count):
        template = POLICY_TEMPLATES[i % len(POLICY_TEMPLATES)]
        start_date = date.today() - timedelta(days=random.randint(30, 365))
        
        policies.append({
            "policy_number": f"POL{2024000 + i}",
            "policy_name": f"{companies[i % len(companies)]} - {template['policy_name']}",
            "product_type": template["product_type"],
            "insured_name": companies[i % len(companies)],
            "policy_start_date": start_date,
            "policy_end_date": start_date + timedelta(days=365),
            "total_lives": random.randint(50, 500),
            "total_sum_insured": random.randint(5, 50) * 1000000,
            "premium_amount": random.randint(10, 100) * 100000,
            "document_text": generate_policy_document(template["policy_name"], i)
        })
    
    return policies


if __name__ == "__main__":
    # Generate sample output
    policies = generate_sample_policies(2)
    for p in policies:
        print(f"Policy: {p['policy_name']}")
        print(f"Lives: {p['total_lives']}")
        print("-" * 50)
