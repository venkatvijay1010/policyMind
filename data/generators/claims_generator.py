"""
Generate synthetic claims data for testing.
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

HOSPITALS = [
    ("Apollo Hospital", "Chennai", "Tamil Nadu", True),
    ("Fortis Hospital", "Mumbai", "Maharashtra", True),
    ("Max Hospital", "Delhi", "Delhi", True),
    ("Manipal Hospital", "Bangalore", "Karnataka", True),
    ("AIIMS", "Delhi", "Delhi", True),
    ("City General Hospital", "Hyderabad", "Telangana", False),
    ("District Hospital", "Pune", "Maharashtra", False),
    ("Medicare Hospital", "Kolkata", "West Bengal", True),
    ("Narayana Health", "Bangalore", "Karnataka", True),
    ("Kokilaben Hospital", "Mumbai", "Maharashtra", True),
]

CLAIM_STATUSES = ["REGISTERED", "UNDER_PROCESS", "APPROVED", "SETTLED", "REJECTED"]

REJECTION_REASONS = [
    "Pre-existing condition - waiting period not completed",
    "Non-disclosure of medical history",
    "Treatment not covered under policy",
    "Claim submitted after 30 days deadline",
    "Missing documentation",
    "Treatment at non-network hospital without pre-authorization",
    "Cosmetic procedure - permanent exclusion",
    "Waiting period for specific disease not completed",
]


def generate_claims(
    policy_id: int,
    member_ids: List[int],
    count: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> List[dict]:
    """
    Generate synthetic claims data.
    
    Args:
        policy_id: ID of the policy
        member_ids: List of member IDs to assign claims to
        count: Number of claims to generate
        start_date: Start of date range
        end_date: End of date range
    """
    if not start_date:
        start_date = date.today() - timedelta(days=365)
    if not end_date:
        end_date = date.today()
    
    claims = []
    date_range = (end_date - start_date).days
    
    for i in range(count):
        # Pick random ICD code
        icd = random.choice(ICD_CODES)
        code, description, category, is_chronic, typical_days, typical_cost = icd
        
        # Pick random hospital
        hospital = random.choice(HOSPITALS)
        hospital_name, city, state, is_network = hospital
        
        # Generate dates
        admission = start_date + timedelta(days=random.randint(0, date_range))
        los = max(1, typical_days + random.randint(-2, 3))  # Length of stay
        discharge = admission + timedelta(days=los)
        
        # Generate amounts
        base_amount = typical_cost * (1 + random.uniform(-0.3, 0.5))
        claim_amount = round(base_amount, -2)  # Round to nearest 100
        
        # Determine claim status with weighted probabilities
        status = random.choices(
            CLAIM_STATUSES,
            weights=[5, 10, 20, 55, 10],  # Most are settled
            k=1
        )[0]
        
        # Calculate approved amount (if applicable)
        approved_amount = 0
        deductible = 0
        copay = 0
        net_payable = 0
        rejection_reason = None
        settlement_date = None
        
        if status in ["APPROVED", "SETTLED"]:
            # Apply deductible
            deductible = 5000
            after_deductible = max(0, claim_amount - deductible)
            
            # Apply copay (20%)
            copay = round(after_deductible * 0.2, -2)
            
            # Non-network penalty
            if not is_network:
                copay += round(after_deductible * 0.2, -2)
            
            approved_amount = claim_amount
            net_payable = round(claim_amount - deductible - copay, -2)
            
            if status == "SETTLED":
                settlement_date = discharge + timedelta(days=random.randint(7, 30))
        
        elif status == "REJECTED":
            rejection_reason = random.choice(REJECTION_REASONS)
        
        claim_type = random.choice(["CASHLESS", "REIMBURSEMENT"])
        claim_category = "IPD" if los > 0 else random.choice(["OPD", "DAYCARE"])
        
        claims.append({
            "claim_number": f"CLM{2024000000 + i}",
            "policy_id": policy_id,
            "member_id": random.choice(member_ids),
            "claim_type": claim_type,
            "claim_category": claim_category,
            "diagnosis_code": code,
            "diagnosis_description": description,
            "treatment_type": category,
            "hospital_name": hospital_name,
            "hospital_city": city,
            "hospital_state": state,
            "admission_date": admission,
            "discharge_date": discharge,
            "claim_amount": claim_amount,
            "approved_amount": approved_amount,
            "deductible_applied": deductible,
            "copay_applied": copay,
            "net_payable": net_payable,
            "claim_status": status,
            "registration_date": admission,
            "settlement_date": settlement_date,
            "rejection_reason": rejection_reason
        })
    
    return claims


def generate_members(
    policy_id: int,
    count: int = 20
) -> List[dict]:
    """Generate synthetic member data."""
    members = []
    
    first_names = ["Raj", "Priya", "Amit", "Sneha", "Vikram", "Anita", "Kiran", "Meera", "Arjun", "Kavya"]
    last_names = ["Sharma", "Patel", "Kumar", "Singh", "Gupta", "Reddy", "Nair", "Joshi", "Iyer", "Das"]
    cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata", "Ahmedabad"]
    states = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Telangana", "Maharashtra", "West Bengal", "Gujarat"]
    relationships = ["SELF", "SPOUSE", "CHILD", "PARENT"]
    
    for i in range(count):
        relationship = random.choices(
            relationships,
            weights=[40, 30, 20, 10],
            k=1
        )[0]
        
        age = random.randint(25, 60) if relationship == "SELF" else (
            random.randint(25, 55) if relationship == "SPOUSE" else
            random.randint(1, 22) if relationship == "CHILD" else
            random.randint(55, 80)
        )
        
        gender = random.choice(["M", "F"])
        city_idx = random.randint(0, len(cities) - 1)
        
        members.append({
            "member_id": f"MEM{1000 + i}",
            "policy_id": policy_id,
            "member_name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "relationship": relationship,
            "gender": gender,
            "date_of_birth": date.today() - timedelta(days=age * 365 + random.randint(0, 365)),
            "age": age,
            "sum_insured": random.choice([300000, 500000, 1000000, 2500000]),
            "status": "ACTIVE",
            "city": cities[city_idx],
            "state": states[city_idx]
        })
    
    return members


if __name__ == "__main__":
    # Test generation
    members = generate_members(1, 10)
    member_ids = list(range(1, 11))
    claims = generate_claims(1, member_ids, 20)
    
    print(f"Generated {len(members)} members")
    print(f"Generated {len(claims)} claims")
    
    # Status breakdown
    from collections import Counter
    status_counts = Counter(c["claim_status"] for c in claims)
    print("\nClaim Status Distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
