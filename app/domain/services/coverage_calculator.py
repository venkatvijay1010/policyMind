"""
Deterministic coverage calculator - never trust LLM for math.
"""
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class CoverageCalculation:
    """Result of a coverage calculation."""
    gross_amount: Decimal
    deductible: Decimal
    copay_amount: Decimal
    sub_limit_applied: Decimal
    net_payable: Decimal
    calculation_breakdown: str


@dataclass
class RoomRentResult:
    """Result of room rent calculation."""
    allowed_amount: float
    excess: float
    proportional_deduction: float


@dataclass
class ClaimCalculationResult:
    """Result of full claim calculation."""
    claim_amount: float
    allowed_claim: float
    deductible: float
    copay: float
    room_rent_deduction: float
    net_payable: float


# Plan-specific configurations
PLAN_CONFIGS = {
    "standard": {
        "room_rent_percent": 0.01,  # 1% of SI
        "room_rent_daily_limit": 5000,
        "copay_percent": 0.20,  # 20%
        "deductible": 5000,
    },
    "enhanced": {
        "room_rent_percent": 0.02,  # 2% of SI
        "room_rent_daily_limit": 10000,
        "copay_percent": 0.10,  # 10%
        "deductible": 2500,
    },
    "premium": {
        "room_rent_percent": None,  # No limit
        "room_rent_daily_limit": None,
        "copay_percent": 0.0,  # No copay
        "deductible": 0,
    },
}


class CoverageCalculator:
    """
    Deterministic calculator for coverage-related math.
    
    IMPORTANT: Never use LLM for these calculations.
    LLMs are unreliable for arithmetic.
    """
    
    @staticmethod
    def calculate_room_rent_limit(
        sum_insured: Decimal,
        room_rent_percentage: Decimal = Decimal("0.01"),
        actual_rent: Decimal = Decimal("0")
    ) -> Decimal:
        """
        Calculate room rent limit.
        Typically 1% of Sum Insured per day or actual, whichever is lower.
        """
        limit = sum_insured * room_rent_percentage
        if actual_rent > 0:
            return min(limit, actual_rent)
        return limit.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    
    @staticmethod
    def calculate_icu_limit(
        sum_insured: Decimal,
        icu_percentage: Decimal = Decimal("0.02")
    ) -> Decimal:
        """
        Calculate ICU charges limit.
        Typically 2% of Sum Insured per day.
        """
        return (sum_insured * icu_percentage).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    
    @staticmethod
    def calculate_copay(
        claim_amount: Decimal,
        copay_percentage: Decimal,
        deductible: Decimal = Decimal("0")
    ) -> Decimal:
        """
        Calculate co-payment amount.
        Co-pay is applied on (claim_amount - deductible).
        """
        amount_after_deductible = max(claim_amount - deductible, Decimal("0"))
        copay = amount_after_deductible * (copay_percentage / Decimal("100"))
        return copay.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    
    @staticmethod
    def calculate_net_payable(
        claim_amount: Decimal,
        deductible: Decimal = Decimal("0"),
        copay_percentage: Decimal = Decimal("0"),
        sub_limit: Optional[Decimal] = None
    ) -> CoverageCalculation:
        """
        Calculate net payable amount after all deductions.
        
        Order of deductions:
        1. Apply sub-limit (if any)
        2. Deduct deductible
        3. Apply co-pay on remaining
        """
        # Step 1: Apply sub-limit
        if sub_limit and claim_amount > sub_limit:
            amount_after_sublimit = sub_limit
            sublimit_applied = claim_amount - sub_limit
        else:
            amount_after_sublimit = claim_amount
            sublimit_applied = Decimal("0")
        
        # Step 2: Deduct deductible
        amount_after_deductible = max(amount_after_sublimit - deductible, Decimal("0"))
        
        # Step 3: Calculate co-pay
        copay_amount = Decimal("0")
        if copay_percentage > 0:
            copay_amount = amount_after_deductible * (copay_percentage / Decimal("100"))
        
        # Final net payable
        net_payable = amount_after_deductible - copay_amount
        net_payable = net_payable.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        
        # Build breakdown
        breakdown = f"""
Claim Amount: ₹{claim_amount:,.2f}
- Sub-limit Applied: ₹{sublimit_applied:,.2f}
= After Sub-limit: ₹{amount_after_sublimit:,.2f}
- Deductible: ₹{deductible:,.2f}
= After Deductible: ₹{amount_after_deductible:,.2f}
- Co-pay ({copay_percentage}%): ₹{copay_amount:,.2f}
= Net Payable: ₹{net_payable:,.2f}
        """.strip()
        
        return CoverageCalculation(
            gross_amount=claim_amount,
            deductible=deductible,
            copay_amount=copay_amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            sub_limit_applied=sublimit_applied,
            net_payable=net_payable,
            calculation_breakdown=breakdown
        )
    
    @staticmethod
    def calculate_aggregate_si_usage(
        sum_insured: Decimal,
        claims_paid: List[Decimal]
    ) -> dict:
        """
        Calculate aggregate Sum Insured usage.
        """
        total_claimed = sum(claims_paid)
        remaining = max(sum_insured - total_claimed, Decimal("0"))
        usage_percentage = (total_claimed / sum_insured * 100) if sum_insured > 0 else Decimal("0")
        
        return {
            "sum_insured": float(sum_insured),
            "total_claimed": float(total_claimed),
            "remaining": float(remaining),
            "usage_percentage": float(usage_percentage.quantize(Decimal("0.01"))),
            "is_exhausted": remaining <= 0
        }
    
    @staticmethod
    def is_within_waiting_period(
        policy_start_date,
        claim_date,
        waiting_period_days: int
    ) -> dict:
        """
        Check if a claim falls within the waiting period.
        """
        from datetime import date, timedelta
        
        if isinstance(policy_start_date, str):
            policy_start_date = date.fromisoformat(policy_start_date)
        if isinstance(claim_date, str):
            claim_date = date.fromisoformat(claim_date)
        
        days_since_inception = (claim_date - policy_start_date).days
        waiting_period_end = policy_start_date + timedelta(days=waiting_period_days)
        is_within = days_since_inception < waiting_period_days
        
        return {
            "is_within_waiting_period": is_within,
            "days_since_inception": days_since_inception,
            "waiting_period_days": waiting_period_days,
            "waiting_period_end_date": waiting_period_end.isoformat(),
            "days_remaining": max(waiting_period_days - days_since_inception, 0) if is_within else 0
        }
    
    def calculate_room_rent(
        self,
        actual_room_rent: float,
        sum_insured: float,
        plan_type: str = "standard",
        daily_limit: Optional[float] = None
    ) -> RoomRentResult:
        """
        Calculate room rent allowed based on plan type.
        """
        plan = PLAN_CONFIGS.get(plan_type.lower(), PLAN_CONFIGS["standard"])
        
        # Premium plan has no limit
        if plan["room_rent_percent"] is None:
            return RoomRentResult(
                allowed_amount=actual_room_rent,
                excess=0,
                proportional_deduction=1.0
            )
        
        # Calculate limit based on percentage of SI
        percent_limit = sum_insured * plan["room_rent_percent"]
        
        # Use provided daily limit or plan default
        fixed_limit = daily_limit or plan["room_rent_daily_limit"]
        
        # Final limit is lower of percentage and fixed limit
        if fixed_limit:
            room_rent_limit = min(percent_limit, fixed_limit)
        else:
            room_rent_limit = percent_limit
        
        # Calculate allowed amount and excess
        allowed = min(actual_room_rent, room_rent_limit)
        excess = max(0, actual_room_rent - room_rent_limit)
        
        # Proportional deduction factor
        proportional = allowed / actual_room_rent if actual_room_rent > 0 else 1.0
        
        return RoomRentResult(
            allowed_amount=allowed,
            excess=excess,
            proportional_deduction=proportional
        )
    
    def calculate_icu_charges(
        self,
        actual_icu_rate: float,
        room_rent_limit: float
    ) -> dict:
        """
        Calculate ICU charges (typically 2x room rent limit).
        """
        icu_limit = room_rent_limit * 2
        allowed = min(actual_icu_rate, icu_limit)
        excess = max(0, actual_icu_rate - icu_limit)
        
        return {
            "allowed_amount": allowed,
            "excess": excess,
            "icu_limit": icu_limit
        }
    
    def calculate_copay(
        self,
        claim_amount: float,
        plan_type: str = "standard",
        is_network_hospital: bool = True
    ) -> float:
        """
        Calculate copay based on plan type and hospital network.
        """
        plan = PLAN_CONFIGS.get(plan_type.lower(), PLAN_CONFIGS["standard"])
        
        copay = claim_amount * plan["copay_percent"]
        
        # Non-network hospital: additional 20% copay
        if not is_network_hospital:
            copay += claim_amount * 0.20
        
        # High-value claim (> 10 lakhs): additional 10% copay
        if claim_amount > 1000000:
            copay += claim_amount * 0.10
        
        return copay
    
    def calculate_deductible(self, plan_type: str = "standard") -> float:
        """
        Get deductible amount for plan type.
        """
        plan = PLAN_CONFIGS.get(plan_type.lower(), PLAN_CONFIGS["standard"])
        return float(plan["deductible"])
    
    def calculate_net_payable(
        self,
        claim_amount: float,
        plan_type: str = "standard",
        sum_insured: float = 500000,
        actual_room_rent: Optional[float] = None,
        room_rent_limit: Optional[float] = None
    ) -> ClaimCalculationResult:
        """
        Calculate net payable after all deductions.
        """
        plan = PLAN_CONFIGS.get(plan_type.lower(), PLAN_CONFIGS["standard"])
        
        # Cap at sum insured
        allowed_claim = min(claim_amount, sum_insured)
        
        # Calculate deductible
        deductible = float(plan["deductible"])
        after_deductible = max(0, allowed_claim - deductible)
        
        # Calculate copay
        copay = after_deductible * plan["copay_percent"]
        
        # Room rent proportional deduction
        room_rent_deduction = 0.0
        if actual_room_rent and room_rent_limit and actual_room_rent > room_rent_limit:
            proportion = room_rent_limit / actual_room_rent
            room_rent_deduction = after_deductible * (1 - proportion)
        
        # Final net payable
        net_payable = max(0, after_deductible - copay - room_rent_deduction)
        
        return ClaimCalculationResult(
            claim_amount=claim_amount,
            allowed_claim=allowed_claim,
            deductible=deductible,
            copay=copay,
            room_rent_deduction=room_rent_deduction,
            net_payable=net_payable
        )
