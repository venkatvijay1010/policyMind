"""
Deterministic coverage calculator - never trust LLM for math.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import List, Optional, TypedDict


@dataclass
class CoverageCalculation:
    """Result of a coverage calculation."""

    gross_amount: Decimal
    deductible: Decimal
    copay_amount: Decimal
    sub_limit_applied: Decimal
    payable_amount: Decimal
    calculation_breakdown: str


@dataclass
class RoomRentResult:
    """Result of room rent calculation."""

    allowed_amount: float
    excess: float
    proportional_deduction: float


@dataclass
class ClaimCalculationResult:
    """Result of a synthetic service-case calculation."""

    requested_amount: float
    eligible_base: float
    deductible: float
    copay: float
    room_rent_deduction: float
    payable_amount: float


class PlanConfig(TypedDict):
    """Calculation settings for a supported synthetic plan."""

    room_rent_percent: float | None
    room_rent_daily_limit: float | None
    copay_percent: float
    deductible: float


# Plan-specific configurations
PLAN_CONFIGS: dict[str, PlanConfig] = {
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
        benefit_ceiling: Decimal,
        room_rent_percentage: Decimal = Decimal("0.01"),
        actual_rent: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Calculate room rent limit.
        Typically 1% of Sum Insured per day or actual, whichever is lower.
        """
        limit = benefit_ceiling * room_rent_percentage
        if actual_rent > 0:
            return min(limit, actual_rent)
        return limit.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @staticmethod
    def calculate_icu_limit(
        benefit_ceiling: Decimal, icu_percentage: Decimal = Decimal("0.02")
    ) -> Decimal:
        """
        Calculate ICU charges limit.
        Typically 2% of Sum Insured per day.
        """
        return (benefit_ceiling * icu_percentage).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @staticmethod
    def calculate_policy_copay(
        requested_amount: Decimal, percentage_share: Decimal, deductible: Decimal = Decimal("0")
    ) -> Decimal:
        """
        Calculate co-payment amount.
        Co-pay is applied on (requested_amount - deductible).
        """
        amount_after_deductible = max(requested_amount - deductible, Decimal("0"))
        copay = amount_after_deductible * (percentage_share / Decimal("100"))
        return copay.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @staticmethod
    def calculate_policy_payable_amount(
        requested_amount: Decimal,
        deductible: Decimal = Decimal("0"),
        percentage_share: Decimal = Decimal("0"),
        sub_limit: Optional[Decimal] = None,
    ) -> CoverageCalculation:
        """
        Calculate a policy-level net payable amount after all deductions.

        Order of deductions:
        1. Apply sub-limit (if any)
        2. Deduct deductible
        3. Apply co-pay on remaining
        """
        # Step 1: Apply sub-limit
        if sub_limit and requested_amount > sub_limit:
            amount_after_sublimit = sub_limit
            sublimit_applied = requested_amount - sub_limit
        else:
            amount_after_sublimit = requested_amount
            sublimit_applied = Decimal("0")

        # Step 2: Deduct deductible
        amount_after_deductible = max(amount_after_sublimit - deductible, Decimal("0"))

        # Step 3: Calculate co-pay
        copay_amount = Decimal("0")
        if percentage_share > 0:
            copay_amount = amount_after_deductible * (percentage_share / Decimal("100"))

        # Final net payable
        payable_amount = amount_after_deductible - copay_amount
        payable_amount = payable_amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Build breakdown
        breakdown = f"""
Requested Amount: CU {requested_amount:,.2f}
- Inner Cap Applied: CU {sublimit_applied:,.2f}
= After Inner Cap: CU {amount_after_sublimit:,.2f}
- Fixed Share: CU {deductible:,.2f}
= After Fixed Share: CU {amount_after_deductible:,.2f}
- Percentage Share ({percentage_share}%): CU {copay_amount:,.2f}
= Payable Amount: CU {payable_amount:,.2f}
        """.strip()

        return CoverageCalculation(
            gross_amount=requested_amount,
            deductible=deductible,
            copay_amount=copay_amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            sub_limit_applied=sublimit_applied,
            payable_amount=payable_amount,
            calculation_breakdown=breakdown,
        )

    @staticmethod
    def calculate_aggregate_si_usage(
        benefit_ceiling: Decimal, service_cases_paid: List[Decimal]
    ) -> dict:
        """
        Calculate aggregate Sum Insured usage.
        """
        total_used = sum(service_cases_paid)
        remaining = max(benefit_ceiling - total_used, Decimal("0"))
        usage_percentage = (
            (total_used / benefit_ceiling * 100) if benefit_ceiling > 0 else Decimal("0")
        )

        return {
            "benefit_ceiling": float(benefit_ceiling),
            "total_used": float(total_used),
            "remaining": float(remaining),
            "usage_percentage": float(usage_percentage.quantize(Decimal("0.01"))),
            "is_exhausted": remaining <= 0,
        }

    @staticmethod
    def is_within_waiting_period(effective_from, service_date, eligibility_delay_days: int) -> dict:
        """
        Check whether a service date falls within the eligibility delay.
        """
        if isinstance(effective_from, str):
            effective_from = date.fromisoformat(effective_from)
        if isinstance(service_date, str):
            service_date = date.fromisoformat(service_date)

        days_since_inception = (service_date - effective_from).days
        waiting_period_end = effective_from + timedelta(days=eligibility_delay_days)
        is_within = days_since_inception < eligibility_delay_days

        return {
            "is_within_waiting_period": is_within,
            "days_since_inception": days_since_inception,
            "eligibility_delay_days": eligibility_delay_days,
            "waiting_period_end_date": waiting_period_end.isoformat(),
            "days_remaining": max(eligibility_delay_days - days_since_inception, 0)
            if is_within
            else 0,
        }

    def calculate_room_rent(
        self,
        actual_room_rent: float,
        benefit_ceiling: float,
        plan_type: str = "standard",
        daily_limit: Optional[float] = None,
    ) -> RoomRentResult:
        """
        Calculate room rent allowed based on plan type.
        """
        plan = PLAN_CONFIGS.get(plan_type.lower(), PLAN_CONFIGS["standard"])

        # Premium plan has no limit
        if plan["room_rent_percent"] is None:
            return RoomRentResult(
                allowed_amount=actual_room_rent, excess=0, proportional_deduction=1.0
            )

        # Calculate limit based on percentage of SI
        percent_limit = benefit_ceiling * plan["room_rent_percent"]

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
            allowed_amount=allowed, excess=excess, proportional_deduction=proportional
        )

    def calculate_icu_charges(self, actual_icu_rate: float, room_rent_limit: float) -> dict:
        """
        Calculate ICU charges (typically 2x room rent limit).
        """
        icu_limit = room_rent_limit * 2
        allowed = min(actual_icu_rate, icu_limit)
        excess = max(0, actual_icu_rate - icu_limit)

        return {"allowed_amount": allowed, "excess": excess, "icu_limit": icu_limit}

    def calculate_copay(
        self,
        requested_amount: float,
        plan_type: str = "standard",
        is_participating_provider: bool = True,
    ) -> float:
        """
        Calculate a synthetic-claim copay based on plan type and hospital network.
        """
        plan = PLAN_CONFIGS.get(plan_type.lower(), PLAN_CONFIGS["standard"])

        copay = requested_amount * plan["copay_percent"]

        # Non-network hospital: additional 20% copay
        if not is_participating_provider:
            copay += requested_amount * 0.20

        # High-value service case: additional 10% percentage share
        if requested_amount > 1000000:
            copay += requested_amount * 0.10

        return copay

    def calculate_deductible(self, plan_type: str = "standard") -> float:
        """
        Get deductible amount for plan type.
        """
        plan = PLAN_CONFIGS.get(plan_type.lower(), PLAN_CONFIGS["standard"])
        return float(plan["deductible"])

    def calculate_payable_amount(
        self,
        requested_amount: float,
        plan_type: str = "standard",
        benefit_ceiling: float = 500000,
        actual_room_rent: Optional[float] = None,
        room_rent_limit: Optional[float] = None,
    ) -> ClaimCalculationResult:
        """
        Calculate a synthetic-claim net payable amount after all deductions.
        """
        plan = PLAN_CONFIGS.get(plan_type.lower(), PLAN_CONFIGS["standard"])

        # Cap at sum insured
        eligible_base = min(requested_amount, benefit_ceiling)

        # Calculate deductible
        deductible = float(plan["deductible"])
        after_deductible = max(0, eligible_base - deductible)

        # Calculate copay
        copay = after_deductible * plan["copay_percent"]

        # Room rent proportional deduction
        room_rent_deduction = 0.0
        if actual_room_rent and room_rent_limit and actual_room_rent > room_rent_limit:
            proportion = room_rent_limit / actual_room_rent
            room_rent_deduction = after_deductible * (1 - proportion)

        # Final net payable
        payable_amount = max(0, after_deductible - copay - room_rent_deduction)

        return ClaimCalculationResult(
            requested_amount=requested_amount,
            eligible_base=eligible_base,
            deductible=deductible,
            copay=copay,
            room_rent_deduction=room_rent_deduction,
            payable_amount=payable_amount,
        )
