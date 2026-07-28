"""
Unit tests for coverage calculator.
"""
import pytest
from app.domain.services.coverage_calculator import (
    CoverageCalculator,
    RoomRentResult,
    ClaimCalculationResult
)


class TestCoverageCalculator:
    """Tests for CoverageCalculator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = CoverageCalculator()
    
    def test_room_rent_no_sublimit(self):
        """Test room rent calculation with no sub-limit."""
        result = self.calculator.calculate_room_rent(
            actual_room_rent=10000,
            sum_insured=1000000,
            plan_type="premium"
        )
        
        assert result.allowed_amount == 10000
        assert result.excess == 0
        assert result.proportional_deduction == 1.0
    
    def test_room_rent_with_percentage_limit(self):
        """Test room rent with percentage-based limit."""
        # Standard plan: 1% of sum insured
        result = self.calculator.calculate_room_rent(
            actual_room_rent=15000,
            sum_insured=500000,  # 1% = 5000
            plan_type="standard"
        )
        
        assert result.allowed_amount == 5000
        assert result.excess == 10000
        assert result.proportional_deduction == pytest.approx(5000 / 15000, rel=0.01)
    
    def test_room_rent_with_fixed_limit(self):
        """Test room rent with fixed daily limit."""
        result = self.calculator.calculate_room_rent(
            actual_room_rent=8000,
            sum_insured=1000000,
            plan_type="standard",
            daily_limit=5000
        )
        
        assert result.allowed_amount == 5000
        assert result.excess == 3000
    
    def test_icu_charges_calculation(self):
        """Test ICU charges (2x room rent limit)."""
        result = self.calculator.calculate_icu_charges(
            actual_icu_rate=20000,
            room_rent_limit=5000  # ICU limit = 10000
        )
        
        assert result["allowed_amount"] == 10000
        assert result["excess"] == 10000
    
    def test_copay_standard_plan(self):
        """Test copay calculation for standard plan."""
        copay = self.calculator.calculate_copay(
            claim_amount=100000,
            plan_type="standard"  # 20% copay
        )
        
        assert copay == 20000
    
    def test_copay_enhanced_plan(self):
        """Test copay calculation for enhanced plan."""
        copay = self.calculator.calculate_copay(
            claim_amount=100000,
            plan_type="enhanced"  # 10% copay
        )
        
        assert copay == 10000
    
    def test_copay_premium_plan_no_copay(self):
        """Test premium plan has no copay."""
        copay = self.calculator.calculate_copay(
            claim_amount=100000,
            plan_type="premium"
        )
        
        assert copay == 0
    
    def test_deductible_standard_plan(self):
        """Test deductible for standard plan."""
        deductible = self.calculator.calculate_deductible(
            plan_type="standard"
        )
        
        assert deductible == 5000
    
    def test_deductible_premium_plan(self):
        """Test premium plan has no deductible."""
        deductible = self.calculator.calculate_deductible(
            plan_type="premium"
        )
        
        assert deductible == 0
    
    def test_net_payable_full_calculation(self):
        """Test full net payable calculation."""
        result = self.calculator.calculate_net_payable(
            claim_amount=100000,
            plan_type="standard",
            sum_insured=500000,
            actual_room_rent=8000,
            room_rent_limit=5000
        )
        
        # Deductible: 5000
        # Copay (20%): 19000
        # Room rent deduction proportional
        
        assert result.deductible == 5000
        assert result.copay > 0
        assert result.net_payable < 100000
        assert result.net_payable > 0
    
    def test_high_value_claim_additional_copay(self):
        """Test additional copay for claims > 10 lakhs."""
        copay = self.calculator.calculate_copay(
            claim_amount=1500000,  # 15 lakhs
            plan_type="standard"
        )
        
        # 20% base + 10% additional for high value
        expected_copay = 1500000 * 0.20 + 1500000 * 0.10
        assert copay == expected_copay
    
    def test_non_network_penalty(self):
        """Test additional copay for non-network hospital."""
        copay = self.calculator.calculate_copay(
            claim_amount=100000,
            plan_type="standard",
            is_network_hospital=False
        )
        
        # 20% base + 20% non-network penalty
        assert copay == 40000


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        self.calculator = CoverageCalculator()
    
    def test_zero_claim_amount(self):
        """Test handling of zero claim amount."""
        result = self.calculator.calculate_net_payable(
            claim_amount=0,
            plan_type="standard",
            sum_insured=500000
        )
        
        assert result.net_payable == 0
    
    def test_claim_below_deductible(self):
        """Test claim amount less than deductible."""
        result = self.calculator.calculate_net_payable(
            claim_amount=3000,  # Less than 5000 deductible
            plan_type="standard",
            sum_insured=500000
        )
        
        assert result.net_payable == 0
    
    def test_claim_exceeds_sum_insured(self):
        """Test claim amount exceeding sum insured."""
        result = self.calculator.calculate_net_payable(
            claim_amount=600000,
            plan_type="standard",
            sum_insured=500000
        )
        
        # Should cap at sum insured
        assert result.allowed_claim <= 500000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
