"""Tests for globally unique synthetic seed identifiers."""

import pytest

from data.generators.service_cases_generator import generate_participants, generate_service_cases

pytestmark = pytest.mark.unit


def test_generated_references_are_unique_across_contracts():
    first_contract_participants = generate_participants(contract_id=1, count=3)
    second_contract_participants = generate_participants(contract_id=2, count=3)
    first_contract_cases = generate_service_cases(contract_id=1, participant_ids=[1, 2], count=3)
    second_contract_cases = generate_service_cases(contract_id=2, participant_ids=[3, 4], count=3)

    participant_references = {
        participant["participant_ref"]
        for participant in first_contract_participants + second_contract_participants
    }
    case_references = {
        service_case["case_ref"] for service_case in first_contract_cases + second_contract_cases
    }

    assert len(participant_references) == 6
    assert len(case_references) == 6
