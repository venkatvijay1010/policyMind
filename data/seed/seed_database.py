"""
Database seeding script - populate database with sample data.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import text

from app.infrastructure.database.postgres import async_session_factory, init_db
from app.infrastructure.llm.embeddings import EmbeddingService
from data.generators.eval_generator import generate_eval_questions
from data.generators.policy_generator import (
    generate_sample_benefit_contracts,
)
from data.generators.service_cases_generator import generate_participants, generate_service_cases


def chunk_text(
    content: str, chunk_size: int = 1000, overlap: int = 200
) -> list[dict[str, int | str]]:
    """Simple text chunking."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunk = content[start:end]
        if chunk.strip():
            chunks.append(
                {
                    "content": chunk.strip(),
                    "passage_order": len(chunks),
                    "source_offset_start": start,
                    "source_offset_end": end,
                }
            )
        if end == len(content):
            break
        start = end - overlap
    return chunks


async def seed_database():
    """Seed the database with sample data."""
    print("Starting database seeding...")

    # Initialize database
    await init_db()

    async with async_session_factory() as session:
        try:
            # 1. Seed ICD codes
            print("\n1. Seeding ICD codes...")
            icd_codes = [
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
            ]

            for code, desc, cat, chronic, days, cost in icd_codes:
                await session.execute(
                    text("""
                        INSERT INTO icd_codes (code, description, category, is_chronic, is_pre_existing,
                                               typical_hospitalization_days, typical_treatment_cost)
                        VALUES (:code, :desc, :cat, :chronic, :pre_existing, :days, :cost)
                        ON CONFLICT (code) DO NOTHING
                    """),
                    {
                        "code": code,
                        "desc": desc,
                        "cat": cat,
                        "chronic": chronic,
                        "pre_existing": chronic,
                        "days": days,
                        "cost": cost,
                    },
                )
            await session.commit()
            print(f"   Seeded {len(icd_codes)} ICD codes")

            # 2. Seed care_providers
            print("\n2. Seeding care_providers...")
            care_providers = [
                ("Northstar Medical Center", "Lakeview", "North District", True),
                ("Willow Community Clinic", "Fairhaven", "West District", True),
                ("Summit Regional Center", "Brookfield", "Central District", True),
                ("Harborview Clinic", "Riverton", "Coastal District", True),
                ("Meadow General Clinic", "Clearwater", "South District", False),
            ]

            for name, city, state, network in care_providers:
                await session.execute(
                    text("""
                        INSERT INTO care_providers (
                            provider_label,
                            provider_kind,
                            city,
                            state,
                            is_active,
                            is_participating
                        )
                        VALUES (:name, :kind, :city, :state, :active, :network)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "name": name,
                        "kind": "PARTICIPATING" if network else "NON_PARTICIPATING",
                        "city": city,
                        "state": state,
                        "active": True,
                        "network": network,
                    },
                )
            await session.commit()
            print(f"   Seeded {len(care_providers)} care_providers")

            # 3. Seed benefit_contracts
            print("\n3. Seeding benefit_contracts...")
            benefit_contracts = generate_sample_benefit_contracts(3)
            contract_ids = []

            for contract in benefit_contracts:
                result = await session.execute(
                    text("""
                        INSERT INTO benefit_contracts (contract_ref, contract_title, plan_category, sponsor_label,
                                            effective_from, effective_until, participant_count,
                                            aggregate_benefit_cap, contribution_amount)
                        VALUES (:contract_ref, :contract_title, :plan_category, :sponsor_label,
                                :effective_from, :effective_until, :participant_count,
                                :aggregate_benefit_cap, :contribution_amount)
                        RETURNING id
                    """),
                    {
                        "contract_ref": contract["contract_ref"],
                        "contract_title": contract["contract_title"],
                        "plan_category": contract["plan_category"],
                        "sponsor_label": contract["sponsor_label"],
                        "effective_from": contract["effective_from"],
                        "effective_until": contract["effective_until"],
                        "participant_count": contract["participant_count"],
                        "aggregate_benefit_cap": contract["aggregate_benefit_cap"],
                        "contribution_amount": contract["contribution_amount"],
                    },
                )
                contract_id = result.scalar()
                contract_ids.append(contract_id)

            await session.commit()
            print(f"   Seeded {len(benefit_contracts)} benefit_contracts")

            # 4. Seed contract passages with embeddings
            print("\n4. Seeding contract passages with embeddings...")
            embedding_service = EmbeddingService()
            total_chunks = 0

            for i, contract in enumerate(benefit_contracts):
                contract_id = contract_ids[i]
                chunks = chunk_text(contract["source_text"])

                # Get embeddings
                chunk_texts = [c["content"] for c in chunks]
                try:
                    embeddings = await embedding_service.embed_batch(chunk_texts)

                    for chunk, embedding in zip(chunks, embeddings):
                        await session.execute(
                            text("""
                                INSERT INTO contract_passages (contract_id, content, passage_order, source_offset_start, source_offset_end, embedding)
                                VALUES (:contract_id, :content, :idx, :start, :end, :embedding)
                            """),
                            {
                                "contract_id": contract_id,
                                "content": chunk["content"],
                                "idx": chunk["passage_order"],
                                "start": chunk["source_offset_start"],
                                "end": chunk["source_offset_end"],
                                "embedding": json.dumps(embedding),
                            },
                        )
                        total_chunks += 1
                except Exception as e:
                    print(f"   Warning: Could not generate embeddings: {e}")
                    print(
                        "   Skipping embedding generation (start Ollama and pull the embedding model)"
                    )

            await session.commit()
            print(f"   Seeded {total_chunks} contract passages")

            # 5. Seed participants
            print("\n5. Seeding participants...")
            all_participant_ids = []

            for contract_id in contract_ids:
                participants = generate_participants(contract_id, 15)

                for participant in participants:
                    result = await session.execute(
                        text("""
                            INSERT INTO participants (participant_ref, contract_id, participant_label, enrolment_role, gender,
                                               birth_date, age, benefit_ceiling, status, city, state)
                            VALUES (:participant_ref, :contract_id, :name, :rel, :gender, :dob, :age, :benefit_cap, :status, :city, :state)
                            RETURNING id
                        """),
                        {
                            "participant_ref": participant["participant_ref"],
                            "contract_id": contract_id,
                            "name": participant["participant_label"],
                            "rel": participant["enrolment_role"],
                            "gender": participant["gender"],
                            "dob": participant["birth_date"],
                            "age": participant["age"],
                            "benefit_cap": participant["benefit_ceiling"],
                            "status": participant["status"],
                            "city": participant["city"],
                            "state": participant["state"],
                        },
                    )
                    all_participant_ids.append(result.scalar())

            await session.commit()
            print(f"   Seeded {len(all_participant_ids)} participants")

            # 6. Seed service_cases
            print("\n6. Seeding service_cases...")
            total_service_cases = 0

            for contract_id in contract_ids:
                # Get participant row IDs for this contract
                result = await session.execute(
                    text("SELECT id FROM participants WHERE contract_id = :pid"),
                    {"pid": contract_id},
                )
                participant_ids = [r[0] for r in result.fetchall()]

                service_cases = generate_service_cases(contract_id, participant_ids, 30)

                for service_case in service_cases:
                    await session.execute(
                        text("""
                            INSERT INTO service_cases (case_ref, contract_id, participant_id, funding_mode, care_setting,
                                              condition_code, condition_label, service_category,
                                              provider_label, provider_city, provider_region,
                                              service_started_on, service_ended_on, requested_amount, eligible_amount,
                                              fixed_share_applied, percentage_share_applied, payable_amount,
                                              case_status, submitted_on, resolved_on, decision_reason)
                            VALUES (:case_ref, :contract_id, :participant_id, :funding_mode, :care_setting,
                                    :condition_code, :condition_label, :service_category,
                                    :provider_label, :provider_city, :provider_region,
                                    :service_started_on, :service_ended_on, :requested_amount, :eligible_amount,
                                    :fixed_share_applied, :percentage_share_applied, :payable_amount,
                                    :case_status, :submitted_on, :resolved_on, :decision_reason)
                        """),
                        service_case,
                    )
                    total_service_cases += 1

            await session.commit()
            print(f"   Seeded {total_service_cases} service_cases")

            # 7. Seed evaluation questions
            print("\n7. Seeding evaluation questions...")
            questions = generate_eval_questions()

            for q in questions:
                await session.execute(
                    text("""
                        INSERT INTO eval_questions (question, ground_truth_answer, query_type, difficulty)
                        VALUES (:question, :expected, :type, :difficulty)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "question": q["question"],
                        "expected": q["expected_answer"],
                        "type": q["query_type"],
                        "difficulty": q["difficulty"],
                    },
                )

            await session.commit()
            print(f"   Seeded {len(questions)} evaluation questions")

            print("\n" + "=" * 50)
            print("Database seeding completed successfully!")
            print("=" * 50)

        except Exception as e:
            await session.rollback()
            print(f"\nError seeding database: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(seed_database())
