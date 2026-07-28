"""
Database seeding script - populate database with sample data.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import text
from app.infrastructure.database.postgres import async_session_factory, init_db
from app.infrastructure.llm.embeddings import EmbeddingService
from data.generators.policy_generator import generate_sample_policies, generate_policy_document
from data.generators.claims_generator import generate_members, generate_claims
from data.generators.eval_generator import generate_eval_questions


async def chunk_text(content: str, chunk_size: int = 1000, overlap: int = 200):
    """Simple text chunking."""
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]
        if chunk.strip():
            chunks.append({
                "content": chunk.strip(),
                "chunk_index": len(chunks),
                "char_start": start,
                "char_end": end
            })
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
                        INSERT INTO icd_codes (code, description, category, is_chronic, typical_hospitalization_days, typical_treatment_cost)
                        VALUES (:code, :desc, :cat, :chronic, :days, :cost)
                        ON CONFLICT (code) DO NOTHING
                    """),
                    {"code": code, "desc": desc, "cat": cat, "chronic": chronic, "days": days, "cost": cost}
                )
            await session.commit()
            print(f"   Seeded {len(icd_codes)} ICD codes")
            
            # 2. Seed hospitals
            print("\n2. Seeding hospitals...")
            hospitals = [
                ("Apollo Hospital", "Chennai", "Tamil Nadu", True),
                ("Fortis Hospital", "Mumbai", "Maharashtra", True),
                ("Max Hospital", "Delhi", "Delhi", True),
                ("Manipal Hospital", "Bangalore", "Karnataka", True),
                ("City General Hospital", "Hyderabad", "Telangana", False),
            ]
            
            for name, city, state, network in hospitals:
                await session.execute(
                    text("""
                        INSERT INTO hospitals (hospital_name, city, state, is_network_hospital)
                        VALUES (:name, :city, :state, :network)
                        ON CONFLICT DO NOTHING
                    """),
                    {"name": name, "city": city, "state": state, "network": network}
                )
            await session.commit()
            print(f"   Seeded {len(hospitals)} hospitals")
            
            # 3. Seed policies
            print("\n3. Seeding policies...")
            policies = generate_sample_policies(3)
            policy_ids = []
            
            for policy in policies:
                result = await session.execute(
                    text("""
                        INSERT INTO policies (policy_number, policy_name, product_type, insured_name, 
                                            policy_start_date, policy_end_date, total_lives, 
                                            total_sum_insured, premium_amount)
                        VALUES (:policy_number, :policy_name, :product_type, :insured_name,
                                :start_date, :end_date, :lives, :sum_insured, :premium)
                        RETURNING id
                    """),
                    {
                        "policy_number": policy["policy_number"],
                        "policy_name": policy["policy_name"],
                        "product_type": policy["product_type"],
                        "insured_name": policy["insured_name"],
                        "start_date": policy["policy_start_date"],
                        "end_date": policy["policy_end_date"],
                        "lives": policy["total_lives"],
                        "sum_insured": policy["total_sum_insured"],
                        "premium": policy["premium_amount"]
                    }
                )
                policy_id = result.scalar()
                policy_ids.append(policy_id)
            
            await session.commit()
            print(f"   Seeded {len(policies)} policies")
            
            # 4. Seed policy chunks with embeddings
            print("\n4. Seeding policy chunks with embeddings...")
            embedding_service = EmbeddingService()
            total_chunks = 0
            
            for i, policy in enumerate(policies):
                policy_id = policy_ids[i]
                chunks = await chunk_text(policy["document_text"])
                
                # Get embeddings
                chunk_texts = [c["content"] for c in chunks]
                try:
                    embeddings = await embedding_service.embed_batch(chunk_texts)
                    
                    for chunk, embedding in zip(chunks, embeddings):
                        await session.execute(
                            text("""
                                INSERT INTO policy_chunks (policy_id, content, chunk_index, char_start, char_end, embedding)
                                VALUES (:policy_id, :content, :idx, :start, :end, :embedding)
                            """),
                            {
                                "policy_id": policy_id,
                                "content": chunk["content"],
                                "idx": chunk["chunk_index"],
                                "start": chunk["char_start"],
                                "end": chunk["char_end"],
                                "embedding": str(embedding)
                            }
                        )
                        total_chunks += 1
                except Exception as e:
                    print(f"   Warning: Could not generate embeddings: {e}")
                    print("   Skipping embedding generation (requires OPENAI_API_KEY)")
            
            await session.commit()
            print(f"   Seeded {total_chunks} policy chunks")
            
            # 5. Seed members
            print("\n5. Seeding members...")
            all_member_ids = []
            
            for policy_id in policy_ids:
                members = generate_members(policy_id, 15)
                
                for member in members:
                    result = await session.execute(
                        text("""
                            INSERT INTO members (member_id, policy_id, member_name, relationship, gender,
                                               date_of_birth, age, sum_insured, status, city, state)
                            VALUES (:member_id, :policy_id, :name, :rel, :gender, :dob, :age, :sum, :status, :city, :state)
                            RETURNING id
                        """),
                        {
                            "member_id": member["member_id"],
                            "policy_id": policy_id,
                            "name": member["member_name"],
                            "rel": member["relationship"],
                            "gender": member["gender"],
                            "dob": member["date_of_birth"],
                            "age": member["age"],
                            "sum": member["sum_insured"],
                            "status": member["status"],
                            "city": member["city"],
                            "state": member["state"]
                        }
                    )
                    all_member_ids.append(result.scalar())
            
            await session.commit()
            print(f"   Seeded {len(all_member_ids)} members")
            
            # 6. Seed claims
            print("\n6. Seeding claims...")
            total_claims = 0
            
            for policy_id in policy_ids:
                # Get member IDs for this policy
                result = await session.execute(
                    text("SELECT id FROM members WHERE policy_id = :pid"),
                    {"pid": policy_id}
                )
                member_ids = [r[0] for r in result.fetchall()]
                
                claims = generate_claims(policy_id, member_ids, 30)
                
                for claim in claims:
                    await session.execute(
                        text("""
                            INSERT INTO claims (claim_number, policy_id, member_id, claim_type, claim_category,
                                              diagnosis_code, diagnosis_description, treatment_type,
                                              hospital_name, hospital_city, hospital_state,
                                              admission_date, discharge_date, claim_amount, approved_amount,
                                              deductible_applied, copay_applied, net_payable,
                                              claim_status, registration_date, settlement_date, rejection_reason)
                            VALUES (:claim_number, :policy_id, :member_id, :claim_type, :claim_category,
                                    :diagnosis_code, :diagnosis_description, :treatment_type,
                                    :hospital_name, :hospital_city, :hospital_state,
                                    :admission_date, :discharge_date, :claim_amount, :approved_amount,
                                    :deductible_applied, :copay_applied, :net_payable,
                                    :claim_status, :registration_date, :settlement_date, :rejection_reason)
                        """),
                        claim
                    )
                    total_claims += 1
            
            await session.commit()
            print(f"   Seeded {total_claims} claims")
            
            # 7. Seed evaluation questions
            print("\n7. Seeding evaluation questions...")
            questions = generate_eval_questions()
            
            for q in questions:
                await session.execute(
                    text("""
                        INSERT INTO eval_questions (question, expected_answer, query_type, difficulty)
                        VALUES (:question, :expected, :type, :difficulty)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "question": q["question"],
                        "expected": q["expected_answer"],
                        "type": q["query_type"],
                        "difficulty": q["difficulty"]
                    }
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
