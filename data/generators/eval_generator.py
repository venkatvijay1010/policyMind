"""
Generate evaluation questions for testing PolicyMind.
"""
from typing import List


# Document QA questions (test RAG)
DOCUMENT_QA_QUESTIONS = [
    {
        "question": "What is the maternity coverage limit?",
        "expected_answer": "The maternity coverage limit is up to ₹50,000 per pregnancy. Normal delivery is covered up to ₹25,000 and Cesarean Section up to ₹50,000.",
        "difficulty": "easy",
        "key_terms": ["50,000", "maternity", "pregnancy"]
    },
    {
        "question": "What is the waiting period for pre-existing diseases?",
        "expected_answer": "The waiting period for pre-existing diseases is 48 months from the policy inception date. After this period, pre-existing conditions are covered at par with other claims.",
        "difficulty": "easy",
        "key_terms": ["48 months", "pre-existing", "waiting period"]
    },
    {
        "question": "What is the room rent limit for the Standard Plan?",
        "expected_answer": "For the Standard Plan, the room rent limit is ₹5,000 per day or 1% of Sum Insured, whichever is lower.",
        "difficulty": "easy",
        "key_terms": ["5,000", "room rent", "Standard"]
    },
    {
        "question": "What are the permanent exclusions in the policy?",
        "expected_answer": "Permanent exclusions include cosmetic treatments, obesity treatment, dental treatments (unless requiring hospitalization), fertility treatments (IVF, IUI), STDs/AIDS, self-inflicted injuries, war, hazardous sports, experimental treatments, and addiction treatments.",
        "difficulty": "medium",
        "key_terms": ["cosmetic", "exclusion", "fertility", "dental"]
    },
    {
        "question": "How do I file a cashless claim?",
        "expected_answer": "For cashless claims: 1) Get pre-authorization at a network hospital, 2) Submit Employee ID and Policy Number, 3) Hospital coordinates with TPA, 4) Approval takes 2-4 hours for planned procedures or 24 hours for emergencies.",
        "difficulty": "medium",
        "key_terms": ["cashless", "pre-authorization", "network hospital"]
    },
    {
        "question": "What is the copay for the Enhanced Plan and how is the net payable calculated?",
        "expected_answer": "The Enhanced Plan has a 10% copay with a ₹2,500 deductible. For example, on a ₹1,00,000 claim: after ₹2,500 deductible = ₹97,500, then 10% copay = ₹9,750, so net payable by insurer = ₹87,750.",
        "difficulty": "hard",
        "key_terms": ["copay", "10%", "deductible", "Enhanced"]
    },
    {
        "question": "What chronic conditions are covered after the waiting period?",
        "expected_answer": "After the 48-month waiting period, chronic conditions covered include: Diabetes, Hypertension, Heart Disease, Thyroid Disorders, and Asthma.",
        "difficulty": "medium",
        "key_terms": ["diabetes", "hypertension", "chronic", "waiting period"]
    },
    {
        "question": "What is the critical illness benefit and how is it different from the base cover?",
        "expected_answer": "The Critical Illness cover provides a lump sum benefit (100% of CI Sum Insured) upon diagnosis of specified conditions like cancer, heart attack, stroke, or kidney failure. A 30-day survival period is required. This is a one-time benefit that does not reduce the base health cover.",
        "difficulty": "hard",
        "key_terms": ["critical illness", "lump sum", "100%", "survival"]
    }
]

# Claims SQL questions (test text-to-SQL)
CLAIMS_SQL_QUESTIONS = [
    {
        "question": "How many claims were registered in 2024?",
        "expected_answer": "The total number of claims registered in 2024.",
        "difficulty": "easy",
        "expected_sql_contains": ["COUNT", "claims", "2024"]
    },
    {
        "question": "What is the total claim amount by claim status?",
        "expected_answer": "Breakdown of total claim amounts grouped by status (REGISTERED, UNDER_PROCESS, APPROVED, SETTLED, REJECTED).",
        "difficulty": "easy",
        "expected_sql_contains": ["SUM", "claim_amount", "GROUP BY", "claim_status"]
    },
    {
        "question": "What is the claim rejection rate?",
        "expected_answer": "The percentage of claims that were rejected compared to total claims.",
        "difficulty": "medium",
        "expected_sql_contains": ["COUNT", "REJECTED", "claims"]
    },
    {
        "question": "Which hospital has the highest number of claims?",
        "expected_answer": "The hospital with the most claims filed.",
        "difficulty": "medium",
        "expected_sql_contains": ["hospital_name", "COUNT", "ORDER BY", "DESC"]
    },
    {
        "question": "What is the average claim amount for maternity cases?",
        "expected_answer": "The average claim amount for claims with maternity-related diagnosis codes.",
        "difficulty": "medium",
        "expected_sql_contains": ["AVG", "claim_amount", "maternity"]
    },
    {
        "question": "Show top 5 cities by total approved claim amount",
        "expected_answer": "The top 5 cities ranked by total approved claim amounts.",
        "difficulty": "medium",
        "expected_sql_contains": ["SUM", "approved_amount", "hospital_city", "LIMIT 5"]
    },
    {
        "question": "What is the average time from admission to settlement for settled claims?",
        "expected_answer": "The average number of days between admission_date and settlement_date for claims with status SETTLED.",
        "difficulty": "hard",
        "expected_sql_contains": ["AVG", "admission_date", "settlement_date", "SETTLED"]
    },
    {
        "question": "What percentage of claims were cashless vs reimbursement, and what is the average processing time for each?",
        "expected_answer": "Comparison of cashless vs reimbursement claims including count percentage and average processing duration.",
        "difficulty": "hard",
        "expected_sql_contains": ["claim_type", "CASHLESS", "REIMBURSEMENT", "COUNT"]
    }
]

# Hybrid questions (need both document context and data)
HYBRID_QUESTIONS = [
    {
        "question": "What is our rejection rate for pre-existing conditions and what does the policy say about pre-existing disease coverage?",
        "expected_answer": "Combines rejection statistics for PED-related claims with policy terms about the 48-month waiting period for pre-existing conditions.",
        "difficulty": "hard",
        "key_terms": ["rejection", "pre-existing", "48 months", "waiting period"]
    },
    {
        "question": "How many maternity claims were filed and what are the maternity benefits as per the policy?",
        "expected_answer": "Combines maternity claim count/statistics with policy coverage details (₹50,000 limit, 36-month waiting period, normal vs cesarean limits).",
        "difficulty": "medium",
        "key_terms": ["maternity", "claims", "50,000", "waiting period"]
    },
    {
        "question": "What is the average copay applied on claims and how does it compare to the policy's copay terms?",
        "expected_answer": "Combines actual copay amounts from claims data with policy copay rules (20% Standard, 10% Enhanced, 0% Premium).",
        "difficulty": "hard",
        "key_terms": ["copay", "average", "Standard", "Enhanced", "Premium"]
    }
]


def generate_eval_questions() -> List[dict]:
    """
    Generate all evaluation questions in database-ready format.
    """
    questions = []
    question_id = 1
    
    # Document QA questions
    for q in DOCUMENT_QA_QUESTIONS:
        questions.append({
            "id": question_id,
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "query_type": "document_qa",
            "difficulty": q["difficulty"],
            "key_terms": q.get("key_terms", [])
        })
        question_id += 1
    
    # Claims SQL questions
    for q in CLAIMS_SQL_QUESTIONS:
        questions.append({
            "id": question_id,
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "query_type": "claims_sql",
            "difficulty": q["difficulty"],
            "expected_sql_contains": q.get("expected_sql_contains", [])
        })
        question_id += 1
    
    # Hybrid questions
    for q in HYBRID_QUESTIONS:
        questions.append({
            "id": question_id,
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "query_type": "hybrid",
            "difficulty": q["difficulty"],
            "key_terms": q.get("key_terms", [])
        })
        question_id += 1
    
    return questions


def generate_eval_sql_inserts() -> str:
    """
    Generate SQL INSERT statements for evaluation questions.
    """
    questions = generate_eval_questions()
    
    inserts = []
    for q in questions:
        expected = q["expected_answer"].replace("'", "''")
        question = q["question"].replace("'", "''")
        
        inserts.append(f"""
INSERT INTO eval_questions (id, question, expected_answer, query_type, difficulty)
VALUES ({q['id']}, '{question}', '{expected}', '{q['query_type']}', '{q['difficulty']}');
""")
    
    return "\n".join(inserts)


if __name__ == "__main__":
    questions = generate_eval_questions()
    print(f"Generated {len(questions)} evaluation questions:")
    print(f"  Document QA: {len(DOCUMENT_QA_QUESTIONS)}")
    print(f"  Claims SQL: {len(CLAIMS_SQL_QUESTIONS)}")
    print(f"  Hybrid: {len(HYBRID_QUESTIONS)}")
    
    print("\n--- Sample Questions ---")
    for q in questions[:3]:
        print(f"\n[{q['query_type']}] {q['question']}")
        print(f"Expected: {q['expected_answer'][:100]}...")
