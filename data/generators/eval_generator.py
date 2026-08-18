"""
Generate evaluation questions for testing PolicyMind.
"""

from typing import List

# Document QA questions (test RAG)
DOCUMENT_QA_QUESTIONS = [
    {
        "question": "What is the maternity coverage limit?",
        "expected_answer": "The family-support benefit is capped at CU 50,000 per event, with CU 25,000 for a standard service and CU 50,000 for a surgical service.",
        "difficulty": "easy",
        "key_terms": ["50,000", "maternity", "pregnancy"],
    },
    {
        "question": "What is the waiting period for pre-existing diseases?",
        "expected_answer": "The waiting period for pre-existing diseases is 48 months from the policy inception date. After this period, pre-existing conditions are covered at par with other service_cases.",
        "difficulty": "easy",
        "key_terms": ["48 months", "pre-existing", "waiting period"],
    },
    {
        "question": "What is the room rent limit for the Standard Plan?",
        "expected_answer": "For the Standard Plan, the facility limit is CU 5,000 per day or 1% of the benefit cap, whichever is lower.",
        "difficulty": "easy",
        "key_terms": ["5,000", "room rent", "Standard"],
    },
    {
        "question": "What are the permanent exclusions in the policy?",
        "expected_answer": "Permanent exclusions include cosmetic treatments, obesity treatment, dental treatments (unless requiring hospitalization), fertility treatments (IVF, IUI), STDs/AIDS, self-inflicted injuries, war, hazardous sports, experimental treatments, and addiction treatments.",
        "difficulty": "medium",
        "key_terms": ["cosmetic", "exclusion", "fertility", "dental"],
    },
    {
        "question": "How do I open a direct-billing service case?",
        "expected_answer": "Ask a participating provider to open the case, share the participant access code and contract reference, and let the provider send a benefit-check request to the service partner.",
        "difficulty": "medium",
        "key_terms": ["direct-billing", "benefit-check", "participating provider"],
    },
    {
        "question": "What is the copay for the Enhanced Plan and how is the net payable calculated?",
        "expected_answer": "The Enhanced Plan has a 10% percentage share and CU 2,500 fixed share. On CU 100,000, the resulting payable amount is CU 87,750.",
        "difficulty": "hard",
        "key_terms": ["copay", "10%", "deductible", "Enhanced"],
    },
    {
        "question": "What chronic conditions are covered after the waiting period?",
        "expected_answer": "After the 48-month waiting period, chronic conditions covered include: Diabetes, Hypertension, Heart Disease, Thyroid Disorders, and Asthma.",
        "difficulty": "medium",
        "key_terms": ["diabetes", "hypertension", "chronic", "waiting period"],
    },
    {
        "question": "What is the critical illness benefit and how is it different from the base cover?",
        "expected_answer": "The Critical Illness cover provides a lump sum benefit (100% of CI Sum Insured) upon diagnosis of specified conditions like cancer, heart attack, stroke, or kidney failure. A 30-day survival period is required. This is a one-time benefit that does not reduce the base health cover.",
        "difficulty": "hard",
        "key_terms": ["critical illness", "lump sum", "100%", "survival"],
    },
]

# Claims SQL questions (test text-to-SQL)
RECORDS_SQL_QUESTIONS = [
    {
        "question": "How many service cases were opened in 2024?",
        "expected_answer": "The total number of service cases opened in 2024.",
        "difficulty": "easy",
        "expected_sql_contains": ["COUNT", "service_cases", "2024"],
    },
    {
        "question": "What is the total requested amount by case status?",
        "expected_answer": "Breakdown of requested amounts grouped by status (OPENED, IN_REVIEW, ELIGIBLE, RESOLVED, DECLINED).",
        "difficulty": "easy",
        "expected_sql_contains": ["SUM", "requested_amount", "GROUP BY", "case_status"],
    },
    {
        "question": "What is the case decline rate?",
        "expected_answer": "The percentage of service cases that were declined compared with all service cases.",
        "difficulty": "medium",
        "expected_sql_contains": ["COUNT", "DECLINED", "service_cases"],
    },
    {
        "question": "Which hospital has the highest number of service_cases?",
        "expected_answer": "The hospital with the most service_cases filed.",
        "difficulty": "medium",
        "expected_sql_contains": ["provider_label", "COUNT", "ORDER BY", "DESC"],
    },
    {
        "question": "What is the average requested amount for family-support cases?",
        "expected_answer": "The average requested amount for family-support service cases.",
        "difficulty": "medium",
        "expected_sql_contains": ["AVG", "requested_amount", "maternity"],
    },
    {
        "question": "Show the top five provider cities by total eligible amount",
        "expected_answer": "The top five provider cities ranked by total eligible amount.",
        "difficulty": "medium",
        "expected_sql_contains": ["SUM", "eligible_amount", "provider_city", "LIMIT 5"],
    },
    {
        "question": "What is the average time from service start to resolution for resolved cases?",
        "expected_answer": "The average number of days between service_started_on and resolved_on for cases with status RESOLVED.",
        "difficulty": "hard",
        "expected_sql_contains": ["AVG", "service_started_on", "resolved_on", "RESOLVED"],
    },
    {
        "question": "What percentage of cases used direct billing versus member-paid funding, and what is the average review time for each?",
        "expected_answer": "Comparison of direct-billing and member-paid cases including count percentage and average review duration.",
        "difficulty": "hard",
        "expected_sql_contains": ["funding_mode", "DIRECT_BILLING", "MEMBER_PAID", "COUNT"],
    },
]

# Hybrid questions (need both document context and data)
HYBRID_QUESTIONS = [
    {
        "question": "What is our rejection rate for pre-existing conditions and what does the policy say about pre-existing disease coverage?",
        "expected_answer": "Combines rejection statistics for PED-related service_cases with policy terms about the 48-month waiting period for pre-existing conditions.",
        "difficulty": "hard",
        "key_terms": ["rejection", "pre-existing", "48 months", "waiting period"],
    },
    {
        "question": "How many maternity service_cases were filed and what are the maternity benefits as per the policy?",
        "expected_answer": "Combines family-support case statistics with contract details (CU 50,000 cap and the configured eligibility delay).",
        "difficulty": "medium",
        "key_terms": ["maternity", "service_cases", "50,000", "waiting period"],
    },
    {
        "question": "What is the average copay applied on service_cases and how does it compare to the policy's copay terms?",
        "expected_answer": "Combines actual copay amounts from service_cases data with policy copay rules (20% Standard, 10% Enhanced, 0% Premium).",
        "difficulty": "hard",
        "key_terms": ["copay", "average", "Standard", "Enhanced", "Premium"],
    },
]


def generate_eval_questions() -> List[dict]:
    """
    Generate all evaluation questions in database-ready format.
    """
    questions = []
    question_id = 1

    # Document QA questions
    for q in DOCUMENT_QA_QUESTIONS:
        questions.append(
            {
                "id": question_id,
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "query_type": "document_qa",
                "difficulty": q["difficulty"],
                "key_terms": q.get("key_terms", []),
            }
        )
        question_id += 1

    # Claims SQL questions
    for q in RECORDS_SQL_QUESTIONS:
        questions.append(
            {
                "id": question_id,
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "query_type": "records_sql",
                "difficulty": q["difficulty"],
                "expected_sql_contains": q.get("expected_sql_contains", []),
            }
        )
        question_id += 1

    # Hybrid questions
    for q in HYBRID_QUESTIONS:
        questions.append(
            {
                "id": question_id,
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "query_type": "hybrid",
                "difficulty": q["difficulty"],
                "key_terms": q.get("key_terms", []),
            }
        )
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
INSERT INTO eval_questions (id, question, ground_truth_answer, query_type, difficulty)
VALUES ({q["id"]}, '{question}', '{expected}', '{q["query_type"]}', '{q["difficulty"]}');
""")

    return "\n".join(inserts)


if __name__ == "__main__":
    questions = generate_eval_questions()
    print(f"Generated {len(questions)} evaluation questions:")
    print(f"  Document QA: {len(DOCUMENT_QA_QUESTIONS)}")
    print(f"  Claims SQL: {len(RECORDS_SQL_QUESTIONS)}")
    print(f"  Hybrid: {len(HYBRID_QUESTIONS)}")

    print("\n--- Sample Questions ---")
    for q in questions[:3]:
        print(f"\n[{q['query_type']}] {q['question']}")
        print(f"Expected: {q['expected_answer'][:100]}...")
