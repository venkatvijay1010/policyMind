"""
Evaluation endpoint - test and benchmark the system.
"""

import uuid
from datetime import datetime
from typing import Any, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.schemas import (
    EvalQuestion,
    EvalResult,
    EvalRunRequest,
    EvalRunResponse,
    EvalSummary,
    QueryTypeEnum,
)
from app.application.graph.orchestrator import PolicyMindGraph
from app.config import settings
from app.infrastructure.database.postgres import get_db_session
from app.infrastructure.rate_limit import limiter

logger = structlog.get_logger()

router = APIRouter(prefix="/eval", tags=["Evaluation"])


def compute_similarity(expected: str, actual: str) -> float:
    """
    Compute semantic similarity between expected and actual answers.
    Simple implementation using word overlap.
    For production, use embeddings-based similarity.
    """
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())

    if not expected_words:
        return 1.0 if not actual_words else 0.0

    intersection = expected_words & actual_words
    union = expected_words | actual_words

    # Jaccard similarity
    jaccard = len(intersection) / len(union) if union else 0.0

    # Recall (how many expected words appear in actual)
    recall = len(intersection) / len(expected_words)

    # Weighted average
    return 0.4 * jaccard + 0.6 * recall


@router.get("/questions", response_model=List[EvalQuestion])
async def list_questions(
    query_type: Optional[QueryTypeEnum] = None,
    difficulty: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db_session),
):
    """
    List available evaluation questions.
    """
    query = "SELECT * FROM eval_questions WHERE 1=1"
    params: dict[str, Any] = {}

    if query_type:
        query += " AND query_type = :query_type"
        params["query_type"] = query_type.value

    if difficulty:
        query += " AND difficulty = :difficulty"
        params["difficulty"] = difficulty

    query += " ORDER BY id LIMIT :limit"
    params["limit"] = limit

    result = await db.execute(text(query), params)
    questions = result.fetchall()

    return [
        EvalQuestion(
            question_id=q.id,
            question=q.question,
            expected_answer=q.ground_truth_answer,
            query_type=QueryTypeEnum(q.query_type),
            difficulty=q.difficulty,
        )
        for q in questions
    ]


@router.post("/run", response_model=EvalRunResponse)
@limiter.limit(settings.rate_limit)
async def run_evaluation(
    request: Request,
    eval_request: EvalRunRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Run evaluation on a set of questions.

    Results are stored in the database for historical tracking.
    """
    run_id = str(uuid.uuid4())
    started_at = datetime.utcnow()

    # Build query for questions
    query = "SELECT * FROM eval_questions WHERE 1=1"
    params: dict[str, Any] = {}

    expanding_parameters = []
    if eval_request.question_ids:
        query += " AND id IN :ids"
        params["ids"] = eval_request.question_ids
        expanding_parameters.append(bindparam("ids", expanding=True))

    if eval_request.query_types:
        query += " AND query_type IN :types"
        params["types"] = [t.value for t in eval_request.query_types]
        expanding_parameters.append(bindparam("types", expanding=True))

    if eval_request.sample_size:
        query += " ORDER BY RANDOM() LIMIT :limit"
        params["limit"] = eval_request.sample_size
    else:
        query += " ORDER BY id"

    statement = text(query)
    if expanding_parameters:
        statement = statement.bindparams(*expanding_parameters)
    result = await db.execute(statement, params)
    questions = result.fetchall()

    if not questions:
        raise HTTPException(status_code=404, detail="No questions found matching criteria")

    logger.info("Starting evaluation run", run_id=run_id, num_questions=len(questions))

    # Run evaluations
    results: List[EvalResult] = []
    graph = PolicyMindGraph(db)

    # Counters for summary
    total = len(questions)
    correct = 0
    total_latency = 0
    total_similarity = 0.0

    by_query_type: dict[str, dict[str, int]] = {}
    by_difficulty: dict[str, dict[str, int]] = {}

    for q in questions:
        try:
            # Execute query
            query_result = await graph.invoke(query=q.question)

            # Compute similarity
            similarity = compute_similarity(q.ground_truth_answer, query_result.answer)
            is_correct = similarity >= 0.7  # 70% threshold

            if is_correct:
                correct += 1

            total_latency += query_result.latency_ms or 0
            total_similarity += similarity

            # Track by type
            qt = str(q.query_type)
            if qt not in by_query_type:
                by_query_type[qt] = {"total": 0, "correct": 0}
            by_query_type[qt]["total"] += 1
            if is_correct:
                by_query_type[qt]["correct"] += 1

            # Track by difficulty
            diff = str(q.difficulty)
            if diff not in by_difficulty:
                by_difficulty[diff] = {"total": 0, "correct": 0}
            by_difficulty[diff]["total"] += 1
            if is_correct:
                by_difficulty[diff]["correct"] += 1

            eval_result = EvalResult(
                question_id=q.id,
                question=q.question,
                expected_answer=q.ground_truth_answer,
                actual_answer=query_result.answer,
                query_type=QueryTypeEnum(qt),
                is_correct=is_correct,
                similarity_score=round(similarity, 4),
                latency_ms=query_result.latency_ms or 0,
                citations_count=len(query_result.citations),
            )

            # Store in database
            await db.execute(
                text("""
                    INSERT INTO eval_results
                    (run_id, question_id, generated_answer, actual_answer, is_correct, similarity_score, latency_ms)
                    VALUES (:run_id, :question_id, :generated_answer, :actual_answer, :is_correct, :similarity, :latency)
                """),
                {
                    "run_id": run_id,
                    "question_id": q.id,
                    "generated_answer": query_result.answer,
                    "actual_answer": query_result.answer,
                    "is_correct": is_correct,
                    "similarity": similarity,
                    "latency": query_result.latency_ms,
                },
            )

            results.append(eval_result)

        except Exception as e:
            logger.error("Evaluation failed for question", question_id=q.id, error=str(e))
            results.append(
                EvalResult(
                    question_id=q.id,
                    question=q.question,
                    expected_answer=q.ground_truth_answer,
                    actual_answer="",
                    query_type=QueryTypeEnum(q.query_type),
                    is_correct=False,
                    similarity_score=0.0,
                    latency_ms=0,
                    citations_count=0,
                    error=str(e),
                )
            )

    await db.commit()

    completed_at = datetime.utcnow()

    # Compute summary
    summary = EvalSummary(
        total_questions=total,
        correct_answers=correct,
        accuracy=round(correct / total, 4) if total > 0 else 0.0,
        avg_latency_ms=round(total_latency / total, 2) if total > 0 else 0.0,
        avg_similarity=round(total_similarity / total, 4) if total > 0 else 0.0,
        by_query_type={
            k: {"accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0, **v}
            for k, v in by_query_type.items()
        },
        by_difficulty={
            k: {"accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0, **v}
            for k, v in by_difficulty.items()
        },
    )

    logger.info(
        "Evaluation run complete",
        run_id=run_id,
        accuracy=summary.accuracy,
        avg_latency=summary.avg_latency_ms,
    )

    return EvalRunResponse(
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        summary=summary,
        results=results,
    )


@router.get("/history")
async def evaluation_history(limit: int = 10, db: AsyncSession = Depends(get_db_session)):
    """
    Get history of evaluation runs.
    """
    result = await db.execute(
        text("""
            SELECT
                run_id,
                COUNT(*) as total_questions,
                SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_answers,
                AVG(similarity_score) as avg_similarity,
                AVG(latency_ms) as avg_latency,
                MIN(created_at) as started_at,
                MAX(created_at) as completed_at
            FROM eval_results
            GROUP BY run_id
            ORDER BY MIN(created_at) DESC
            LIMIT :limit
        """),
        {"limit": limit},
    )

    runs = result.fetchall()

    return [
        {
            "run_id": r.run_id,
            "total_questions": r.total_questions,
            "correct_answers": r.correct_answers,
            "accuracy": round(r.correct_answers / r.total_questions, 4)
            if r.total_questions > 0
            else 0.0,
            "avg_similarity": round(float(r.avg_similarity or 0), 4),
            "avg_latency_ms": round(float(r.avg_latency or 0), 2),
            "started_at": r.started_at,
            "completed_at": r.completed_at,
        }
        for r in runs
    ]


@router.get("/run/{run_id}")
async def get_run_details(run_id: str, db: AsyncSession = Depends(get_db_session)):
    """
    Get detailed results for a specific evaluation run.
    """
    result = await db.execute(
        text("""
            SELECT
                er.*, eq.question, eq.ground_truth_answer, eq.query_type, eq.difficulty
            FROM eval_results er
            JOIN eval_questions eq ON er.question_id = eq.id
            WHERE er.run_id = :run_id
            ORDER BY er.question_id
        """),
        {"run_id": run_id},
    )

    results = result.fetchall()

    if not results:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return {
        "run_id": run_id,
        "results": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "expected_answer": r.ground_truth_answer,
                "actual_answer": r.actual_answer,
                "query_type": r.query_type,
                "difficulty": r.difficulty,
                "is_correct": r.is_correct,
                "similarity_score": round(float(r.similarity_score or 0), 4),
                "latency_ms": r.latency_ms,
                "created_at": r.created_at,
            }
            for r in results
        ],
    }
