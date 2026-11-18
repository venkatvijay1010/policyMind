from unittest.mock import AsyncMock, patch

from app.application.warm_up import warm_up_models


async def test_warm_up_primes_router_and_chat_prompts():
    with (
        patch("app.application.warm_up.QueryClassifier") as classifier,
        patch("app.application.warm_up.ConversationAgent") as chat,
    ):
        classifier.return_value.classify = AsyncMock()
        chat.return_value.answer = AsyncMock()

        await warm_up_models()

    classifier.return_value.classify.assert_awaited_once()
    chat.return_value.answer.assert_awaited_once()


async def test_warm_up_failure_never_raises():
    with (
        patch("app.application.warm_up.QueryClassifier") as classifier,
        patch("app.application.warm_up.ConversationAgent") as chat,
    ):
        classifier.return_value.classify = AsyncMock(side_effect=RuntimeError("offline"))
        chat.return_value.answer = AsyncMock()

        await warm_up_models()

    chat.return_value.answer.assert_not_awaited()
