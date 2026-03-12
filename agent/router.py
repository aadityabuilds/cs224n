"""LLM-based routing: the model decides its own learning strategy.

The student model sees the problem, its response, score, and feedback,
then decides which action to take:
- <sdpo>: Fundamental failure → needs gradient update via self-distillation
- <rag>: Partial success with specific errors → store feedback as knowledge
- <pass>: Solution is correct → no action needed

Falls back to rule-based routing if the model output can't be parsed.
"""
import logging
import re
import torch

logger = logging.getLogger(__name__)

# System prompt for the routing decision
ROUTING_SYSTEM_PROMPT = """You are a self-improving coding agent analyzing the results of your code generation attempts.

After each attempt, you must choose exactly ONE learning action by outputting the corresponding tag:

<pass> - Your solution is correct or nearly correct (score >= 0.9). No action needed.

<rag> - Your solution has partial understanding but specific errors (0 < score < 0.9). You should store a LESSON about what mistake you made (e.g., off-by-one error, wrong algorithm, missed edge case) so you can avoid it in future similar problems.

<sdpo> - Your solution is fundamentally wrong (score = 0 or very low). You need targeted training on this type of problem to improve your approach.

Analyze the result carefully and output ONLY the action tag on a single line. No explanation needed."""

ROUTING_USER_TEMPLATE = """Problem summary: {problem_summary}

Your solution scored: {score:.2f} ({accuracy:.1%} of tests passed)

Feedback from test execution:
{feedback}

Choose your action: <sdpo>, <rag>, or <pass>"""


def llm_route_decision(agent_model, problem: str, score: float, accuracy: float,
                        feedback: str) -> tuple[str, str | None]:
    """Use the student model to decide the routing action.

    Args:
        agent_model: AgentModel instance
        problem: the problem description (truncated for prompt)
        score: verification score (0-1)
        accuracy: test accuracy (0-1)
        feedback: verification feedback text

    Returns:
        (action, payload) where action is "sdpo", "rag", or "pass"
        and payload is the knowledge chunk for RAG (or None).
    """
    # Build the routing prompt
    problem_summary = problem[:500] + ("..." if len(problem) > 500 else "")
    feedback_text = feedback[:1000] if feedback else "All tests passed."

    user_msg = ROUTING_USER_TEMPLATE.format(
        problem_summary=problem_summary,
        score=score,
        accuracy=accuracy,
        feedback=feedback_text,
    )

    # Generate routing decision (short, greedy)
    try:
        responses, _, _ = agent_model.generate(
            prompt=user_msg,
            system_prompt=ROUTING_SYSTEM_PROMPT,
            num_return_sequences=1,
            temperature=0.0,
            max_new_tokens=32,  # Only need a tag
        )
        response = responses[0].strip().lower()
        logger.info(f"  Router LLM raw output: '{response[:120]}'")

        # Parse the action tag
        action = _parse_action(response)
        if action:
            logger.info(f"  Router decision: <{action}> (LLM-chosen)")
            payload = _build_payload(action, feedback) if action == "rag" else None
            return action, payload
        else:
            logger.info(f"  Router: could not parse LLM output, falling back to rule-based")

    except Exception as e:
        logger.warning(f"  Router LLM error: {e} — falling back to rule-based")

    # Fallback to rule-based
    action, payload = rule_based_route(score, accuracy, feedback)
    logger.info(f"  Router decision: <{action}> (rule-based fallback, score={score:.3f})")
    return action, payload


def _parse_action(text: str) -> str | None:
    """Parse an action tag from model output."""
    text = text.strip().lower()

    # Look for tags anywhere in the text
    if '<sdpo>' in text or 'sdpo' in text.split():
        return 'sdpo'
    if '<rag>' in text or 'rag' in text.split():
        return 'rag'
    if '<pass>' in text or 'pass' in text.split():
        return 'pass'

    return None


def _build_payload(action: str, feedback: str) -> str | None:
    """Build a lesson-oriented payload for RAG storage.

    Stores what went wrong (the mistake type) rather than the raw
    failed solution, so retrieved chunks help avoid errors rather
    than contaminate with bad code.
    """
    if action == "rag" and feedback:
        return f"Lesson from mistake: {feedback[:500]}"
    return None


def rule_based_route(score: float, accuracy: float, feedback: str) -> tuple[str, str | None]:
    """Fallback rule-based routing.

    - score >= 0.9: pass
    - 0 < score < 0.9: rag (partial success, store feedback)
    - score == 0: sdpo (total failure, needs gradient update)
    """
    if score >= 0.9:
        return ("pass", None)

    if score > 0:
        knowledge = f"Lesson from mistake: {feedback[:500]}" if feedback else None
        return ("rag", knowledge)

    return ("sdpo", None)
