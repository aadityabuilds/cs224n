"""Rule-based routing: decides SDPO / RAG / pass based on score and feedback."""
import logging

logger = logging.getLogger(__name__)


def route_decision(score: float, accuracy: float, feedback: str) -> tuple[str, str | None]:
    """Decide next action based on verification results.

    Rules:
    - score == 1.0: pass (solution is correct)
    - score > 0 but < 1.0: rag (partial success — store what went wrong as knowledge)
    - score == 0: sdpo (total failure — needs gradient update to fix procedure)
    """
    if score > 0.99:
        return ("pass", None)

    if score > 0:
        # Partial success: extract useful knowledge from feedback
        knowledge = f"Common mistake: {feedback[:500]}" if feedback else None
        return ("rag", knowledge)

    # Total failure: SDPO gradient update
    return ("sdpo", None)
