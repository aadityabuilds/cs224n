"""Wrapper around SDPO's compute_score, loaded via importlib to avoid ray dependency."""
import importlib.util
import logging
import os
import pathlib
import re

logger = logging.getLogger(__name__)

_compute_score = None


def _get_compute_score():
    global _compute_score
    if _compute_score is not None:
        return _compute_score

    sdpo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SDPO")
    path = pathlib.Path(sdpo_path) / "verl/utils/reward_score/feedback/code.py"
    spec = importlib.util.spec_from_file_location("feedback_code", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _compute_score = mod.compute_score
    return _compute_score


def _fix_truncated_code_blocks(solution: str) -> str:
    """If the response has an opening ```python but no closing ```, append one."""
    # Count backtick blocks
    opens = len(re.findall(r'```', solution))
    if opens % 2 == 1:
        # Odd number of ``` means one is unclosed — append closing
        solution = solution.rstrip() + "\n```"
    return solution


def verify_solution(solution: str, tests_json: str, split: str = "train") -> dict:
    """Run solution against test cases and return structured feedback."""
    # Fix truncated code blocks (from hitting max_new_tokens)
    solution = _fix_truncated_code_blocks(solution)

    compute_score = _get_compute_score()
    result = compute_score(
        solution=solution,
        ground_truth=tests_json,
        extra_info={"split": split},
        sparse_rewards=False,
    )
    return result
