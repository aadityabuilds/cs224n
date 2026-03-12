"""Wrapper around SDPO's compute_score, loaded via importlib to avoid ray dependency."""
# This file uses code from the SDPO (Self-Distillation with Policy Optimization) framework.
# SDPO is licensed under the Apache License, Version 2.0.
# Copyright 2025 Hübotter, Lübeck, Behric, Baumann, Bagatella, Marta, Hakimi, Shenfeld, Kleine Buening, Guestrin, Krause
# Source: https://github.com/lasgroup/SDPO
# License: http://www.apache.org/licenses/LICENSE-2.0
import importlib.util
import json
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


def _unwrap_class_solution(solution: str, fn_name: str) -> str:
    """Make class methods callable as top-level functions for the verifier.

    The model often generates LeetCode-style code:
        class Solution:
            def circularGameLosers(self, n, k):
                ...

    But the verifier expects a top-level function named ``fn_name``.
    Instead of regex-extracting and dedenting the method (fragile), we append
    a one-line alias ``fn_name = ClassName().fn_name`` at the end of the code
    block. The bound method already has ``self`` filled in, so the verifier
    can call it like a normal function.
    """
    if not fn_name:
        return solution

    blocks = list(re.finditer(r'```\w*\n(.*?)```', solution, re.DOTALL))
    if not blocks:
        return solution

    for block_match in blocks:
        code = block_match.group(1)

        # Already has a top-level function with this name — no transformation needed
        if re.search(r'^def\s+' + re.escape(fn_name) + r'\s*\(', code, re.MULTILINE):
            continue

        # Find a class containing this method (with self parameter)
        class_match = re.search(
            r'^class\s+(\w+).*?:'
            r'.*?'
            r'def\s+' + re.escape(fn_name) + r'\s*\(\s*self\b',
            code, re.DOTALL | re.MULTILINE
        )
        if not class_match:
            continue

        class_name = class_match.group(1)
        alias_line = f"\n{fn_name} = {class_name}().{fn_name}\n"

        old_block = block_match.group(0)
        lang_match = re.match(r'```(\w*)', old_block)
        lang = lang_match.group(1) if lang_match else ''
        new_code = code.rstrip() + alias_line
        new_block = f"```{lang}\n{new_code}```"
        solution = solution.replace(old_block, new_block)
        break

    return solution


def _compact_test_inputs(tests_json: str) -> str:
    """Compact JSON in test inputs to avoid whitespace-splitting issues.

    The SDPO verifier splits string inputs with str.split() (all whitespace),
    which breaks JSON arrays containing spaces like '[1, 3, 2]' into
    ['[1,', '3,', '2]']. Fix by re-serializing each argument's JSON compactly.
    Multi-argument inputs are newline-separated, so we split on '\\n' first.
    """
    try:
        test_cases = json.loads(tests_json)
        inputs = test_cases.get("inputs")
        if not inputs:
            return tests_json

        changed = False
        new_inputs = []
        for inp in inputs:
            if not isinstance(inp, str):
                new_inputs.append(inp)
                continue
            parts = inp.split('\n')
            compacted = []
            for part in parts:
                try:
                    val = json.loads(part)
                    compact = json.dumps(val, separators=(',', ':'))
                    if compact != part:
                        changed = True
                    compacted.append(compact)
                except (json.JSONDecodeError, ValueError):
                    compacted.append(part)
            new_inputs.append('\n'.join(compacted))

        if changed:
            test_cases['inputs'] = new_inputs
            return json.dumps(test_cases)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return tests_json


def verify_solution(solution: str, tests_json: str, split: str = "train") -> dict:
    """Run solution against test cases and return structured feedback."""
    # Fix truncated code blocks (from hitting max_new_tokens)
    solution = _fix_truncated_code_blocks(solution)

    # Unwrap class Solution methods into standalone functions
    # so the verifier can find them by fn_name
    try:
        test_cases = json.loads(tests_json)
        fn_name = test_cases.get("fn_name", "")
        if fn_name:
            solution = _unwrap_class_solution(solution, fn_name)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Compact test inputs to avoid split()-related JSON parsing errors
    tests_json = _compact_test_inputs(tests_json)

    compute_score = _get_compute_score()
    result = compute_score(
        solution=solution,
        ground_truth=tests_json,
        extra_info={"split": split},
        sparse_rewards=False,
    )
    return result
