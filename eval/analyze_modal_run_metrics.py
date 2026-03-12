#!/usr/bin/env python3
"""Pull running metrics for active 7B Modal SDPO runs.

For each target app, this script:
1) Detects if it is currently active via `modal app list --json`
2) Finds the latest checkpoint step in its backing volume
3) Reads checkpoint metrics to compute action counts + initial perfect count
4) Computes "future-attempt perfects" from decision logs (v2/v3) or log fallback (v1)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


PERFECT_THRESHOLD = 0.999


@dataclass(frozen=True)
class RunSpec:
    app_name: str
    volume: str
    log_file: str
    decision_file: str | None


RUN_SPECS: dict[str, RunSpec] = {
    "cs224n-7b-sdpo": RunSpec(
        app_name="cs224n-7b-sdpo",
        volume="cs224n-7b-results",
        log_file="/training_7b.log",
        decision_file=None,
    ),
    "cs224n-7b-sdpo-v2": RunSpec(
        app_name="cs224n-7b-sdpo-v2",
        volume="cs224n-7b-v2-results",
        log_file="/training_7b_v2.log",
        decision_file="decision_log.json",
    ),
    "cs224n-7b-sdpo-v3": RunSpec(
        app_name="cs224n-7b-sdpo-v3",
        volume="cs224n-7b-v3-results",
        log_file="/training_7b_v3.log",
        decision_file="decisions.json",
    ),
}


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def get_app_list() -> list[dict[str, Any]]:
    text = run_cmd(["modal", "app", "list", "--json"])
    return json.loads(text)


def latest_step_for_volume(volume_name: str) -> int:
    text = run_cmd(["modal", "volume", "ls", volume_name, "/checkpoints"])
    steps = [int(m.group(1)) for m in re.finditer(r"step_(\d+)", text)]
    if not steps:
        raise RuntimeError(f"No checkpoints found in volume {volume_name}")
    return max(steps)


def get_volume_json(volume_name: str, remote_path: str) -> Any:
    text = run_cmd(["modal", "volume", "get", volume_name, remote_path, "-"])
    stripped = text.lstrip()
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(stripped)
    return obj


def get_volume_text(volume_name: str, remote_path: str) -> str:
    return run_cmd(["modal", "volume", "get", volume_name, remote_path, "-"])


def compute_action_counts(metrics: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"sdpo": 0, "rag": 0, "pass": 0}
    for m in metrics:
        action = str(m.get("action", "")).lower()
        if action in counts:
            counts[action] += 1
    return counts


def compute_initial_perfect_count(metrics: list[dict[str, Any]]) -> int:
    return sum(1 for m in metrics if float(m.get("score", 0.0)) >= PERFECT_THRESHOLD)


def compute_future_perfects_from_decisions(decisions: list[dict[str, Any]]) -> tuple[int, int]:
    # Returns: (total_sdpo_attempt_perfects, perfects_at_attempt_gt_1)
    perfect_total = 0
    perfect_future = 0
    for d in decisions:
        details = d.get("sdpo_rollout_details")
        if not details:
            continue
        for problem_rollout in details:
            for rollout in problem_rollout.get("rollouts", []):
                for att in rollout.get("attempts", []):
                    if att.get("is_perfect"):
                        perfect_total += 1
                        if int(att.get("attempt", 1)) > 1:
                            perfect_future += 1
    return perfect_total, perfect_future


def compute_future_perfects_from_log(log_text: str) -> tuple[int, int]:
    # Fallback for v1 where checkpoint decision logs are unavailable.
    perfect_total = 0
    perfect_future = 0
    for line in log_text.splitlines():
        m = re.search(r"PERFECT at attempt (\d+)", line)
        if not m:
            continue
        perfect_total += 1
        if int(m.group(1)) > 1:
            perfect_future += 1
    return perfect_total, perfect_future


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze active 7B Modal run metrics.")
    parser.add_argument(
        "--apps",
        nargs="*",
        default=list(RUN_SPECS.keys()),
        help=f"Subset of apps to analyze. Choices: {', '.join(RUN_SPECS.keys())}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    unknown = [a for a in args.apps if a not in RUN_SPECS]
    if unknown:
        print(f"Unknown app(s): {unknown}", file=sys.stderr)
        return 2

    apps = get_app_list()
    app_by_name = {a.get("Description"): a for a in apps}

    rows: list[dict[str, Any]] = []
    for app_name in args.apps:
        spec = RUN_SPECS[app_name]
        app_info = app_by_name.get(spec.app_name)

        is_active = False
        app_id = None
        state = "not_found"
        if app_info:
            state = str(app_info.get("State", "unknown"))
            app_id = app_info.get("App ID")
            is_active = "ephemeral" in state.lower() and "detached" in state.lower()

        latest_step = latest_step_for_volume(spec.volume)
        metrics = get_volume_json(
            spec.volume,
            f"/checkpoints/step_{latest_step}/metrics.json",
        )
        if not isinstance(metrics, list):
            raise RuntimeError(
                f"Unexpected metrics format for {spec.app_name} at step {latest_step}"
            )

        action_counts = compute_action_counts(metrics)
        initial_perfect = compute_initial_perfect_count(metrics)

        perfect_total = None
        perfect_future = None
        future_source = None

        if spec.decision_file:
            try:
                decisions = get_volume_json(
                    spec.volume,
                    f"/checkpoints/step_{latest_step}/{spec.decision_file}",
                )
                if isinstance(decisions, list):
                    perfect_total, perfect_future = compute_future_perfects_from_decisions(
                        decisions
                    )
                    future_source = "decision_log"
            except Exception:
                pass

        if perfect_total is None or perfect_future is None:
            try:
                log_text = get_volume_text(spec.volume, spec.log_file)
                perfect_total, perfect_future = compute_future_perfects_from_log(log_text)
                future_source = "training_log"
            except Exception:
                perfect_total, perfect_future = 0, 0
                future_source = "unavailable"

        rows.append(
            {
                "app_name": spec.app_name,
                "app_id": app_id,
                "state": state,
                "is_active": is_active,
                "volume": spec.volume,
                "latest_step": latest_step,
                "num_steps_observed": len(metrics),
                "action_counts": action_counts,
                "initial_perfect_count": initial_perfect,
                "sdpo_attempt_perfect_total": perfect_total,
                "sdpo_attempt_perfect_future": perfect_future,
                "future_perfect_source": future_source,
            }
        )

    print(json.dumps({"runs": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
