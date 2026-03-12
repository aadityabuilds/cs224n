"""Export RAG chunks from v1/v2/v3 runs into a local Markdown file.

This script reads all checkpoint rag_db.json files from:
- cs224n-7b-results
- cs224n-7b-v2-results
- cs224n-7b-v3-results

It also includes /results/rag_db.json (if present) for each run.

Usage:
  modal run export_rag_chunks_md.py
  modal run export_rag_chunks_md.py --output-file rag_chunks_all_runs.md
  modal run export_rag_chunks_md.py --run v1-newrag --output-file rag_chunks_v1_newrag.md
"""

from __future__ import annotations

import modal

app = modal.App("cs224n-rag-export")

volume_v1 = modal.Volume.from_name("cs224n-7b-results", create_if_missing=True)
volume_v1_newrag = modal.Volume.from_name("cs224n-7b-v1-newrag-results", create_if_missing=True)
volume_v2 = modal.Volume.from_name("cs224n-7b-v2-results", create_if_missing=True)
volume_v3 = modal.Volume.from_name("cs224n-7b-v3-results", create_if_missing=True)

RESULTS_PATH = "/results"


def _collect_one_run(run_label: str) -> dict:
    import json
    import os
    import re

    out: dict = {
        "run_label": run_label,
        "root_chunks": [],
        "checkpoints": [],
    }

    # Optional top-level RAG DB.
    root_rag = os.path.join(RESULTS_PATH, "rag_db.json")
    if os.path.exists(root_rag):
        with open(root_rag) as f:
            data = json.load(f)
        out["root_chunks"] = data if isinstance(data, list) else []

    ckpt_root = os.path.join(RESULTS_PATH, "checkpoints")
    if not os.path.exists(ckpt_root):
        return out

    steps: list[int] = []
    for name in os.listdir(ckpt_root):
        m = re.match(r"step_(\d+)$", name)
        if m:
            steps.append(int(m.group(1)))
    steps.sort()

    for step in steps:
        rag_path = os.path.join(ckpt_root, f"step_{step}", "rag_db.json")
        if not os.path.exists(rag_path):
            continue
        with open(rag_path) as f:
            data = json.load(f)
        chunks = data if isinstance(data, list) else []
        out["checkpoints"].append(
            {
                "step": step,
                "chunk_count": len(chunks),
                "chunks": chunks,
            }
        )

    return out


@app.function(timeout=60 * 20, volumes={RESULTS_PATH: volume_v1})
def collect_v1() -> dict:
    return _collect_one_run("v1 (cs224n-7b-results)")


@app.function(timeout=60 * 20, volumes={RESULTS_PATH: volume_v1_newrag})
def collect_v1_newrag() -> dict:
    return _collect_one_run("v1-newrag (cs224n-7b-v1-newrag-results)")


@app.function(timeout=60 * 20, volumes={RESULTS_PATH: volume_v2})
def collect_v2() -> dict:
    return _collect_one_run("v2 (cs224n-7b-v2-results)")


@app.function(timeout=60 * 20, volumes={RESULTS_PATH: volume_v3})
def collect_v3() -> dict:
    return _collect_one_run("v3 (cs224n-7b-v3-results)")


def _format_run_section(run_data: dict) -> str:
    lines: list[str] = []
    run_label = run_data["run_label"]
    root_chunks = run_data.get("root_chunks", [])
    checkpoints = run_data.get("checkpoints", [])

    lines.append(f"## {run_label}")
    lines.append("")
    lines.append(f"- Top-level `rag_db.json` chunks: **{len(root_chunks)}**")
    lines.append(f"- Checkpoints with `rag_db.json`: **{len(checkpoints)}**")
    lines.append("")

    if root_chunks:
        lines.append("### Top-level `rag_db.json`")
        lines.append("")
        for i, chunk in enumerate(root_chunks, start=1):
            lines.append(f"#### Root Chunk {i}")
            lines.append("")
            lines.append("```text")
            lines.append(str(chunk))
            lines.append("```")
            lines.append("")

    for ckpt in checkpoints:
        step = ckpt["step"]
        chunks = ckpt.get("chunks", [])
        lines.append(f"### Checkpoint step_{step} ({len(chunks)} chunks)")
        lines.append("")
        for i, chunk in enumerate(chunks, start=1):
            lines.append(f"#### step_{step} - Chunk {i}")
            lines.append("")
            lines.append("```text")
            lines.append(str(chunk))
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


@app.local_entrypoint()
def main(output_file: str = "rag_chunks_all_runs.md", run: str = "all"):
    """run: 'all' (v1,v2,v3), 'v1-newrag', or any single run label."""
    if run == "v1-newrag":
        print("Collecting RAG chunks from v1-newrag run...")
        data = collect_v1_newrag.remote()
        run_data_list = [data]
        title = "RAG Chunk Export (v1-newrag)"
    elif run == "all":
        print("Collecting RAG chunks from all three runs in parallel...")
        f1 = collect_v1.spawn()
        f2 = collect_v2.spawn()
        f3 = collect_v3.spawn()
        data_v1 = f1.get()
        data_v2 = f2.get()
        data_v3 = f3.get()
        run_data_list = [data_v1, data_v2, data_v3]
        title = "RAG Chunk Export (v1, v2, v3)"
    else:
        print(f"Unknown run: {run}. Use 'all' or 'v1-newrag'.")
        return

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        "This file contains the raw text chunks stored in each run's "
        "`rag_db.json` at both top-level and per-checkpoint."
    )
    lines.append("")

    for run_data in run_data_list:
        lines.append(_format_run_section(run_data))
        lines.append("")

    content = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Done. Wrote Markdown export to: {output_file}")
