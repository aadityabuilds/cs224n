"""Sweep eval across ALL checkpoints for v1 (4 attempts).

Discovers every step_N checkpoint on the volume, evaluates each for
model+sdpo and model+rag+sdpo in parallel, runs baseline once, and
prints a progression table at the end.

Usage:
  modal run eval_sweep_7b.py
  modal run eval_sweep_7b.py --num-eval-problems 50
"""
# This file uses code from the SDPO (Self-Distillation with Policy Optimization) framework.
# SDPO is licensed under the Apache License, Version 2.0.
# Copyright 2025 Hübotter, Lübeck, Behric, Baumann, Bagatella, Marta, Hakimi, Shenfeld, Kleine Buening, Guestrin, Krause
# Source: https://github.com/lasgroup/SDPO
# License: http://www.apache.org/licenses/LICENSE-2.0
import modal

app = modal.App("cs224n-7b-eval-sweep")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.45.0",
        "accelerate>=0.30.0",
        "datasets>=2.18.0",
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.0",
        "sentence-transformers>=2.2.0",
        "huggingface_hub",
        "bitsandbytes",
    )
    .run_commands(
        f"python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; "
        f"AutoTokenizer.from_pretrained('{MODEL_NAME}'); "
        f"AutoModelForCausalLM.from_pretrained('{MODEL_NAME}')\"",
        f"python -c \"from sentence_transformers import SentenceTransformer; "
        f"SentenceTransformer('{EMBEDDING_MODEL}')\"",
    )
    .add_local_dir(".", remote_path="/root/cs224n", ignore=[
        ".venv", "__pycache__", ".git", "uv.lock",
        "*.pyc", "*.egg-info", "*.so", "checkpoints", "*.log",
    ])
)

volume = modal.Volume.from_name("cs224n-7b-results", create_if_missing=True)
VOLUME_PATH = "/results"


def _setup_env():
    import os, sys, torch
    os.chdir("/root/cs224n")
    sys.path.insert(0, "/root/cs224n")
    sys.path.insert(0, "/root/cs224n/SDPO")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.makedirs("checkpoints", exist_ok=True)
    return torch


@app.function(
    image=image, gpu="H100", timeout=3600 * 4,
    volumes={VOLUME_PATH: volume}, memory=32768,
)
def eval_one_checkpoint(checkpoint_step: int, num_eval_problems: int = 50):
    """Evaluate a single checkpoint for model+sdpo and model+rag+sdpo."""
    import json, logging, os, shutil

    torch = _setup_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()], force=True,
    )
    logger = logging.getLogger(f"sweep_v1_step{checkpoint_step}")
    volume.reload()

    vol_model = f"{VOLUME_PATH}/checkpoints/step_{checkpoint_step}/model"
    vol_rag = f"{VOLUME_PATH}/checkpoints/step_{checkpoint_step}/rag_db.json"
    local_model = "checkpoints/eval_model"
    local_rag = "checkpoints/eval_rag.json"

    if not os.path.exists(vol_model):
        logger.error(f"Model not found: {vol_model}")
        return {"step": checkpoint_step, "error": "model not found"}

    shutil.copytree(vol_model, local_model, dirs_exist_ok=True)

    has_rag = os.path.exists(vol_rag)
    if has_rag:
        shutil.copy(vol_rag, local_rag)

    from data.utils.livecodebench import load_livecodebench
    from agent.config import build_code_prompt
    from agent.model_7b import AgentModel7B
    from agent.verification import verify_solution
    from agent.rag import RAGDatabase

    logger.info(f"Loading test dataset...")
    test_dataset = load_livecodebench("test")

    trained = AgentModel7B(model_name=local_model, lr=1e-6, ema_rate=0.005)

    from eval_7b import evaluate_model

    sdpo_metrics = evaluate_model(
        trained, test_dataset, f"step{checkpoint_step}_sdpo",
        max_problems=num_eval_problems, rag_db=None,
    )

    rag_metrics = None
    if has_rag:
        rag_db = RAGDatabase()
        with open(local_rag) as f:
            for chunk in json.load(f):
                rag_db.add(chunk)
        logger.info(f"RAG: {rag_db.size} chunks")
        rag_metrics = evaluate_model(
            trained, test_dataset, f"step{checkpoint_step}_rag_sdpo",
            max_problems=num_eval_problems, rag_db=rag_db,
        )

    del trained
    torch.cuda.empty_cache()

    result = {
        "step": checkpoint_step,
        "model+sdpo": {
            "avg_score": sdpo_metrics["avg_score"],
            "pass_rate": sdpo_metrics["pass_rate"],
            "num_correct": sdpo_metrics["num_correct"],
            "num_problems": sdpo_metrics["num_problems"],
        },
    }
    if rag_metrics:
        result["model+rag+sdpo"] = {
            "avg_score": rag_metrics["avg_score"],
            "pass_rate": rag_metrics["pass_rate"],
            "num_correct": rag_metrics["num_correct"],
            "num_problems": rag_metrics["num_problems"],
        }

    logger.info(f"\n=== STEP {checkpoint_step} DONE ===")
    logger.info(f"  model+sdpo:     acc={sdpo_metrics['avg_score']:.4f}  solve={sdpo_metrics['pass_rate']:.4f}")
    if rag_metrics:
        logger.info(f"  model+rag+sdpo: acc={rag_metrics['avg_score']:.4f}  solve={rag_metrics['pass_rate']:.4f}")

    with open(f"{VOLUME_PATH}/sweep_v1_step{checkpoint_step}.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()

    return result


@app.function(
    image=image, gpu="H100", timeout=3600 * 4,
    volumes={VOLUME_PATH: volume}, memory=32768,
)
def eval_baseline(num_eval_problems: int = 50):
    """Evaluate the base Qwen 7B model (run once)."""
    import json, logging
    torch = _setup_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()], force=True,
    )
    logger = logging.getLogger("sweep_v1_baseline")

    from data.utils.livecodebench import load_livecodebench
    from agent.model_7b import AgentModel7B
    from eval_7b import evaluate_model

    test_dataset = load_livecodebench("test")
    base = AgentModel7B(model_name=MODEL_NAME, lr=1e-6, ema_rate=0.005)
    base_metrics = evaluate_model(
        base, test_dataset, "base_qwen7b",
        max_problems=num_eval_problems, rag_db=None,
    )
    del base
    torch.cuda.empty_cache()

    result = {
        "step": 0,
        "base_qwen7b": {
            "avg_score": base_metrics["avg_score"],
            "pass_rate": base_metrics["pass_rate"],
            "num_correct": base_metrics["num_correct"],
            "num_problems": base_metrics["num_problems"],
        },
    }
    with open(f"{VOLUME_PATH}/sweep_v1_baseline.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    return result


@app.function(
    image=image, timeout=120,
    volumes={VOLUME_PATH: volume}, memory=512,
)
def list_checkpoints():
    """Discover all step_N checkpoints on the volume."""
    import os, re
    volume.reload()
    ckpt_dir = f"{VOLUME_PATH}/checkpoints"
    if not os.path.exists(ckpt_dir):
        return []
    steps = []
    for name in os.listdir(ckpt_dir):
        m = re.match(r"step_(\d+)", name)
        if m and os.path.exists(f"{ckpt_dir}/{name}/model"):
            steps.append(int(m.group(1)))
    return sorted(steps)


@app.local_entrypoint()
def main(num_eval_problems: int = 50, min_step: int = 0):
    """Sweep eval across v1 checkpoints.

    Usage:
      modal run eval_sweep_7b.py
      modal run eval_sweep_7b.py --num-eval-problems 30
      modal run eval_sweep_7b.py --min-step 200   # only checkpoints after 200 (250, 300, ...)
    """
    print("=" * 80)
    print("  v1 CHECKPOINT SWEEP EVAL (4 attempts)")
    print("=" * 80)

    all_steps = list_checkpoints.remote()
    steps = [s for s in all_steps if s > min_step] if min_step else all_steps
    print(f"\nFound checkpoints: {all_steps} (evaluating steps > {min_step}: {steps})")
    if not steps:
        print("No checkpoints to evaluate!")
        return

    print(f"\nLaunching {len(steps)} checkpoint evals + 1 baseline in parallel...")
    print(f"  Eval problems per checkpoint: {num_eval_problems}")
    print()

    baseline_future = eval_baseline.spawn(num_eval_problems=num_eval_problems)
    checkpoint_futures = []
    for step in steps:
        f = eval_one_checkpoint.spawn(
            checkpoint_step=step, num_eval_problems=num_eval_problems,
        )
        checkpoint_futures.append((step, f))

    baseline_result = baseline_future.get()
    base_acc = baseline_result["base_qwen7b"]["avg_score"]
    base_solve = baseline_result["base_qwen7b"]["pass_rate"]
    base_correct = baseline_result["base_qwen7b"]["num_correct"]
    base_n = baseline_result["base_qwen7b"]["num_problems"]

    results = {}
    for step, f in checkpoint_futures:
        try:
            r = f.get()
            results[step] = r
        except Exception as e:
            print(f"  [WARN] Step {step} failed: {e}")
            results[step] = {"step": step, "error": str(e)}

    # Print progression table
    print("\n" + "=" * 100)
    print(f"  v1 CHECKPOINT PROGRESSION (baseline acc={base_acc:.4f}  solve={base_solve:.4f}  [{base_correct}/{base_n}])")
    print("=" * 100)
    header = (f"  {'Step':>6}  |  {'SDPO Acc':>10} {'SDPO Solve':>12} {'Correct':>9}"
              f"  |  {'RAG+SDPO Acc':>13} {'RAG+SDPO Solve':>15} {'Correct':>9}")
    print(header)
    print("-" * 100)

    for step in sorted(results.keys()):
        r = results[step]
        if "error" in r:
            print(f"  {step:>6}  |  {'ERROR':>10} {r.get('error','')}")
            continue

        sdpo = r.get("model+sdpo", {})
        rag = r.get("model+rag+sdpo", {})

        sdpo_acc = f"{sdpo['avg_score']:.4f}" if sdpo else "N/A"
        sdpo_solve = f"{sdpo['pass_rate']:.4f}" if sdpo else "N/A"
        sdpo_correct = f"{sdpo.get('num_correct','?')}/{sdpo.get('num_problems','?')}" if sdpo else "N/A"

        rag_acc = f"{rag['avg_score']:.4f}" if rag else "N/A"
        rag_solve = f"{rag['pass_rate']:.4f}" if rag else "N/A"
        rag_correct = f"{rag.get('num_correct','?')}/{rag.get('num_problems','?')}" if rag else "N/A"

        print(f"  {step:>6}  |  {sdpo_acc:>10} {sdpo_solve:>12} {sdpo_correct:>9}"
              f"  |  {rag_acc:>13} {rag_solve:>15} {rag_correct:>9}")

    print("=" * 100)
    print(f"  Baseline: acc={base_acc:.4f}  solve={base_solve:.4f}  [{base_correct}/{base_n}]")
    print("=" * 100)
    print("  Test-Case Acc = avg fraction of test cases passed per problem")
    print("  Solve Rate    = fraction of problems fully solved (100% test cases)")
