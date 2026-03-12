"""Modal app: Qwen2.5-7B v2 — 16 sequential attempts, full decision logging.

Key differences from modal_app_7b.py:
- max_sequential_attempts = 16 (up from 4)
- sdpo_batch_size = 1 (longer rollouts = more time per batch)
- Full decision log saved to volume
- Eval over 100 test problems
"""
import modal

app = modal.App("cs224n-7b-sdpo-v2")

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
        ".venv",
        "__pycache__",
        ".git",
        "uv.lock",
        "*.pyc",
        "*.egg-info",
        "*.so",
        "checkpoints",
        "*.log",
    ])
)

volume = modal.Volume.from_name("cs224n-7b-v2-results", create_if_missing=True)
VOLUME_PATH = "/results"


def _setup_env():
    import logging
    import os
    import sys
    import torch

    os.chdir("/root/cs224n")
    sys.path.insert(0, "/root/cs224n")
    sys.path.insert(0, "/root/cs224n/SDPO")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    os.makedirs("checkpoints", exist_ok=True)
    return torch


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,  # 24 hours (Modal max)
    volumes={VOLUME_PATH: volume},
    memory=32768,
)
def run_training_and_eval(
    lr: float = 2e-6,
    ema_rate: float = 0.005,
    num_rollouts: int = 2,
    max_sequential_attempts: int = 16,
    sdpo_batch_size: int = 1,
    temperature: float = 0.7,
    max_problems: int = 647,
    num_eval_problems: int = 100,
    ref_kl_beta: float = 0.0,
    alpha: float = 0.5,
    distillation_topk: int = 50,
    max_new_tokens: int = 1536,
    checkpoint_every: int = 50,
    feedback_truncate: int = 300,
    feedback_window: int = 6,
):
    import json
    import logging
    import os
    import shutil

    torch = _setup_env()
    os.makedirs(f"{VOLUME_PATH}/checkpoints", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{VOLUME_PATH}/training_7b_v2.log"),
        ],
        force=True,
    )
    logger = logging.getLogger("modal_7b_v2")

    logger.info(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory
        logger.info(f"GPU memory: {mem / 1e9:.1f} GB")

    # ---- PHASE 1: Training ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: TRAINING v2 (Qwen2.5-7B, 16 attempts)")
    logger.info("=" * 70)

    from agent.config import SelfDistillationConfig
    from main_7b_v2 import AgentConfig7Bv2, run_training_loop_7b_v2

    config = AgentConfig7Bv2(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_problems=max_problems,
        num_rollouts=num_rollouts,
        sdpo_batch_size=sdpo_batch_size,
        temperature=temperature,
        lr=lr,
        ema_rate=ema_rate,
        max_sequential_attempts=max_sequential_attempts,
        max_new_tokens=max_new_tokens,
        checkpoint_every=checkpoint_every,
        max_reprompt_tokens=3072,
        feedback_truncate=feedback_truncate,
        feedback_window=feedback_window,
    )

    sdpo_config = SelfDistillationConfig(
        reference_kl_beta=ref_kl_beta,
        alpha=alpha,
        distillation_topk=distillation_topk,
    )

    logger.info(f"Config: {config}")
    logger.info(f"SDPO config: {sdpo_config}")
    logger.info(f"GPU memory before model load: "
                f"{torch.cuda.memory_allocated()/1e9:.1f}GB allocated, "
                f"{torch.cuda.memory_reserved()/1e9:.1f}GB reserved")

    def checkpoint_callback(ckpt_path, rag_path, step, metrics, decision_log):
        dst = f"{VOLUME_PATH}/checkpoints/step_{step}"
        os.makedirs(dst, exist_ok=True)
        if os.path.isdir(ckpt_path):
            shutil.copytree(ckpt_path, f"{dst}/model", dirs_exist_ok=True)
        if os.path.exists(rag_path):
            shutil.copy(rag_path, f"{dst}/rag_db.json")
        rag_map_path = rag_path.replace("_rag.json", "_rag_id_map.json")
        if os.path.exists(rag_map_path):
            shutil.copy(rag_map_path, f"{dst}/rag_id_map.json")
        with open(f"{dst}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(f"{dst}/decision_log.json", "w") as f:
            json.dump(decision_log, f, indent=2, default=str)
        volume.commit()
        logger.info(f"Checkpoint persisted: {dst}")

    all_metrics, agent_model, rag_db, decision_log = run_training_loop_7b_v2(
        config, sdpo_config=sdpo_config, checkpoint_callback=checkpoint_callback
    )

    rag_db_path = "checkpoints/rag_db.json"
    with open(rag_db_path, "w") as f:
        json.dump(rag_db.texts, f, indent=2)

    # Save all artifacts to volume
    if os.path.exists("training_metrics_v2.json"):
        shutil.copy("training_metrics_v2.json", f"{VOLUME_PATH}/training_metrics_v2.json")
    if os.path.exists("training_decisions_v2.json"):
        shutil.copy("training_decisions_v2.json", f"{VOLUME_PATH}/training_decisions_v2.json")
    shutil.copy(rag_db_path, f"{VOLUME_PATH}/rag_db.json")
    volume.commit()

    del agent_model
    torch.cuda.empty_cache()

    # ---- PHASE 2: Evaluation (100 problems) ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: EVALUATION v2 (7B, 100 problems)")
    logger.info("=" * 70)

    from eval_7b import run_eval_7b

    latest_ckpt = "checkpoints/final_model"
    if not os.path.exists(latest_ckpt):
        latest_ckpt = f"checkpoints/checkpoint_step{max_problems}"

    eval_results = run_eval_7b(
        trained_model_path=latest_ckpt,
        rag_db_path=rag_db_path,
        base_model_name="Qwen/Qwen2.5-7B-Instruct",
        max_problems=num_eval_problems,
    )

    if os.path.exists("eval_results_7b.json"):
        shutil.copy("eval_results_7b.json", f"{VOLUME_PATH}/eval_results_7b_v2.json")
    volume.commit()

    _print_results(logger, eval_results)
    return eval_results


@app.function(
    image=image,
    gpu="H100",
    timeout=3600 * 6,
    volumes={VOLUME_PATH: volume},
    memory=32768,
)
def run_eval_only(checkpoint_step: int = 50, num_eval_problems: int = 100,
                  skip_base: bool = False, only_base: bool = False):
    import json
    import logging
    import os
    import shutil

    torch = _setup_env()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger("modal_eval_7b_v2")
    volume.reload()

    vol_model = f"{VOLUME_PATH}/checkpoints/step_{checkpoint_step}/model"
    vol_rag = f"{VOLUME_PATH}/checkpoints/step_{checkpoint_step}/rag_db.json"
    local_model = "checkpoints/eval_model"
    local_rag = "checkpoints/eval_rag.json"

    if not only_base:
        if os.path.exists(vol_model):
            shutil.copytree(vol_model, local_model, dirs_exist_ok=True)
        else:
            logger.error(f"Model not found: {vol_model}")
            return {"error": f"Model not found at {vol_model}"}
        if os.path.exists(vol_rag):
            shutil.copy(vol_rag, local_rag)

    from eval_7b import run_eval_7b
    eval_results = run_eval_7b(
        trained_model_path=local_model if not only_base else "unused",
        rag_db_path=local_rag if not only_base else "nonexistent",
        base_model_name="Qwen/Qwen2.5-7B-Instruct",
        max_problems=num_eval_problems,
        skip_base=skip_base,
        only_base=only_base,
    )

    with open(f"{VOLUME_PATH}/eval_7b_v2_step{checkpoint_step}.json", "w") as f:
        json.dump(eval_results, f, indent=2, default=str)
    volume.commit()

    _print_results(logger, eval_results)
    return eval_results


def _print_results(logger, eval_results):
    logger.info("\n" + "#" * 80)
    logger.info("# 7B v2 RESULTS (16 attempts)")
    logger.info("#" * 80)
    logger.info(f"  {'Config':<25} {'Test-Case Acc':>14} {'Solve Rate':>12} {'Correct':>10}")
    logger.info("-" * 80)
    for key in ["model+rag+sdpo", "base_qwen7b", "model+sdpo"]:
        m = eval_results.get(key, {})
        if "avg_score" in m:
            logger.info(f"  {m['label']:<25} {m['avg_score']:>14.4f} "
                        f"{m['pass_rate']:>12.4f} "
                        f"{m['num_correct']:>5}/{m['num_problems']}")
    logger.info("#" * 80)


@app.local_entrypoint()
def main(
    eval_only: bool = False,
    checkpoint_step: int = 50,
    num_eval_problems: int = 100,
    skip_base: bool = False,
    only_base: bool = False,
    lr: float = 2e-6,
    ema_rate: float = 0.005,
    num_rollouts: int = 2,
    max_sequential_attempts: int = 16,
    sdpo_batch_size: int = 1,
    temperature: float = 0.7,
    max_problems: int = 647,
    ref_kl_beta: float = 0.0,
    alpha: float = 0.5,
    distillation_topk: int = 50,
    max_new_tokens: int = 1536,
    checkpoint_every: int = 50,
    feedback_truncate: int = 300,
    feedback_window: int = 6,
):
    """7B SDPO v2 — 16 attempts, full decision logging, 100 eval problems.

    Usage:
      modal run modal_app_7b_v2.py
      modal run modal_app_7b_v2.py --lr 1e-6 --max-sequential-attempts 16
      modal run modal_app_7b_v2.py --eval-only --checkpoint-step 100
    """
    if eval_only:
        print(f"Launching 7B v2 EVAL-ONLY on H100...")
        print(f"  checkpoint_step={checkpoint_step}, num_eval_problems={num_eval_problems}, only_base={only_base}")
        result = run_eval_only.remote(
            checkpoint_step=checkpoint_step,
            num_eval_problems=num_eval_problems,
            skip_base=skip_base,
            only_base=only_base,
        )
    else:
        print("Launching 7B v2 TRAINING + EVAL on H100...")
        print(f"  model=Qwen2.5-7B-Instruct")
        print(f"  lr={lr}, ema_rate={ema_rate}, temperature={temperature}")
        print(f"  num_rollouts={num_rollouts} x max_attempts={max_sequential_attempts}")
        print(f"  sdpo_batch_size={sdpo_batch_size}")
        print(f"  ref_kl_beta={ref_kl_beta}, alpha={alpha}, topk={distillation_topk}")
        print(f"  max_new_tokens={max_new_tokens}, max_problems={max_problems}")
        print(f"  feedback_truncate={feedback_truncate}, feedback_window={feedback_window}")
        print(f"  num_eval_problems={num_eval_problems}")
        print(f"  UPDATE CONDITION: 100% accuracy only")
        print()
        result = run_training_and_eval.remote(
            lr=lr, ema_rate=ema_rate,
            num_rollouts=num_rollouts,
            max_sequential_attempts=max_sequential_attempts,
            sdpo_batch_size=sdpo_batch_size,
            temperature=temperature,
            max_problems=max_problems,
            num_eval_problems=num_eval_problems,
            ref_kl_beta=ref_kl_beta,
            alpha=alpha,
            distillation_topk=distillation_topk,
            max_new_tokens=max_new_tokens,
            checkpoint_every=checkpoint_every,
            feedback_truncate=feedback_truncate,
            feedback_window=feedback_window,
        )

    print("\n" + "=" * 80)
    print("7B v2 FINAL RESULTS (16 attempts)")
    print("=" * 80)
    print(f"  {'Config':<25} {'Test-Case Acc':>14} {'Solve Rate':>12} {'Correct':>10}")
    print("-" * 80)
    for key in ["model+rag+sdpo", "base_qwen7b", "model+sdpo"]:
        m = result.get(key, {})
        if "avg_score" in m:
            print(f"  {m['label']:<25} {m['avg_score']:>14.4f} "
                  f"{m['pass_rate']:>12.4f} "
                  f"{m['num_correct']:>5}/{m['num_problems']}")
    print("=" * 80)
    print("Test-Case Acc = avg fraction of test cases passed per problem")
    print("Solve Rate    = fraction of problems fully solved (100% test cases)")
