"""Modal app: runs the full self-evolving agent training and evaluation on A100 GPU.

A100-80GB optimizations:
- Flash Attention 2 for 2-3x faster attention
- TF32 matmul for faster linear layers
- No CUDA_LAUNCH_BLOCKING (was serializing all GPU ops)
- num_rollouts=8, sdpo_batch_size=8 (fits comfortably in 80GB)
- Model pre-downloaded in image for faster cold starts
"""
import modal

app = modal.App("cs224n-self-evolving-agent")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
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
    )
    # Pre-download models into image so cold starts are fast
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

# Persistent volume for checkpoints and results
volume = modal.Volume.from_name("cs224n-results", create_if_missing=True)
VOLUME_PATH = "/results"


def _setup_env():
    """Common environment setup for all Modal functions."""
    import logging
    import os
    import sys
    import torch

    os.chdir("/root/cs224n")
    sys.path.insert(0, "/root/cs224n")
    sys.path.insert(0, "/root/cs224n/SDPO")

    # A100 performance flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    os.makedirs("checkpoints", exist_ok=True)

    return torch


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600 * 12,
    volumes={VOLUME_PATH: volume},
    memory=32768,
)
def run_training_and_eval():
    """Run full training loop (70% of LiveCodeBench) then evaluation."""
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
            logging.FileHandler(f"{VOLUME_PATH}/training.log"),
        ],
        force=True,
    )
    logger = logging.getLogger("modal_app")

    logger.info(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory
        logger.info(f"GPU memory: {mem / 1e9:.1f} GB")
        logger.info(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")

    # ---- PHASE 1: Training ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: TRAINING")
    logger.info("=" * 70)

    from agent.config import AgentConfig
    from main import run_training_loop

    # A100-80GB optimized config
    config = AgentConfig(
        max_problems=647,       # 70% of 924
        num_rollouts=8,         # 8 rollouts → better chance of finding demonstrations
        sdpo_batch_size=8,      # 8 problems per gradient step → stable gradients
        temperature=0.7,        # Diverse rollouts
        lr=1e-5,
        max_new_tokens=2048,
        checkpoint_every=50,
    )
    logger.info(f"Config: rollouts={config.num_rollouts}, batch={config.sdpo_batch_size}")

    def checkpoint_callback(ckpt_path, rag_path, step, metrics):
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
        volume.commit()
        logger.info(f"Checkpoint persisted: {dst}")

    all_metrics, agent_model, rag_db = run_training_loop(
        config, checkpoint_callback=checkpoint_callback
    )

    # Save final RAG
    rag_db_path = "checkpoints/rag_db.json"
    with open(rag_db_path, "w") as f:
        json.dump(rag_db.texts, f, indent=2)

    if os.path.exists("training_metrics.json"):
        shutil.copy("training_metrics.json", f"{VOLUME_PATH}/training_metrics.json")
    shutil.copy(rag_db_path, f"{VOLUME_PATH}/rag_db.json")
    volume.commit()

    del agent_model
    torch.cuda.empty_cache()

    # ---- PHASE 2: Evaluation ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: EVALUATION")
    logger.info("=" * 70)

    from eval import run_eval

    latest_ckpt = "checkpoints/final_model"
    if not os.path.exists(latest_ckpt):
        latest_ckpt = "checkpoints/checkpoint_step647"

    eval_results = run_eval(
        trained_model_path=latest_ckpt,
        rag_db_path=rag_db_path,
        config=config,
    )

    if os.path.exists("eval_results.json"):
        shutil.copy("eval_results.json", f"{VOLUME_PATH}/eval_results.json")
    if os.path.exists("eval.log"):
        shutil.copy("eval.log", f"{VOLUME_PATH}/eval.log")
    volume.commit()

    _print_results(logger, eval_results)
    return eval_results


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600 * 4,
    volumes={VOLUME_PATH: volume},
    memory=32768,
)
def run_eval_only(checkpoint_step: int = None, num_eval_problems: int = 100,
                  only_base_and_full: bool = False):
    """Run evaluation from a persisted checkpoint."""
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
    logger = logging.getLogger("modal_eval")

    volume.reload()

    step = checkpoint_step or 50
    vol_model = f"{VOLUME_PATH}/checkpoints/step_{step}/model"
    vol_rag = f"{VOLUME_PATH}/checkpoints/step_{step}/rag_db.json"

    local_model = "checkpoints/eval_model"
    local_rag = "checkpoints/eval_rag.json"

    if os.path.exists(vol_model):
        shutil.copytree(vol_model, local_model, dirs_exist_ok=True)
        logger.info(f"Loaded model: {vol_model}")
    else:
        logger.error(f"Model not found: {vol_model}")
        return {"error": f"Model not found at {vol_model}"}

    if os.path.exists(vol_rag):
        shutil.copy(vol_rag, local_rag)

    from eval import run_eval
    from agent.config import AgentConfig

    config = AgentConfig(max_problems=num_eval_problems)
    eval_results = run_eval(
        trained_model_path=local_model,
        rag_db_path=local_rag,
        config=config,
        only_base_and_full=only_base_and_full,
    )

    with open(f"{VOLUME_PATH}/eval_step{step}.json", "w") as f:
        json.dump(eval_results, f, indent=2, default=str)
    volume.commit()

    _print_results(logger, eval_results)
    return eval_results


def _print_results(logger, eval_results):
    logger.info("\n" + "#" * 70)
    logger.info("# RESULTS")
    logger.info("#" * 70)
    for key in ["base_qwen", "model+sdpo", "model+rag", "model+rag+sdpo"]:
        m = eval_results.get(key, {})
        if "avg_score" in m:
            logger.info(f"  {m['label']:<25} Score={m['avg_score']:.4f}  "
                        f"PassRate={m['pass_rate']:.4f}  "
                        f"Correct={m['num_correct']}/{m['num_problems']}")
    logger.info("#" * 70)


@app.local_entrypoint()
def main():
    """Kicks off training + eval on Modal A100."""
    print("Launching on Modal A100-80GB...")
    print("  num_rollouts=8, sdpo_batch_size=8, flash_attention_2, tf32")
    print("  Checkpoints every 50 problems → Modal volume 'cs224n-results'")
    print("  Check Modal dashboard for live logs.")
    print()
    result = run_training_and_eval.remote()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for key in ["base_qwen", "model+sdpo", "model+rag", "model+rag+sdpo"]:
        m = result.get(key, {})
        if "avg_score" in m:
            print(f"  {m['label']:<25} Score={m['avg_score']:.4f}  "
                  f"PassRate={m['pass_rate']:.4f}  "
                  f"Correct={m['num_correct']}/{m['num_problems']}")
    print("=" * 70)
