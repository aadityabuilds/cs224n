"""Modal app: runs the full self-evolving agent training and evaluation on A100 GPU."""
import modal

app = modal.App("cs224n-self-evolving-agent")

# Docker image with all dependencies + local code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "datasets>=2.18.0",
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.0",
        "sentence-transformers>=2.2.0",
        "huggingface_hub",
    )
    .add_local_dir(".", remote_path="/root/cs224n", ignore=[
        ".venv",
        "__pycache__",
        ".git",
        "uv.lock",
        "*.pyc",
        "*.egg-info",
        "*.so",
    ])
)

# Persistent volume for checkpoints and results
volume = modal.Volume.from_name("cs224n-results", create_if_missing=True)

VOLUME_PATH = "/results"


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600 * 12,  # 12 hours for full 70% run
    volumes={VOLUME_PATH: volume},
    memory=32768,
)
def run_training_and_eval():
    """Run the full training loop (70% of dataset) with checkpoints every 50 problems, then eval."""
    import json
    import logging
    import os
    import shutil
    import sys

    os.chdir("/root/cs224n")
    sys.path.insert(0, "/root/cs224n")
    sys.path.insert(0, "/root/cs224n/SDPO")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    os.makedirs("checkpoints", exist_ok=True)
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

    import torch
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        logger.info(f"GPU memory: {mem / 1e9:.1f} GB")

    # ---- PHASE 1: Training ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: TRAINING (70% of LiveCodeBench)")
    logger.info("=" * 70)

    from agent.config import AgentConfig
    from main import run_training_loop

    # 70% of 924 = 647
    config = AgentConfig(
        max_problems=647,
    )

    def checkpoint_callback(ckpt_path, rag_path, step, metrics):
        """Save checkpoint to persistent Modal volume."""
        dst = f"{VOLUME_PATH}/checkpoints/step_{step}"
        os.makedirs(dst, exist_ok=True)
        # Copy model checkpoint
        if os.path.isdir(ckpt_path):
            shutil.copytree(ckpt_path, f"{dst}/model", dirs_exist_ok=True)
        # Copy RAG db
        if os.path.exists(rag_path):
            shutil.copy(rag_path, f"{dst}/rag_db.json")
        # Copy metrics
        with open(f"{dst}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        # Commit volume so data persists
        volume.commit()
        logger.info(f"Checkpoint persisted to volume: {dst}")

    all_metrics, agent_model, rag_db = run_training_loop(config, checkpoint_callback=checkpoint_callback)

    # Save final RAG database
    rag_db_path = "checkpoints/rag_db.json"
    with open(rag_db_path, "w") as f:
        json.dump(rag_db.texts, f, indent=2)
    logger.info(f"RAG database saved ({rag_db.size} chunks)")

    # Copy final artifacts to volume
    if os.path.exists("training_metrics.json"):
        shutil.copy("training_metrics.json", f"{VOLUME_PATH}/training_metrics.json")
    shutil.copy(rag_db_path, f"{VOLUME_PATH}/rag_db.json")
    volume.commit()

    # Free training model from memory
    del agent_model
    torch.cuda.empty_cache()

    # ---- PHASE 2: Evaluation ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: EVALUATION")
    logger.info("=" * 70)

    from eval import run_eval

    # Use the latest checkpoint for eval
    latest_ckpt = "checkpoints/final_model"
    if not os.path.exists(latest_ckpt):
        latest_ckpt = "checkpoints/checkpoint_step647"

    eval_results = run_eval(
        trained_model_path=latest_ckpt,
        rag_db_path=rag_db_path,
        config=config,
    )

    # Copy eval results to volume
    if os.path.exists("eval_results.json"):
        shutil.copy("eval_results.json", f"{VOLUME_PATH}/eval_results.json")
    if os.path.exists("eval.log"):
        shutil.copy("eval.log", f"{VOLUME_PATH}/eval.log")
    volume.commit()

    # Print final results
    logger.info("\n\n" + "#" * 70)
    logger.info("# FINAL RESULTS")
    logger.info("#" * 70)
    for key in ["base_qwen", "model+sdpo", "model+rag", "model+rag+sdpo"]:
        m = eval_results.get(key, {})
        if "avg_score" in m:
            logger.info(f"  {m['label']:<25} Score={m['avg_score']:.4f}  "
                        f"PassRate={m['pass_rate']:.4f}  "
                        f"Correct={m['num_correct']}/{m['num_problems']}")
    logger.info("#" * 70)

    return eval_results


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600 * 4,
    volumes={VOLUME_PATH: volume},
    memory=32768,
)
def run_eval_only(checkpoint_step: int = None, num_eval_problems: int = 50):
    """Run only evaluation from a persisted checkpoint on the volume."""
    import json
    import logging
    import os
    import shutil
    import sys

    os.chdir("/root/cs224n")
    sys.path.insert(0, "/root/cs224n")
    sys.path.insert(0, "/root/cs224n/SDPO")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger("modal_eval")

    # Reload volume to see latest data
    volume.reload()

    # Copy checkpoint from volume to local dir
    os.makedirs("checkpoints", exist_ok=True)
    if checkpoint_step:
        vol_model = f"{VOLUME_PATH}/checkpoints/step_{checkpoint_step}/model"
        vol_rag = f"{VOLUME_PATH}/checkpoints/step_{checkpoint_step}/rag_db.json"
    else:
        vol_model = f"{VOLUME_PATH}/checkpoints/step_50/model"
        vol_rag = f"{VOLUME_PATH}/checkpoints/step_50/rag_db.json"

    local_model = "checkpoints/eval_model"
    local_rag = "checkpoints/eval_rag.json"

    if os.path.exists(vol_model):
        shutil.copytree(vol_model, local_model, dirs_exist_ok=True)
        logger.info(f"Copied model from volume: {vol_model}")
    else:
        logger.error(f"Model not found at {vol_model}")
        return {"error": f"Model not found at {vol_model}"}

    if os.path.exists(vol_rag):
        shutil.copy(vol_rag, local_rag)
        logger.info(f"Copied RAG db from volume: {vol_rag}")

    from eval import run_eval
    from agent.config import AgentConfig

    config = AgentConfig(max_problems=num_eval_problems)

    eval_results = run_eval(
        trained_model_path=local_model,
        rag_db_path=local_rag,
        config=config,
    )

    # Save to volume
    with open(f"{VOLUME_PATH}/eval_step{checkpoint_step or 50}.json", "w") as f:
        json.dump(eval_results, f, indent=2, default=str)
    volume.commit()

    # Print summary
    logger.info("\n" + "#" * 70)
    logger.info("# EVAL RESULTS")
    logger.info("#" * 70)
    for key in ["base_qwen", "model+sdpo", "model+rag", "model+rag+sdpo"]:
        m = eval_results.get(key, {})
        if "avg_score" in m:
            logger.info(f"  {m['label']:<25} Score={m['avg_score']:.4f}  "
                        f"Acc={m['avg_accuracy']:.4f}  "
                        f"PassRate={m['pass_rate']:.4f}  "
                        f"Correct={m['num_correct']}/{m['num_problems']}")
    logger.info("#" * 70)

    return eval_results


@app.local_entrypoint()
def main():
    """Local entrypoint: kicks off training + eval on Modal GPU."""
    print("Launching training (70% of LiveCodeBench, ~647 problems) + evaluation on Modal A100...")
    print("Checkpoints saved every 50 problems to Modal volume 'cs224n-results'.")
    print("This will take several hours. Check Modal dashboard for live logs.")
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
