"""
Compute training statistics for Methods section 2.5.

Reads the Medium configuration's published model config and environment
JSON (the model Section 3.2 uses as its primary pretraining-convergence
narrative — see results/final/1-train-mlm/medium/); reports all
hyperparameters, convergence trajectory, and hardware details.

Reads from the results/final/ mirror using stable, generic filenames (no
run-ID in the path), so this keeps working across retraining runs without
edits.

Output: paper/training_stats.json
Run from project root: python paper/training_stats.py
"""

import json
import math
from pathlib import Path

MIRROR_DIR = Path("results/final/1-train-mlm/medium")
RUN_JSON = MIRROR_DIR / "config.json"
ENV_JSON = MIRROR_DIR / "env.json"
OUTPUT_JSON = Path("paper/training_stats.json")

# ── Load ───────────────────────────────────────────────────────────────────
if not RUN_JSON.exists() or not ENV_JSON.exists():
    raise FileNotFoundError(
        f"{RUN_JSON} / {ENV_JSON} not found — has train_mlm.sbatch been run "
        f"for MODEL_SIZE=medium?"
    )
run = json.loads(RUN_JSON.read_text())
env = json.loads(ENV_JSON.read_text())

mc = run["model_config"]
tc = run["training_config"]
h  = run["history"]

# ── Derived training schedule values ──────────────────────────────────────
N_TRAIN        = 325_576
N_VAL          = 36_175
steps_per_epoch = math.ceil(N_TRAIN / tc["batch_size"])
total_steps    = steps_per_epoch * tc["num_epochs"]
warmup_pct     = tc["warmup_steps"] / total_steps * 100

# ── Convergence at key epochs ──────────────────────────────────────────────
key_epochs = [1, 5, 10, 25, 50, 75, 100]
convergence = []
for ep in key_epochs:
    i = ep - 1
    convergence.append({
        "epoch": ep,
        "train_loss": round(h["train_losses"][i], 4),
        "val_loss":   round(h["val_losses"][i], 4),
        "val_top1_acc_pct":     round(h["val_top1_accuracy"][i] * 100, 2),
        "content_top1_acc_pct": round(h["val_content_top1_accuracy"][i] * 100, 2),
    })

# Final 10-epoch plateau check
delta_val_last10 = h["val_losses"][-1] - h["val_losses"][-11]
rel_improvement_pct = abs(delta_val_last10) / h["val_losses"][-11] * 100

results = {
    "hyperparameters": {
        "optimizer": "AdamW",
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.01,
        "learning_rate": tc["learning_rate"],
        "lr_schedule": "linear warmup then cosine annealing to 0",
        "warmup_steps": tc["warmup_steps"],
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "warmup_pct_of_training": round(warmup_pct, 2),
        "epochs": tc["num_epochs"],
        "batch_size": tc["batch_size"],
        "max_seq_len": mc["max_seq_len"],
        "gradient_clip_max_norm": tc["max_grad_norm"],
        "val_split": tc["val_split"],
        "n_train": N_TRAIN,
        "n_val": N_VAL,
    },
    "implementation": {
        "framework": f"PyTorch {env['torch_version']}",
        "amp_dtype": "bfloat16",
        "torch_compile": True,
        "num_dataloader_workers": tc["num_workers"],
    },
    "hardware": {
        "gpu": env["gpu_name"],
        "gpu_memory_gb": env["gpu_memory_gb"],
        "n_gpus": env["gpu_count"],
        "cpu": env["cpu_model"],
        "ram_gb": env["ram_gb"],
        "cuda_version": env["cuda_version"],
        "python_version": env["python_version"],
    },
    "convergence": {
        "trajectory": convergence,
        "val_loss_change_final_10_epochs": round(delta_val_last10, 4),
        "val_loss_relative_improvement_final_10_pct": round(rel_improvement_pct, 1),
        "final_train_loss": round(h["train_losses"][-1], 4),
        "final_val_loss":   round(h["val_losses"][-1], 4),
        "final_val_top1_acc_pct":     round(h["val_top1_accuracy"][-1] * 100, 2),
        "final_val_top5_acc_pct":     round(h["val_top5_accuracy"][-1] * 100, 2),
        "final_content_top1_acc_pct": round(h["val_content_top1_accuracy"][-1] * 100, 2),
        "final_content_top5_acc_pct": round(h["val_content_top5_accuracy"][-1] * 100, 2),
        "training_time_h": round(h["training_time_s"] / 3600, 2),
    },
}

# ── Print summary ──────────────────────────────────────────────────────────
hp = results["hyperparameters"]
im = results["implementation"]
hw = results["hardware"]
cv = results["convergence"]

print("=== HYPERPARAMETERS ===")
print(f"Optimizer      : {hp['optimizer']}  (β1={hp['beta1']}, β2={hp['beta2']}, ε={hp['eps']}, wd={hp['weight_decay']})")
print(f"Learning rate  : {hp['learning_rate']}")
print(f"LR schedule    : {hp['lr_schedule']}")
print(f"Warmup steps   : {hp['warmup_steps']:,}  ({hp['warmup_pct_of_training']:.2f}% of {hp['total_steps']:,} total)")
print(f"Batch size     : {hp['batch_size']}")
print(f"Epochs         : {hp['epochs']}  ({hp['steps_per_epoch']:,} steps/epoch)")
print(f"MAX_LENGTH     : {hp['max_seq_len']}")
print(f"Grad clip      : max norm {hp['gradient_clip_max_norm']}")
print(f"Train / Val    : {hp['n_train']:,} / {hp['n_val']:,}  ({hp['val_split']*100:.0f}% val)")

print("\n=== IMPLEMENTATION ===")
print(f"Framework      : {im['framework']}")
print(f"AMP dtype      : {im['amp_dtype']}")
print(f"torch.compile  : {im['torch_compile']}")
print(f"DataLoader workers: {im['num_dataloader_workers']}")

print("\n=== HARDWARE ===")
print(f"GPU            : {hw['gpu']}  ({hw['gpu_memory_gb']} GB, ×{hw['n_gpus']})")
print(f"CUDA version   : {hw['cuda_version']}")
print(f"CPU            : {hw['cpu']}")

print("\n=== CONVERGENCE TRAJECTORY ===")
print(f"{'Epoch':>6}  {'Train loss':>11}  {'Val loss':>9}  {'Val top-1':>10}  {'Content top-1':>14}")
for row in cv["trajectory"]:
    print(f"{row['epoch']:>6}  {row['train_loss']:>11.4f}  {row['val_loss']:>9.4f}"
          f"  {row['val_top1_acc_pct']:>9.2f}%  {row['content_top1_acc_pct']:>13.2f}%")

print(f"\nFinal 10-epoch val loss improvement: {cv['val_loss_change_final_10_epochs']:.4f}"
      f"  ({cv['val_loss_relative_improvement_final_10_pct']:.1f}% relative)")
print(f"Training time  : {cv['training_time_h']:.2f} h")

# ── Save ───────────────────────────────────────────────────────────────────
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUTPUT_JSON}")