"""Collect hardware and software environment info and save to JSON.

Called from the SLURM job so environment info is captured on the compute
node (where the GPU is) rather than on the machine generating the report.

Usage
-----
    python scripts/collect_env.py --output models/smarts_transformer_<RUN_ID>_env.json
"""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def collect_env() -> dict:
    env: dict = {}

    env["python_version"] = sys.version.split()[0]
    env["platform"] = platform.platform()

    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
        else:
            env["cuda_version"] = None
            env["gpu_count"] = 0
            env["gpu_name"] = None
            env["gpu_memory_gb"] = None
    except ImportError:
        env["torch_version"] = "N/A"
        env["cuda_available"] = False

    try:
        cpu_lines = Path("/proc/cpuinfo").read_text().splitlines()
        model_lines = [line for line in cpu_lines if line.startswith("model name")]
        env["cpu_model"] = model_lines[0].split(": ", 1)[1] if model_lines else "N/A"
        env["cpu_cores"] = len(
            [line for line in cpu_lines if line.startswith("processor")]
        )
    except Exception:
        env["cpu_model"] = platform.processor() or "N/A"
        env["cpu_cores"] = None

    try:
        mem_lines = Path("/proc/meminfo").read_text().splitlines()
        total_kb = int(
            [line for line in mem_lines if line.startswith("MemTotal")][0].split()[1]
        )
        env["ram_gb"] = round(total_kb / 1024**2, 1)
    except Exception:
        env["ram_gb"] = None

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        env["driver_version"] = out.splitlines()[0] if out else None
    except Exception:
        env["driver_version"] = None

    for pkg in ["numpy", "pandas", "scikit-learn", "sentencepiece", "rdkit"]:
        try:
            env[f"pkg_{pkg.replace('-', '_')}"] = importlib.metadata.version(pkg)
        except Exception:
            env[f"pkg_{pkg.replace('-', '_')}"] = "N/A"

    return env


def main() -> None:
    p = argparse.ArgumentParser(description="Collect environment info to JSON.")
    p.add_argument("--output", required=True, help="Output JSON path.")
    args = p.parse_args()

    env = collect_env()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2))
    print(f"Environment info saved to {out}")


if __name__ == "__main__":
    main()
