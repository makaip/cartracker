import os
import subprocess
import time
import argparse
import yaml
import sys
from datetime import datetime
from pathlib import Path


def load_config(config_path: Path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_cmd(cmd, shell=False, check=True, cwd=None):
    cmd_strs = [str(c) for c in cmd] if isinstance(cmd, list) else cmd
    # print(f"Running: {' '.join(cmd_strs) if isinstance(cmd, list) else cmd}")
    return subprocess.run(cmd, shell=shell, check=check, text=True, capture_output=True, cwd=cwd)

def run_ssh(remote_uri, command, check=True):
    cmd = ["ssh", remote_uri, command]
    # print(f"SSH: {command[:100]}...")
    try:
        return subprocess.run(cmd, check=check, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"SSH: {command[:100]}...")
        print(f"SSH Command failed with exit code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Deploy and run training job on supercomputer")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--upload-datasets", action="store_true", help="Force re-upload of datasets (even if they exist)")
    args = parser.parse_args()

    local_dir = Path(__file__).parent.absolute()
    config = load_config(local_dir / args.config)
    
    remote_user = config["remote"]["user"]
    remote_host = config["remote"]["host"]
    remote_dir = config["remote"]["dir"]
    job_script = config["job"]["script"]
    poll_seconds = config["job"].get("poll_seconds", 10)
    
    remote_uri = f"{remote_user}@{remote_host}"
    
    remote_results_dir = f"{remote_dir}/results"
    
    project_dir = local_dir.parent
    local_results_dir = project_dir / "results"

    # print(f"Using local project: {project_dir}")
    # print(f"Using remote target: {remote_uri}:{remote_dir}")

    print("\n[1/7] Setting up directories...")
    try:
        run_ssh(remote_uri, f"mkdir -p '{remote_dir}/results'")
        local_results_dir.mkdir(parents=True, exist_ok=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to setup remote directories. Error: {e.stderr}")
        sys.exit(1)

    print("\n[2/7] Verifying/Setting up remote virtual environment...")
    venv_path = f"{remote_dir}/.venv"
    setup_venv_cmd = (
        f"if [ ! -d '{venv_path}' ]; then "
        f"  echo 'Creating venv...'; python3 -m venv '{venv_path}'; "
        f"fi"
    )

    run_ssh(remote_uri, setup_venv_cmd)

    print("\n[3/7] Setting up datasets...")
    check_datasets = run_ssh(remote_uri, f"[ -d '{remote_dir}/datasets' ] && echo 'EXISTS' || echo 'MISSING'", check=False)
    if args.upload_datasets or "MISSING" in check_datasets.stdout:
        run_cmd(["scp", "-r", str(project_dir / "datasets"), f"{remote_uri}:{remote_dir}/"])
    else:
        print("Datasets already synced. Skipping upload. (Use --upload-datasets to force re-upload)")

    print("\n[4/7] Transferring codebase to cluster...")    
    try:
        run_cmd(["scp", str(local_dir / "train.py"), str(local_dir / job_script), str(project_dir / "requirements.txt"), f"{remote_uri}:{remote_dir}/"])
        
        # print("Upgrading pip...")
        # run_ssh(remote_uri, f"source '{venv_path}/bin/activate' && pip install --upgrade pip")
        
        # print("Installing python requirements...")
        # run_ssh(remote_uri, f"cd '{remote_dir}' && source '{venv_path}/bin/activate' && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121")

    except subprocess.CalledProcessError as e:
        print(f"Command failed. Exit code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        sys.exit(1)

    print("\n[5/7] Submitting job with sbatch...")
    submit_cmd = (
        f"cd '{remote_dir}' && "
        f"sbatch --parsable --chdir='{remote_dir}' "
        f"--output='{remote_results_dir}/slurm-%j.out' "
        f"--error='{remote_results_dir}/slurm-%j.err' "
        f"--export=ALL,RUN_WORKDIR='{remote_dir}',VENV_DIR='{venv_path}' '{job_script}'"
    )
    
    result = run_ssh(remote_uri, submit_cmd)
    job_id = result.stdout.strip().split(';')[0]
    
    if not job_id:
        print(f"Failed to parse Slurm Job ID. Output: {result.stderr or result.stdout}")
        sys.exit(1)

    print(f"Submitted Job ID: {job_id}")

    print("\n[6/7] Waiting for job to complete...")
    try:
        load = ["-", "\\", "|", "/"]
        prev_status = ""
        i = 0
        while True:
            status_result = run_ssh(remote_uri, f"squeue -j '{job_id}' -h -o '%T'", check=False)
            status = status_result.stdout.strip()

            if not status:
                break

            if status == prev_status:
                print(f"  Job {job_id} status: {status} \t {load[i % len(load)]} \r")
                prev_status = status
            else:
                print(f"  Job {job_id} status: {status}")
                prev_status = status

            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        print(f"\nPolling interrupted! Job {job_id} is still running on the cluster.")
        print(f"To cancel it, run: ssh {remote_uri} scancel {job_id}")
        sys.exit(1)

    state_result = run_ssh(remote_uri, f"sacct -j '{job_id}' --format=State --noheader 2>/dev/null | head -n 1 | xargs", check=False)
    final_state = state_result.stdout.strip() or "UNKNOWN"
    print(f"Job {job_id} finished with state: {final_state}")

    print("\n[7/7] Syncing results back to local machine...")
    
    try:
        local_results_dir.mkdir(parents=True, exist_ok=True)
        print("Using scp to sync remote results to local results/ directory...")
        run_cmd(["scp", "-r", f"{remote_uri}:{remote_dir}/results/", str(project_dir)])
    except Exception as e:
        print(f"Failed to sync results back: {e}")

    if not final_state.startswith("COMPLETED"):
        print(f"\nWarning: Slurm job state is {final_state}. Check logs in {local_results_dir}")

    print("\nDone.")

if __name__ == "__main__":
    main()
