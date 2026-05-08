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
    return subprocess.run(cmd, shell=shell, check=check, text=True, capture_output=True, cwd=cwd)

def run_ssh(remote_uri, command, check=True):
    cmd = ["ssh", remote_uri, command]
    try:
        return subprocess.run(cmd, check=check, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"SSH: {command[:100]}...")
        print(f"SSH Command failed with exit code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise

def scp(src, dest, recursive=False, check=True):
    flags = ["-r"] if recursive else []
    cmd   = ["scp"] + flags + [str(src), str(dest)]
    try:
        return subprocess.run(cmd, check=check, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"SCP failed: {src} → {dest}")
        print(f"  exit code: {e.returncode}")
        print(f"  stderr:    {e.stderr.strip()}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Deploy and run inference server on supercomputer")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--upload-db", action="store_true",
        help="Force re-upload of vehicles.db and uploads/ directory",
    )
    args = parser.parse_args()

    local_dir = Path(__file__).parent.absolute()
    config    = load_config(local_dir / args.config)

    remote_user = config["remote"]["user"]
    remote_host = config["remote"]["host"]
    remote_dir  = config["remote"]["dir"]
    job_script  = config["job"]["script"]

    remote_uri        = f"{remote_user}@{remote_host}"
    remote_results_dir = f"{remote_dir}/results"

    project_dir      = local_dir.parent
    local_results_dir = project_dir / "results"

    local_db_path      = project_dir / "vehicles.db"
    local_uploads_dir  = project_dir / "uploads"
    local_server_dir   = local_dir / "server"
    local_cameras_json = project_dir / "traffic_cameras.json"

    print(f"Using local project:  {project_dir}")
    print(f"Using remote target:  {remote_uri}:{remote_dir}")

    try:
        run_ssh(remote_uri, f"mkdir -p '{remote_dir}/results' '{remote_dir}/server'")
        local_results_dir.mkdir(parents=True, exist_ok=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to set up remote directories: {e.stderr}")
        sys.exit(1)


    venv_path = f"{remote_dir}/.venv"
    setup_venv_cmd = (
        f"if [ ! -d '{venv_path}' ]; then "
        f"  echo 'Creating venv...'; python3 -m venv '{venv_path}'; "
        f"fi"
    )
    run_ssh(remote_uri, setup_venv_cmd)

    db_missing_result = run_ssh(
        remote_uri,
        f"[ -f '{remote_dir}/vehicles.db' ] && echo 'EXISTS' || echo 'MISSING'",
        check=False,
    )
    db_missing = "MISSING" in db_missing_result.stdout

    if args.upload_db or db_missing:
        # vehicles.db
        if local_db_path.is_file():
            print(f"  Uploading vehicles.db ({local_db_path.stat().st_size / 1024:.1f} KB)...")
            scp(local_db_path, f"{remote_uri}:{remote_dir}/vehicles.db")
        else:
            print("  WARNING: vehicles.db not found locally — skipping.")

        # uploads/
        if local_uploads_dir.is_dir():
            n_files = sum(1 for _ in local_uploads_dir.rglob("*") if _.is_file())
            print(f"  Uploading uploads/ ({n_files} files)...")
            scp(local_uploads_dir, f"{remote_uri}:{remote_dir}/", recursive=True)
            print("  uploads/ transferred.")
        else:
            print("  uploads/ directory not found locally — skipping.")
    else:
        print("  vehicles.db already present remotely. (Pass --upload-db to force re-upload.)")

    try:
        # core files
        files_to_copy = [
            local_dir / "detector.py",
            local_dir / job_script,
            project_dir / "requirements.txt",
        ]
        if local_cameras_json.is_file():
            files_to_copy.append(local_cameras_json)
        else:
            print("  WARNING: traffic_cameras.json not found — skipping.")

        run_cmd(["scp"] + [str(f) for f in files_to_copy if f.is_file()]
                + [f"{remote_uri}:{remote_dir}/"])

        if local_server_dir.is_dir():
            print("  Uploading server/ directory...")
            scp(local_server_dir, f"{remote_uri}:{remote_dir}/", recursive=True)
        else:
            print("  WARNING: server/ directory not found locally — skipping.")


        print("  Upgrading pip...")
        run_ssh(remote_uri,
                f"source '{venv_path}/bin/activate' && pip install --upgrade pip -q")

        print("  Installing Python requirements...")
        run_ssh(
            remote_uri,
            f"cd '{remote_dir}' && source '{venv_path}/bin/activate' && "
            f"pip install -r requirements.txt "
            f"--extra-index-url https://download.pytorch.org/whl/cu121 -q",
        )

    except subprocess.CalledProcessError as e:
        print(f"Command failed (exit {e.returncode}):\n{e.stdout}\n{e.stderr}")
        sys.exit(1)

    for weight_file in ["yolov8s.pt", "veri_emb_rn50.pt"]:
        local_wt = project_dir / weight_file
        remote_check = run_ssh(
            remote_uri,
            f"[ -f '{remote_dir}/{weight_file}' ] && echo 'EXISTS' || echo 'MISSING'",
            check=False,
        )
        if "MISSING" in remote_check.stdout:
            if local_wt.is_file():
                print(f"  Uploading {weight_file} ({local_wt.stat().st_size / 1e6:.0f} MB)...")
                scp(local_wt, f"{remote_uri}:{remote_dir}/{weight_file}")
                print(f"  ✓ {weight_file} transferred.")
            else:
                print(f"  WARNING: {weight_file} not found locally — skipping.")
        else:
            print(f"  {weight_file} already present remotely.")

    submit_cmd = (
        f"cd '{remote_dir}' && "
        f"sbatch --parsable --chdir='{remote_dir}' "
        f"--output='{remote_results_dir}/slurm-%j.out' "
        f"--error='{remote_results_dir}/slurm-%j.err' "
        f"--export=ALL,RUN_WORKDIR='{remote_dir}',VENV_DIR='{venv_path}' "
        f"'{job_script}'"
    )

    result = run_ssh(remote_uri, submit_cmd)
    job_id = result.stdout.strip().split(";")[0]

    if not job_id.isdigit():
        print(f"Failed to parse Slurm Job ID from output:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)

    print(f"  Slurm Job ID : {job_id}")
    print(f"  stdout log   : {remote_results_dir}/slurm-{job_id}.out")
    print(f"  stderr log   : {remote_results_dir}/slurm-{job_id}.err")


if __name__ == "__main__":
    main()