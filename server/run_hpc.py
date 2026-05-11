import subprocess
import argparse
import yaml
import sys
import time
from pathlib import Path


def load_config(config_path: Path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
    cmd = ["scp"] + flags + [str(src), str(dest)]
    try:
        return subprocess.run(cmd, check=check, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"SCP failed: {src} -> {dest}")
        print(f"  exit code: {e.returncode}")
        print(f"  stderr:    {e.stderr.strip()}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Deploy and run inference server on supercomputer")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    local_dir = Path(__file__).parent.absolute()
    project_dir = local_dir.parent
    config = load_config(local_dir / args.config)

    remote_user = config["remote"]["user"]
    remote_host = config["remote"]["host"]
    remote_dir = config["remote"]["dir"]
    job_script = config["job"]["script"]

    remote_uri = f"{remote_user}@{remote_host}"

    print(f"Using local server dir:  {local_dir}")
    print(f"Using remote target:     {remote_uri}:{remote_dir}")

    try:
        run_ssh(remote_uri, f"mkdir -p '{remote_dir}'")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set up remote directory: {e.stderr}")
        sys.exit(1)

    venv_path = f"{remote_dir}/.venv"
    setup_venv_cmd = (
        f"if [ ! -d '{venv_path}' ]; then "
        f"  echo 'Creating venv...'; python3 -m venv '{venv_path}'; "
        f"fi"
    )
    run_ssh(remote_uri, setup_venv_cmd)

    try:
        if (project_dir / "requirements.txt").exists():
            print("  Uploading project requirements.txt...")
            scp(project_dir / "requirements.txt", f"{remote_uri}:{remote_dir}/")

        print("  Uploading server/ directory...")
        scp(local_dir, f"{remote_uri}:{remote_dir}/", recursive=True)

        print("  Upgrading pip...")
        run_ssh(
            remote_uri,
            f"source '{venv_path}/bin/activate' && pip install --upgrade pip -q",
        )

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

    remote_job_script = f"{remote_dir}/server/{job_script}"
    submit_cmd = (
        f"cd '{remote_dir}' && "
        f"sbatch --parsable --chdir='{remote_dir}' "
        f"--output='{remote_dir}/slurm-%j.out' "
        f"--error='{remote_dir}/slurm-%j.err' "
        f"--export=ALL,RUN_WORKDIR='{remote_dir}',VENV_DIR='{venv_path}' "
        f"'{remote_job_script}'"
    )

    print(f"  Submitting job: {job_script} ...")
    result = run_ssh(remote_uri, submit_cmd)
    job_id = result.stdout.strip().split(";")[0]

    if not job_id.isdigit():
        print(f"Failed to parse Slurm Job ID from output:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)

    print(f"  Slurm Job ID : {job_id}")
    print(f"  stdout log   : {remote_dir}/slurm-{job_id}.out")
    print(f"  stderr log   : {remote_dir}/slurm-{job_id}.err")
    # Poll Slurm for job state until it starts running (or finishes)
    last_state = None
    print("  Polling job state until RUNNING...")
    while True:
        # Try squeue first (shows jobs in queue or running)
        res = run_ssh(remote_uri, f"squeue -j {job_id} -h -o %T", check=False)
        state = ""
        if getattr(res, 'returncode', 1) == 0 and res.stdout.strip():
            state = res.stdout.strip().splitlines()[0].strip()
        else:
            # Fall back to sacct for completed/finished states
            res2 = run_ssh(remote_uri, f"sacct -j {job_id} --format=State --noheader -n", check=False)
            if getattr(res2, 'returncode', 1) == 0 and res2.stdout.strip():
                for line in res2.stdout.splitlines():
                    s = line.strip()
                    if s:
                        state = s.split()[0]
                        break

        if state and state != last_state:
            print(f"  Slurm state: {state}")
            last_state = state

        if state.startswith("RUNNING"):
            print("  Job is now RUNNING. Stopping polling.")
            break

        if state.startswith("COMPLETED") or state.startswith("FAILED") or state.startswith("CANCELLED") or state.startswith("TIMEOUT") or state.startswith("NODE_FAIL"):
            print(f"  Job finished with state {state}. Stopping polling.")
            break

        time.sleep(5)

    print("press q to cancel job")
    c = False
    while c is False:
        c = input()
        if c.lower() == "q":
            print(f"Cancelling job {job_id}...")
            run_ssh(remote_uri, f"scancel {job_id}")
            print("Cancelled.")
            break


if __name__ == "__main__":
    main()