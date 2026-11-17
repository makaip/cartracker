import subprocess
import argparse
import yaml
import sys
import time
import re
from pathlib import Path


SSH_COMMON_OPTIONS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=~/.ssh/known_hosts",
]


def load_config(config_path: Path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_ssh(remote_uri, command, check=True):
    cmd = ["ssh", *SSH_COMMON_OPTIONS, remote_uri, command]
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
    cmd = ["scp", *SSH_COMMON_OPTIONS] + flags + [str(src), str(dest)]
    try:
        return subprocess.run(cmd, check=check, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"SCP failed: {src} -> {dest}")
        print(f"  exit code: {e.returncode}")
        print(f"  stderr:    {e.stderr.strip()}")
        raise


def extract_node_from_slurm(output: str):
    match = re.search(r"(?:NodeList|BatchHost)=([^\s]+)", output)
    if match:
        node = match.group(1).strip()
        if node and node != "(null)":
            return node
    return None


def get_job_state_and_node(remote_uri, job_id):
    res = run_ssh(remote_uri, f"squeue -j {job_id} -h -o '%T|%N'", check=False)
    state = ""
    node = ""

    if getattr(res, "returncode", 1) == 0 and res.stdout.strip():
        first_line = res.stdout.strip().splitlines()[0].strip()
        parts = first_line.split("|", 1)
        state = parts[0].strip()
        if len(parts) > 1:
            node = parts[1].strip()

    if state.startswith("RUNNING") and (not node or node == "(null)"):
        res2 = run_ssh(remote_uri, f"scontrol show job {job_id} -o", check=False)
        if getattr(res2, "returncode", 1) == 0 and res2.stdout.strip():
            node = extract_node_from_slurm(res2.stdout) or ""

    return state, node


def wait_for_running(remote_uri, job_id):
    """poll Slurm until the job runs; returns its node, or None if it ended first"""
    last_state = None
    while True:
        state, node = get_job_state_and_node(remote_uri, job_id)

        if not state:
            # Fall back to sacct for completed/finished states
            res = run_ssh(remote_uri, f"sacct -j {job_id} --format=State --noheader -n", check=False)
            if getattr(res, 'returncode', 1) == 0 and res.stdout.strip():
                for line in res.stdout.splitlines():
                    s = line.strip()
                    if s:
                        state = s.split()[0]
                        break

        if state and state != last_state:
            print(f"  [{job_id}] Slurm state: {state}")
            last_state = state

        if state.startswith("RUNNING") and node:
            print(f"  [{job_id}] RUNNING on node: {node}")
            return node

        if state.startswith(("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL")):
            print(f"  [{job_id}] finished with state {state}. Stopping polling.")
            return None

        time.sleep(5)


def start_ssh_tunnel(local_port, remote_port, remote_user, remote_host, node):
    tunnel_cmd = [
        "ssh",
        *SSH_COMMON_OPTIONS,
        "-N",
        "-L",
        f"{local_port}:localhost:{remote_port}",
        "-J",
        f"{remote_user}@{remote_host}",
        f"{remote_user}@{node}",
    ]
    print("  Starting SSH tunnel:")
    print(f"    {' '.join(tunnel_cmd)}")
    return subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


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
    ws_port = config["server"]["ws_port"]
    num_nodes = max(1, int(config["job"].get("nodes", 1)))

    remote_uri = f"{remote_user}@{remote_host}"

    print(f"Using local server dir:  {local_dir}")
    print(f"Using remote target:     {remote_uri}:{remote_dir}")

    try:
        run_ssh(remote_uri, f"mkdir -p '{remote_dir}'")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set up remote directory: {e.stderr}")
        sys.exit(1)

    try:
        run_ssh(remote_uri, f"mkdir -p '{remote_dir}/.cache/ultralytics' '{remote_dir}/.cache/torch'")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create remote cache dirs: {e.stderr}")
        sys.exit(1)

    venv_path = f"{remote_dir}/.venv"
    # setup_venv_cmd = (
    #     f"if [ ! -d '{venv_path}' ]; then "
    #     f"  echo 'Creating venv...'; python3 -m venv '{venv_path}'; "
    #     f"fi"
    # )
    # run_ssh(remote_uri, setup_venv_cmd)

    try:
        if (project_dir / "requirements.txt").exists():
            print("  Uploading project requirements.txt...")
            scp(project_dir / "requirements.txt", f"{remote_uri}:{remote_dir}/")

        print("  Uploading server/ directory...")
        scp(local_dir, f"{remote_uri}:{remote_dir}/", recursive=True)

        # print("  Upgrading pip...")
        # run_ssh(
        #     remote_uri,
        #     f"source '{venv_path}/bin/activate' && pip install --upgrade pip -q",
        # )

        # print("  Installing Python requirements...")
        # run_ssh(
        #     remote_uri,
        #     f"cd '{remote_dir}' && source '{venv_path}/bin/activate' && "
        #     f"pip install -r requirements.txt "
        #     f"--extra-index-url https://download.pytorch.org/whl/cu121 -q",
        # )
        
    except subprocess.CalledProcessError as e:
        print(f"Command failed (exit {e.returncode}):\n{e.stdout}\n{e.stderr}")
        sys.exit(1)

    remote_job_script = f"{remote_dir}/server/{job_script}"
    remote_yolo = f"{remote_dir}/.cache/ultralytics"
    remote_torch = f"{remote_dir}/.cache/torch"
    # one job per node; node i handles every num_nodes-th camera and is reached
    # locally on port ws_port + i
    job_ids = []
    for shard in range(num_nodes):
        submit_cmd = (
            f"cd '{remote_dir}' && "
            f"sbatch --parsable --chdir='{remote_dir}' "
            f"--output='{remote_dir}/slurm-%j.out' "
            f"--error='{remote_dir}/slurm-%j.err' "
            f"--export=ALL,RUN_WORKDIR='{remote_dir}',VENV_DIR='{venv_path}',YOLO_CONFIG_DIR='{remote_yolo}',TORCH_HOME='{remote_torch}',"
            f"SHARD_INDEX={shard},SHARD_COUNT={num_nodes} "
            f"'{remote_job_script}'"
        )

        print(f"  Submitting job {shard + 1}/{num_nodes}: {job_script} ...")
        result = run_ssh(remote_uri, submit_cmd)
        job_id = result.stdout.strip().split(";")[0]

        if not job_id.isdigit():
            print(f"Failed to parse Slurm Job ID from output:\n{result.stdout}\n{result.stderr}")
            for submitted in job_ids:
                run_ssh(remote_uri, f"scancel {submitted}", check=False)
            sys.exit(1)

        job_ids.append(job_id)
        print(f"  Slurm Job ID : {job_id}")
        print(f"  stdout log   : {remote_dir}/slurm-{job_id}.out")
        print(f"  stderr log   : {remote_dir}/slurm-{job_id}.err")

    print("  Polling job state until RUNNING...")
    tunnels = []
    for shard, job_id in enumerate(job_ids):
        remote_node = wait_for_running(remote_uri, job_id)
        if remote_node:
            local_port = ws_port + shard
            tunnels.append(start_ssh_tunnel(local_port, ws_port, remote_user, remote_host, remote_node))
            print(f"  Node {shard + 1}/{num_nodes}: http://localhost:{local_port}")
        else:
            print(f"  [{job_id}] No running node found; SSH tunnel not started.")

    print("press q to cancel job")
    while True:
        for tunnel_proc in list(tunnels):
            if tunnel_proc.poll() is not None:
                print(f"SSH tunnel exited with code {tunnel_proc.returncode}.")
                tunnels.remove(tunnel_proc)
        c = input()
        if c.lower() == "q":
            for tunnel_proc in tunnels:
                if tunnel_proc.poll() is None:
                    print("Stopping SSH tunnel...")
                    tunnel_proc.terminate()
                    try:
                        tunnel_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        tunnel_proc.kill()
                        tunnel_proc.wait(timeout=5)
            print(f"Cancelling job(s) {' '.join(job_ids)}...")
            run_ssh(remote_uri, f"scancel {' '.join(job_ids)}")
            print("Cancelled.")
            break

if __name__ == "__main__":
    main()