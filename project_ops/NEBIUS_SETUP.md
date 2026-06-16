# Nebius Cloud Setup Guide for HW3

> This guide exists because both `LEARNING_GUIDE.md` and `SOLUTIONS_REFERENCE.md` assume you already have a running VM with a public IP and SSH access. If you are starting from zero, start here. This covers everything from creating a Nebius account through having all five ports forwarded and the observability stack healthy in your laptop browser.

---

## Overview: What You Are Provisioning

For HW3 you need exactly one piece of cloud hardware: a single H100 80GB GPU VM. Everything else runs on that same VM:

| Service | Port | What it is |
|---------|------|------------|
| vLLM | 8000 | Serves Qwen3-30B-A3B-Instruct-2507 |
| Agent server (yours) | 8001 | LangGraph HTTP server |
| Langfuse | 3001 | Local trace store (Docker) |
| Prometheus | 9090 | Metrics scraper (Docker) |
| Grafana | 3000 | Dashboard (Docker) |

All five services bind to `localhost` on the VM. SSH port forwarding is how your laptop browser reaches them. The H100 is expensive (~$2–4/hr on-demand). Section 10 of this guide tells you when you do and do not need it running.

**Estimated setup time to first working SSH session:** 20–40 minutes for a new account.

---

## Step 1: Nebius Account and Console Access

### 1.1 Create an Account

1. Go to **[console.nebius.ai](https://console.nebius.ai)** and sign up.
2. You will be asked for a credit card during sign-up. This is required to access GPU quota — Nebius does not offer free GPU tiers.
3. After email verification and billing setup, you land on the main console dashboard.

### 1.2 Understanding the Console Structure

Nebius organizes resources into a hierarchy:

```
Account (your login)
  └── Tenant (organization)
        └── Project
              └── Resources (VMs, disks, networks, ...)
```

When you first sign up, a default project is created for you. You will provision your VM inside this project. You can verify which project is active by checking the project selector in the top bar of the console.

### 1.3 Console Navigation

The left sidebar in the Nebius console contains:

- **Compute Cloud** — where you create and manage VMs
- **Virtual Private Cloud (VPC)** — networks and subnets (you will use the default)
- **Object Storage** — not needed for HW3
- **IAM** — service accounts and SSH key management

The URL structure is `console.nebius.ai/<region>/<project-id>/<service>/...`. You will spend most of your time in the Compute Cloud section.

---

## Step 2: Request or Verify GPU Quota

### 2.1 Default Quota After Billing Setup

As of 2025, Nebius grants the following GPU quota automatically once you add billing details:

- Up to **16 H100 GPUs** accessible immediately via the console
- Up to **2 L40S GPUs** immediately available

This means most students can create a 1× H100 VM without contacting support. However, quotas are per-region and per-project. If you see an error like "quota exceeded" when creating your VM, follow the steps in 2.2.

### 2.2 How to Check Your Current Quota

In the Nebius console:
1. Navigate to your project settings (gear icon next to the project name in the top bar, or **IAM** → **Quotas** in the left sidebar).
2. Look for compute quotas under **GPU** resources for the `gpu-h100-sxm` platform.
3. If the quota shows 0 or is lower than 1, proceed to 2.3.

### 2.3 How to Request a Quota Increase

If you need more than the default quota:
1. Go to the [Nebius quota request form](https://nebius.com/contact) or use the in-console "Contact Support" button.
2. Specify: project ID, region, resource type (`gpu-h100-sxm`), requested quantity (1 GPU / 1 VM), and use-case (academic assignment / MLOps course).
3. Typical wait time for academic requests: **1–3 business days**.

### 2.4 What to Do While Waiting for Quota

The H100 is only needed for phases where real performance numbers matter. While waiting, you can complete:

| Phase | Can do without H100? | How |
|-------|---------------------|-----|
| Phase 0 (setup) | Yes — for everything except starting vLLM | Clone repo, set up Docker stack, configure port forwards |
| Phase 2 (Grafana) | Yes — with CPU vLLM | Use `Qwen/Qwen3-0.6B` on CPU; metrics are real, numbers are unrepresentative |
| Phase 3 (Agent) | Yes | Use any OpenAI-compatible API or CPU vLLM |
| Phase 4 (Tracing) | Yes | Langfuse captures spans regardless of backend |
| Phase 1, 5, 6 | No — real numbers required | Must be on H100 with the 30B model |

---

## Step 3: Create an H100 VM

### 3.1 Navigate to Compute Cloud

In the Nebius console left sidebar, click **Compute Cloud** → **Virtual Machines**. Click **Create VM** (the blue button in the top right).

### 3.2 Instance Type Selection

In the VM creation form:

**Platform**: Select `gpu-h100-sxm`
(This is the NVIDIA H100 SXM platform — 80GB HBM3, the one the assignment is designed for.)

**Preset**: Select `1gpu-16vcpu-200gb`
(This means: 1 H100 GPU, 16 vCPUs, 200 GiB RAM. This is the single-GPU H100 node. It is sufficient for serving Qwen3-30B-A3B-Instruct-2507 in FP8.)

> Do not select `8gpu-128vcpu-1600gb` unless you are experimenting with multi-GPU tensor parallelism. For this assignment, 1 GPU is the target hardware.

### 3.3 OS Image Selection

Under **Boot Disk** → **Image**:

Select **Ubuntu 22.04 LTS** with CUDA pre-installed. The image name in Nebius is typically `ubuntu22.04-cuda12.x` (the exact CUDA version number varies by what Nebius has packaged — pick the latest available Ubuntu 22.04 CUDA image).

Why Ubuntu 22.04 and not 24.04:
- vLLM's `torch.compile` path and some CUDA toolkit dependencies are better tested on 22.04 as of mid-2025.
- The `python3-dev` headers (required by vLLM) are straightforward to install on 22.04.
- Docker CE is well-supported on 22.04.

Why not use the base Ubuntu image without CUDA:
- The CUDA drivers take a long time to install correctly from scratch and are a common source of setup failures. The pre-packaged image saves 30–60 minutes.

### 3.4 Disk Size

Under **Boot Disk** → **Size (GiB)**:

Set to at least **200 GiB**. The default may be 50 GiB, which is not enough. Here is the breakdown:

| Item | Approximate size |
|------|-----------------|
| OS + CUDA drivers | ~10 GB |
| Docker images (Prometheus + Grafana + Langfuse + Postgres) | ~8–12 GB |
| Qwen3-30B-A3B-Instruct-2507 model weights (FP8) | ~35–40 GB |
| vLLM Python environment + dependencies | ~15–20 GB |
| BIRD data | ~2 GB |
| HuggingFace model cache overhead | ~5 GB |
| Safety margin | ~20 GB |
| **Total** | ~100–110 GB minimum, **200 GiB recommended** |

200 GiB gives you comfortable headroom to experiment without running into disk-full errors at 2am during an eval run.

**Disk type**: Leave at **Network SSD** (the default). There is no reason to use HDD for this workload.

### 3.5 Network Settings

Under **Network Interfaces**:

- **Subnet**: Use the default subnet in the default VPC. You do not need to create a custom network.
- **Public IP Address**: **Enable this.** Without a public IP, you cannot SSH into the VM from your laptop. Toggle the "Public IP" or "Public Access" option to on.

Leave all other network defaults as-is (default security group allows inbound SSH on port 22).

### 3.6 Cost Estimate

At the time of writing (2025–2026), Nebius H100 pricing:

| Pricing type | Approximate cost per hour |
|-------------|--------------------------|
| On-demand (no commitment) | ~$3.85/hr |
| Reserved (1-month commitment) | ~$2.15/hr |
| Preemptible (can be reclaimed) | ~$1.50–2.00/hr |

For this assignment, **on-demand is recommended** unless you already know you will use the VM continuously for a month. Preemptible instances are cheaper but can be interrupted, which is disruptive mid-eval-run.

**How to avoid unnecessary charges**: See Step 10. The key rule: stop the VM (not delete — stop) when you are not actively using it. A stopped VM does not charge for compute (only for the persistent disk, which is minimal).

---

## Step 4: SSH Key Setup

### 4.1 Generate an SSH Key Pair (if you do not already have one)

On your laptop:

```bash
ssh-keygen -t ed25519 -C "nebius-hw3"
```

When prompted for a file path, press Enter to accept the default (`~/.ssh/id_ed25519`). Set a passphrase or leave blank (blank is fine for a course VM).

This creates two files:
- `~/.ssh/id_ed25519` — your private key (never share this)
- `~/.ssh/id_ed25519.pub` — your public key (this goes into Nebius)

If you already have an SSH key (`~/.ssh/id_ed25519` or `~/.ssh/id_rsa`), you can reuse it. Just copy the public key content:

```bash
cat ~/.ssh/id_ed25519.pub
```

### 4.2 Add the Public Key During VM Creation

In the VM creation form, scroll to the **Access** section (sometimes labeled **User data** or **SSH keys**):

1. **Username**: Enter a username for the VM (e.g., `ubuntu` or `user`). Do not use `root` or `admin` — these are reserved and will fail.
2. **SSH Key**: Paste the full contents of your `~/.ssh/id_ed25519.pub` file.

Alternatively, if Nebius prompts for "User configuration" in cloud-init format, you can use:

```yaml
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - ssh-ed25519 AAAA...your-key... nebius-hw3
```

### 4.3 Launch the VM

Click **Create** (or **Launch**). The VM will move through states: `Provisioning` → `Starting` → `Running`. This typically takes 2–5 minutes for an H100 instance.

### 4.4 Get the Public IP Address

Once the VM is in `Running` state, find its public IP in the VM detail page. It is listed under **Network Interfaces** → **Public IP Address**. Copy this IP — you will need it for every SSH connection.

### 4.5 Connect via SSH

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@<YOUR_VM_PUBLIC_IP>
```

Replace `ubuntu` with whatever username you chose in Step 4.2, and replace `<YOUR_VM_PUBLIC_IP>` with the IP from Step 4.4.

If you see a fingerprint prompt on the first connection, type `yes`.

### 4.6 Verify GPU Access

Once connected, run:

```bash
nvidia-smi
```

Expected output: a table showing one H100 80GB GPU with driver version, CUDA version, and current memory usage (should be ~0 MiB used of 81920 MiB total).

If `nvidia-smi: command not found` → the VM may not have NVIDIA drivers installed. This should not happen with the CUDA-pre-installed Ubuntu image. If it does, see Troubleshooting section.

---

## Step 5: Initial VM Configuration

All commands in this section run on the VM (after SSH-ing in).

### 5.1 Update Packages

```bash
sudo apt-get update && sudo apt-get upgrade -y
```

This takes 2–5 minutes. Do it once at the start to ensure you are not working with stale packages.

### 5.2 Verify Docker

The Ubuntu CUDA image from Nebius may or may not have Docker pre-installed. Check:

```bash
docker --version
```

If Docker is not installed:

```bash
# Install Docker CE
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to the docker group (so you do not need sudo every time)
sudo usermod -aG docker $USER

# Apply the group change without logging out
newgrp docker

# Verify
docker run --rm hello-world
```

### 5.3 Verify docker-compose (compose plugin)

```bash
docker compose version
```

Modern Docker installs include Compose V2 as a plugin (`docker compose`, not `docker-compose`). If the command is not found:

```bash
sudo apt-get install docker-compose-plugin -y
docker compose version
```

The HW3 repo uses `docker compose` (V2 syntax). If you only have the older `docker-compose` (V1), install the plugin above.

### 5.4 Install uv

`uv` is the Python package manager used in HW3 (replaces pip for dependency management). It is dramatically faster for installing vLLM and its complex dependency tree.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or source ~/.zshrc depending on your shell
uv --version       # verify
```

### 5.5 Install Python Development Headers

vLLM's `torch.compile` path requires Python C headers:

```bash
sudo apt-get install python3-dev python3-pip -y
```

Verify the headers are present:

```bash
python3 -c "import sysconfig; print(sysconfig.get_path('include'))"
# Should print something like /usr/include/python3.10
```

### 5.6 Verify CUDA

```bash
nvcc --version
```

This should show CUDA 12.x (the version packaged with the Nebius Ubuntu image). vLLM requires CUDA 12.1 or later.

If `nvcc` is not found but `nvidia-smi` works, the CUDA toolkit headers are not in your PATH. Add them:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

### 5.7 Check Available Disk Space

```bash
df -h /
```

You should see ~200 GiB total with most of it free. If you see only 50 GiB, the disk size did not apply correctly during VM creation — you can resize it in the Nebius console (Compute Cloud → Disks → select disk → Edit → resize).

---

## Step 6: Port Forwarding Setup

### Why Port Forwarding Is Needed

All five services in HW3 (vLLM, agent, Grafana, Prometheus, Langfuse) listen on `localhost` on the VM. They are not exposed to the internet directly — this is intentional for security. SSH port forwarding creates encrypted tunnels so that `localhost:<port>` on your laptop transparently reaches the same port on the VM.

The five ports you need to forward:

| Port | Service | URL once forwarded |
|------|---------|-------------------|
| 3000 | Grafana | http://localhost:3000 |
| 9090 | Prometheus | http://localhost:9090 |
| 3001 | Langfuse | http://localhost:3001 |
| 8000 | vLLM API | http://localhost:8000 |
| 8001 | Agent server | http://localhost:8001 |

### Method A: VS Code or Cursor Remote-SSH (Strongly Recommended)

This is the recommended method. It gives you automatic port forwarding, a remote file explorer, and an integrated terminal — all in one.

**Setup:**

1. Install the **Remote-SSH** extension in VS Code or Cursor (`Cmd+Shift+X` → search "Remote - SSH" → Install).
2. Open the Command Palette (`F1` or `Cmd+Shift+P`).
3. Type and select: **Remote-SSH: Connect to Host...**
4. Click **Add New SSH Host...**
5. Enter the SSH command: `ssh ubuntu@<YOUR_VM_PUBLIC_IP>`
6. Select your SSH config file to save to (usually `~/.ssh/config`).
7. Click **Connect to Host** → select your VM's IP.

VS Code/Cursor will install a remote server on the VM automatically (takes ~1 minute the first time) and open a new window connected to the VM.

**Forwarding ports in VS Code/Cursor:**

1. In the bottom panel, click the **Ports** tab (next to Terminal).
2. Click **Forward a Port**.
3. Type `3000` and press Enter.
4. Repeat for `9090`, `3001`, `8000`, `8001`.
5. Each forwarded port shows a green dot and a "Local Address" of `localhost:<port>`.

Now `localhost:3000` in your laptop's browser reaches Grafana on the VM.

**Why this is better than plain SSH:**
- Port forwards survive terminal window closures (they are managed by the VS Code process, not a shell session).
- You can edit files on the VM as if they were local.
- The integrated terminal runs on the VM, so no extra SSH session needed.

### Method B: Plain SSH with -L Flags (Fallback)

If you prefer the terminal or cannot use VS Code:

```bash
ssh -i ~/.ssh/id_ed25519 \
    -L 3000:localhost:3000 \
    -L 9090:localhost:9090 \
    -L 3001:localhost:3001 \
    -L 8000:localhost:8000 \
    -L 8001:localhost:8001 \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    ubuntu@<YOUR_VM_PUBLIC_IP>
```

The `-L local:remote:port` flags create one tunnel each. `-o ServerAliveInterval=60` sends a keepalive every 60 seconds to prevent the tunnel from timing out during long operations (model download, eval runs).

**Critical gotcha with plain SSH**: If you close the terminal window that ran this command, all port forwards die. Services on the VM keep running, but you lose access from your laptop until you reconnect. To prevent this, run the SSH command inside a `tmux` session on your laptop, or use VS Code instead.

**Verifying port forwards:**

Open each URL in your laptop browser before proceeding. Before services are running:
- Prometheus (`localhost:9090`): Should show the Prometheus UI even before vLLM starts.
- Grafana (`localhost:3000`): Should show the login page.
- Langfuse (`localhost:3001`): Should show the sign-up/login page.
- vLLM (`localhost:8000`): Will time out until you start vLLM in Phase 1. That is expected.
- Agent (`localhost:8001`): Will time out until you start the agent server in Phase 3. That is expected.

If Prometheus is unreachable, the Docker stack is not running yet (see Step 9). If Grafana is unreachable but Prometheus is fine, check the Docker logs for the Grafana container.

If nothing is reachable and you are using plain SSH, the port forward session may have died. Reconnect with the command above.

---

## Step 7: Clone the Repo and Install Dependencies

All commands in this section run on the VM terminal (VS Code integrated terminal, or SSH session).

### 7.1 Clone the Repository

```bash
git clone https://github.com/GlebBerjoskin/mlops-assignment.git
cd mlops-assignment
```

### 7.2 Install Python Dependencies with uv

```bash
uv sync
```

What `uv sync` does:
- Reads `pyproject.toml` and `uv.lock` to determine exact dependency versions.
- Creates a virtual environment at `.venv/`.
- Installs all dependencies including vLLM, LangGraph, Langfuse, and their transitive dependencies.

This takes 5–15 minutes the first time because vLLM and PyTorch are large packages. Subsequent runs are fast (reads from cache).

Do not use `pip install` directly for this project — it bypasses the lockfile and can install incompatible versions of packages that vLLM is sensitive about.

### 7.3 Set Up the Environment File

```bash
cp .env.example .env
```

Open `.env` in your editor and review the defaults. You do not need to fill in Langfuse API keys yet — those come in Phase 4 once Langfuse is running. The `VLLM_BASE_URL` and `VLLM_MODEL` fields have sensible defaults that point to your local vLLM instance.

### 7.4 Understand the Project Structure

```
mlops-assignment/
├── agent/           # LangGraph agent (graph.py, prompts.py) — you implement this
├── evals/           # Eval runner and eval set
├── infra/           # Docker Compose and Grafana/Prometheus configs
├── load_test/       # Load test driver for Phase 6
├── results/         # Where eval output files go
├── screenshots/     # Where screenshots go for submission
├── scripts/         # setup scripts (load_data.py, start_vllm.sh)
├── docker-compose.yml
├── .env.example
└── pyproject.toml
```

---

## Step 8: Load BIRD Data

### What BIRD-bench Is

BIRD (Big Bench for Large-scale Database Grounded Text-to-SQL Evaluation) is an academic benchmark dataset for evaluating text-to-SQL systems. HW3 uses a curated 30-question subset from this benchmark. The data consists of SQLite database files (real databases with actual data) and corresponding question/gold-SQL JSON files.

### Load the Data

```bash
uv run python scripts/load_data.py
```

This script downloads approximately 500 MB of data. Expect it to take 2–10 minutes depending on network speed.

### Verify the Data

```bash
ls data/bird/
# Should show database subdirectories

find data/bird/ -name "*.sqlite" | wc -l
# Should show at least 5 .sqlite files

ls evals/eval_set.jsonl
# Should exist and contain 30 lines
wc -l evals/eval_set.jsonl
```

If `data/bird/` is empty or the script failed, check the error output — it is usually a network issue. Re-run the script.

---

## Step 9: Start the Observability Stack

### 9.1 Start Docker Services

```bash
docker compose up -d
```

This starts four Docker containers:
- **prometheus**: Scrapes vLLM's `/metrics` endpoint
- **grafana**: Dashboard service with pre-configured Prometheus data source
- **langfuse**: Trace storage API (requires the next two)
- **postgres**: Langfuse's database backend
- **redis** (if present): Langfuse queue

The first run pulls Docker images (~3–4 GB total). Expect 2–5 minutes.

### 9.2 Verify Each Service Is Healthy

```bash
docker compose ps
```

All containers should show `Up` or `healthy` status. If any show `Exited`, check the logs:

```bash
docker compose logs langfuse
docker compose logs prometheus
docker compose logs grafana
```

### 9.3 Access Each Service from Your Laptop

With port forwards active (Step 6), open in your laptop browser:

**Prometheus** (`http://localhost:9090`):
- Click Status → Targets.
- You should see a target for `vllm` listed as DOWN. This is correct — vLLM is not running yet. Prometheus itself being up and the target being listed (even as DOWN) confirms the scrape config is correct.

**Grafana** (`http://localhost:3000`):
- Login: `admin` / `admin` (you will be prompted to change the password — you can skip or set a new one).
- Click the Dashboards icon (four squares) in the left sidebar.
- You should see a dashboard called something like "vLLM Serving" or "LLM Serving Dashboard". Click it. Panels will show "No data" — this is expected until vLLM starts.

**Langfuse** (`http://localhost:3001`):

Langfuse requires a one-time local setup:

1. Click **Sign up**.
2. Enter any email and password (this is a local instance — no verification email is sent, no data leaves the VM).
3. Click **Create account**.
4. You are prompted to create an **Organization** — enter any name (e.g., "hw3").
5. Create a **Project** — enter any name (e.g., "sql-agent").
6. You will see the project dashboard. Click **Settings** in the left sidebar.
7. Under **API Keys**, click **Create API Key**.
8. Copy the **Public Key** and **Secret Key**. Store them immediately — the secret key is only shown once.
9. Add both keys to your `.env` file on the VM:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=http://localhost:3001
   ```

These keys are used in Phase 4 when you add the Langfuse callback to your agent.

---

## Step 10: Cost Management on Nebius

The H100 at ~$3.85/hr on-demand is meaningful money. Here is how to spend it wisely.

### Stop vs. Delete: Know the Difference

| Action | What happens | When to use |
|--------|-------------|-------------|
| **Stop VM** | CPU/GPU billing stops. Disk is preserved. VM can restart in 2–3 minutes. | Every time you take a break longer than 30 minutes |
| **Delete VM** | Everything including disk is destroyed. | Only when fully done with the assignment |
| **Restart VM** | VM reboots. SSH sessions close temporarily. | When you need to apply kernel updates or recover from a hung process |

To stop the VM: Nebius console → Compute Cloud → Virtual Machines → select your VM → **Stop** (not Delete).

To restart it: same path → **Start**. You will get the same public IP if you reserved an IP, or a new public IP if you used an ephemeral one. Reserve your IP (Nebius VPC → Public IPs → Reserve) to avoid updating your SSH config every time.

### When You Need the H100

| Phase | H100 Required? | Approximate H100 Hours |
|-------|---------------|------------------------|
| 0 (Setup, Docker, data load) | No | 0 |
| 1 (vLLM configuration, first model load) | Yes | 1–3 hours |
| 2 (Grafana dashboard) | CPU vLLM OK | 0 (H100) |
| 3 (Agent implementation) | No | 0 |
| 4 (Langfuse tracing setup) | No (any backend) | 0 |
| 5 (Eval run with 30B model) | Yes | 1–2 hours |
| 6 (SLO diagnosis, multiple load test runs) | Yes | 3–6 hours |
| **Total focused run** | | **~6–12 H100 hours** |

**Recommended workflow to minimize cost:**

1. Complete Phases 0, 2 (Grafana panel building), 3, and 4 with the H100 stopped.
2. Start the H100 only when you are ready to load vLLM (Phase 1). Stay focused — model load takes 5–10 minutes, so plan your work session.
3. After Phase 1 config is stable, stop the H100. Come back for Phase 5 (eval) and Phase 6 (load test).
4. For Phase 6, plan two or three focused 2-hour sessions rather than leaving the VM running all day.

### Practical Cost Discipline

- **Set a billing alert** in the Nebius console (Billing → Notifications) to email you if daily spend exceeds a threshold you set.
- **Check your VM status** before and after every work session. The console shows VM state on the main Compute page.
- **One VM is enough.** You do not need separate VMs for different phases.
- **Do not leave vLLM loading a model and walk away.** If the process hangs and you are not watching, you pay for a stuck GPU.

---

## Troubleshooting Common Nebius / Setup Issues

### VM Won't Start

**Symptom**: VM stuck in "Provisioning" or immediately goes to "Error" state.

**Common causes:**
1. GPU quota exceeded — you tried to create a VM but your quota is 0 H100 GPUs. Go to Step 2.3.
2. The selected subnet has no available IPs (rare with default subnet). Try creating a new subnet or contact Nebius support.
3. Nebius platform issue — check the Nebius status page at [status.nebius.com](https://status.nebius.com).

### SSH Connection Refused

**Symptom**: `ssh: connect to host X.X.X.X port 22: Connection refused`

**Debugging steps:**
1. Check that the VM is in `Running` state in the console (not `Stopped` or `Starting`).
2. Wait 30–60 seconds after the VM shows `Running` — the SSH daemon takes a moment to start.
3. Check that you have a public IP assigned (not just a private IP).
4. Check that port 22 is allowed in the network security group (should be the default).
5. Try with verbose output: `ssh -v ubuntu@<IP>` — this shows exactly where the handshake is failing.

### SSH Key Rejected (Permission Denied)

**Symptom**: `ubuntu@X.X.X.X: Permission denied (publickey)`

**Fixes:**
1. Make sure you are using the private key matching the public key you put in Nebius: `ssh -i ~/.ssh/id_ed25519 ubuntu@<IP>`
2. Verify the username matches what you set during VM creation.
3. Check key file permissions: `chmod 600 ~/.ssh/id_ed25519`
4. If you pasted the wrong public key into Nebius, you need to recreate the VM (or add the correct key via Nebius console's serial console access if available).

### Port Forward Not Working

**Symptom**: `localhost:3000` times out or refuses connection in your browser.

**Diagnosis steps:**
1. First: confirm the service is running on the VM. From your SSH session: `curl -s localhost:3000 | head -5`. If this also fails, the service is not running — not a port forward problem.
2. If the service responds on the VM but not from your laptop, the port forward is broken. Re-establish it (reconnect via VS Code Remote-SSH or re-run the SSH -L command).
3. With plain SSH: check that the SSH session with port forwards is still open. Run `ps aux | grep ssh` on your laptop to see if the session is alive.
4. Check that no local process on your laptop is using the same port: `lsof -i :3000`

### Docker Permission Denied

**Symptom**: `permission denied while trying to connect to the Docker daemon socket`

**Fix:**
```bash
sudo usermod -aG docker $USER
newgrp docker
# Verify:
docker ps
```

If `newgrp docker` does not work, log out and back in via SSH.

### nvidia-smi Shows No Devices

**Symptom**: `nvidia-smi` shows "No devices were found" even on an H100 VM.

**Diagnosis:**
```bash
lspci | grep -i nvidia   # Should show the GPU hardware
lsmod | grep nvidia      # Should show nvidia driver modules
dmesg | grep -i nvidia   # Look for driver load errors
```

If the hardware is present but drivers are not loaded, the CUDA image may not have initialized correctly. Reboot the VM:

```bash
sudo reboot
```

Reconnect after 60–90 seconds and try `nvidia-smi` again. If the problem persists after reboot, the VM image may not have CUDA drivers — contact Nebius support or try a different image.

### Out of Disk Space

**Symptom**: Operations fail with "No space left on device".

**What to clear first (in order of impact):**

```bash
# 1. Check what is using space
df -h /
du -sh /home/* /var/* /tmp/* 2>/dev/null | sort -rh | head -20

# 2. Clean Docker unused images and build cache (often 5–20 GB)
docker system prune -f
docker image prune -a -f

# 3. Clean HuggingFace model cache (careful — this removes cached model weights)
du -sh ~/.cache/huggingface/
# If you need the space: rm -rf ~/.cache/huggingface/hub/

# 4. Clean pip/uv cache
uv cache clean

# 5. Clean apt package cache
sudo apt-get clean
```

If you are genuinely out of space after cleaning, you need to resize the disk. In the Nebius console: Compute Cloud → Disks → select disk → Edit → increase size. Then on the VM:

```bash
sudo growpart /dev/vda 1
sudo resize2fs /dev/vda1
df -h /   # Verify new size
```

### vLLM Fails to Load Model

**Symptom**: `vllm` starts but crashes during model loading.

**Possible causes:**

1. **CUDA OOM**: Not enough GPU memory. Switch to FP8 quantization (`--quantization fp8`). Check that no other process is using the GPU: `nvidia-smi`. If another vLLM process is already running, kill it: `pkill -f vllm`.

2. **Disk space**: The model download fails mid-way. Check `df -h /`. HuggingFace downloads to `~/.cache/huggingface/`. Clear partial downloads: `rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/blobs/incomplete_*`

3. **Network issue during download**: Qwen3-30B-A3B-Instruct-2507 is ~35–70 GB. If the download is interrupted, the model will not load. Re-run vLLM and it will resume the download.

4. **Wrong CUDA version**: vLLM requires CUDA 12.1+. Check `nvcc --version` and compare against vLLM's requirements.

---

## Quick Reference Card

### SSH and Connection

```bash
# Connect with port forwarding (all 5 ports)
ssh -i ~/.ssh/id_ed25519 \
    -L 3000:localhost:3000 \
    -L 9090:localhost:9090 \
    -L 3001:localhost:3001 \
    -L 8000:localhost:8000 \
    -L 8001:localhost:8001 \
    -o ServerAliveInterval=60 \
    ubuntu@<YOUR_VM_IP>

# Check GPU
nvidia-smi

# Check disk
df -h /
```

### Docker Stack

```bash
# Start all services
docker compose up -d

# Check all services are up
docker compose ps

# View logs for a specific service
docker compose logs langfuse
docker compose logs grafana
docker compose logs prometheus

# Stop all services
docker compose down

# Restart a single service
docker compose restart grafana
```

### vLLM (run in a tmux session on the VM)

```bash
# Start vLLM (see Phase 1 / scripts/start_vllm.sh for full flags)
uv run python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 --port 8000 \
  --quantization fp8

# Check vLLM is responding (from VM or via port forward from laptop)
curl http://localhost:8000/health

# Smoke test: get a SQL query
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-30B-A3B-Instruct-2507","messages":[{"role":"user","content":"Write SQL to count rows in a table called users. /no_think"}],"max_tokens":100}'

# View metrics (pipe through sort to find vllm: metrics)
curl -s localhost:8000/metrics | grep "^vllm:" | sort | head -30
```

### Service URLs (from Your Laptop Browser)

| Service | URL | Credentials |
|---------|-----|------------|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | none |
| Langfuse | http://localhost:3001 | whatever you set at sign-up |
| vLLM API | http://localhost:8000 | none (api_key="token-abc" from .env) |
| Agent server | http://localhost:8001 | none |

### Useful One-Liners

```bash
# Check all running processes on the VM
ps aux | grep -E "vllm|python|uvicorn" | grep -v grep

# Kill any hung vLLM process
pkill -9 -f "vllm.entrypoints"

# Check how much GPU memory is being used
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Run a single eval question manually (agent must be running)
curl -s -X POST http://localhost:8001/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "How many employees are there?", "db": "employee_hire_evaluation"}' \
  | python3 -m json.tool

# Watch Prometheus metrics live
watch -n 2 'curl -s localhost:8000/metrics | grep -E "^vllm:(num_requests|gpu_cache)" | sort'
```

### tmux Basics (Recommended for Long-Running Processes)

vLLM and the load test runner need to keep running even if your SSH session drops. Use `tmux`:

```bash
# Start a named session
tmux new -s vllm

# Run vLLM inside tmux
uv run python -m vllm.entrypoints.openai.api_server ...

# Detach from session (keeps vLLM running)
Ctrl-b d

# Re-attach later
tmux attach -t vllm

# List all sessions
tmux ls
```

---

## Documentation Links

For Nebius-specific details beyond this guide, the authoritative references are:

- VM creation: [docs.nebius.com/compute/virtual-machines/manage](https://docs.nebius.com/compute/virtual-machines/manage)
- GPU VM types and preset names: [docs.nebius.com/compute/virtual-machines/types](https://docs.nebius.com/compute/virtual-machines/types)
- SSH connection guide: [docs.nebius.com/compute/virtual-machines/connect](https://docs.nebius.com/compute/virtual-machines/connect)
- CLI configuration: [docs.nebius.com/cli/configure](https://docs.nebius.com/cli/configure)
- Compute pricing: [docs.nebius.com/compute/resources/pricing](https://docs.nebius.com/compute/resources/pricing)
- Compute quickstart: [docs.nebius.com/compute/quickstart](https://docs.nebius.com/compute/quickstart)
