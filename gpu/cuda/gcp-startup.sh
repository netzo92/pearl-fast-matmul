#!/usr/bin/env bash
# Intended only for a temporary CUDA-ready VM with a GCE-enforced 30-minute TTL.
# No service account, customer requests, pool credentials or mining submission.
set -euo pipefail
exec > >(tee -a /var/log/pearl-fast-matmul.log /dev/ttyS0) 2>&1
trap 'rc=$?; echo "PEARL_BENCHMARK_EXIT=$rc"' EXIT
export PATH="/usr/local/cuda/bin:$PATH"
export DEBIAN_FRONTEND=noninteractive
export CARGO_HOME=/opt/pearl-cargo
export RUSTUP_HOME=/opt/pearl-rustup
revision="$(curl -fsS --max-time 10 -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/pearl-revision)"
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo 'Missing exact source revision'; exit 2; }
command -v nvcc
nvidia-smi
apt-get update -qq
apt-get install -y -qq git curl ca-certificates python3-venv build-essential
mkdir -p /opt/pearl-benchmark
cd /opt/pearl-benchmark
git init -q source
git -C source fetch --depth=1 https://github.com/netzo92/pearl-fast-matmul.git "$revision"
git -C source checkout --detach -q FETCH_HEAD
curl -fsS --proto '=https' --tlsv1.2 https://sh.rustup.rs -o rustup-init.sh
sh rustup-init.sh -y --profile minimal --default-toolchain 1.93.1 --no-modify-path
export PATH="$CARGO_HOME/bin:$PATH"
python3 -m venv venv
venv/bin/python -m pip install -r source/gpu/cuda/requirements.txt
# Timeout below covers compilation, fixture generation, validation and timing.
PYTHON=/opt/pearl-benchmark/venv/bin/python \
    timeout 900s bash source/gpu/cuda/run.sh /opt/pearl-benchmark/report.json
venv/bin/python - <<'PY'
import base64, gzip, hashlib
from pathlib import Path
raw = Path('/opt/pearl-benchmark/report.json').read_bytes()
print('PEARL_REPORT_SHA256=' + hashlib.sha256(raw).hexdigest(), flush=True)
encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
chunks = [encoded[i:i+4000] for i in range(0, len(encoded), 4000)]
for i, chunk in enumerate(chunks):
    print(f'PEARL_REPORT_CHUNK={i}/{len(chunks)}:{chunk}', flush=True)
PY
