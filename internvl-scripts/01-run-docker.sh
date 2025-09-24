#!/usr/bin/env bash
set -euo pipefail

IMAGE="vault.habana.ai/gaudi-docker/1.22.1/ubuntu24.04/habanalabs/pytorch-installer-2.7.1:latest"

# Make sure the repo exists at $HOME/vllm-fork (run your clone script first)
if [[ ! -d "$HOME/vllm-fork" ]]; then
  echo "vllm-fork not found at \$HOME/vllm-fork. Clone it first." >&2
  exit 1
fi

docker run -it --rm \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  -v "$HOME":/workdir \
  -w /workdir/vllm-fork-internvl \
  --cap-add=sys_nice \
  --net=host \
  --ipc=host \
  --entrypoint bash \
  "$IMAGE" -lc '
    set -euo pipefail

    # Sanity checks
    echo "[inside] python:" && python3 -V || true
    echo "[inside] pip:"    && python3 -m pip -V || true
    echo "[inside] pwd:"    && pwd
    echo "[inside] ls:"     && ls -la

    # Upgrade pip first
    python3 -m pip install -U pip

    # Install HPU requirements from the repo
    python3 -m pip install -r requirements-hpu.txt

    # Install vLLM for HPU in editable mode
    VLLM_TARGET_DEVICE=hpu python3 -m pip install -e . --no-build-isolation

    # Tools you wanted
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends rclone git-lfs

    # Extras
    python3 -m pip install "huggingface_hub[hf_xet]" datasets

    echo "[inside] Setup complete."

    bash
  '
