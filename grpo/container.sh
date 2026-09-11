#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${GRPO_IMAGE:-probed-grpo:verl0.5.0}"
action="${1:-shell}"
if [[ $# -gt 0 ]]; then shift; fi
case "${action}" in
    build)
        exec docker build --platform linux/amd64 -f "${ROOT_DIR}/grpo/Dockerfile" -t "${IMAGE}" "${ROOT_DIR}"
        ;;
    shell|run)
        options=(--rm --gpus all --network host --shm-size 16g
            --mount "type=bind,source=${ROOT_DIR},target=/workspace/probed-grpo"
            --workdir /workspace/probed-grpo)
        if [[ "${action}" == shell ]]; then options+=(-it); fi
        # Forward only named settings, including an optional HF token.
        for name in N_GPUS TENSOR_PARALLEL_SIZE MONOLITH_URL JUDGE_WORKERS HF_TOKEN WANDB_API_KEY WANDB_MODE; do
            if [[ -n "${!name:-}" ]]; then options+=(--env "${name}"); fi
        done
        if [[ $# == 0 ]]; then set -- bash; fi
        exec docker run "${options[@]}" "${IMAGE}" "$@"
        ;;
    *) echo "Usage: bash grpo/container.sh {build|shell|run [command...]}" >&2; exit 2 ;;
esac
