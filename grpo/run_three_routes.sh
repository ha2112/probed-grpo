#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
route="${1:-all}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${route}" in
    random|official|probed)
        "${PYTHON_BIN}" "${SCRIPT_DIR}/codeforces_reward.py" --check
        exec bash "${SCRIPT_DIR}/train.sh" "${route}" "$@"
        ;;
    all)
        "${PYTHON_BIN}" "${SCRIPT_DIR}/codeforces_reward.py" --check
        for name in random official probed; do
            bash "${SCRIPT_DIR}/train.sh" "${name}" "$@"
        done
        ;;
    *)
        echo "Usage: $0 [random|official|probed|all] [additional verl overrides...]" >&2
        exit 2
        ;;
esac
