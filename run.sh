#!/usr/bin/env bash
# Run commands for this repo inside the 2019 adversarial-policies Docker image.
#
# Usage:
#   ./run.sh test [pytest args]   run tests (e.g. ./run.sh test -m "not slow")
#   ./run.sh lint                 check formatting (black, isort) and style (flake8)
#   ./run.sh fmt                  auto-format with black and isort
#   ./run.sh shell                interactive shell in the container
#   ./run.sh <command...>         run any command, e.g. ./run.sh python -m noisebackup.sweep ...
#
# The image's *installed* copy of the original repo (`aprl`) is used; this repo is
# mounted at /backup and put on PYTHONPATH.
set -euo pipefail

# Pinned by digest (the build tagged :latest as of 2026-11-12, created 2019-12-05) so everyone
# runs identical software even if the tag is ever moved.
IMAGE="docker.io/humancompatibleai/adversarial_policies@sha256:1df16d7a129f55d997aa9faba55407f76a8ee1b6305461e5faa56ea599602b8c"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MJKEY="${HOME}/.mujoco/mjkey.txt"

if ! command -v podman >/dev/null; then
    echo "podman is required; install it with your distribution's package manager" >&2
    echo "(see Setup in README.md)." >&2
    exit 1
fi

if [[ ! -f "${MJKEY}" ]]; then
    echo "MuJoCo key not found at ${MJKEY}" >&2
    echo "Download it from https://www.roboti.us/file/mjkey.txt" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    sed -n '2,12p' "$0"
    exit 1
fi

TTY_FLAGS=()
if [[ -t 0 && -t 1 ]]; then
    TTY_FLAGS=(-it)
fi

container() {
    podman run --rm "${TTY_FLAGS[@]}" \
        -v "${MJKEY}:/root/.mujoco/mjkey.txt:ro,Z" \
        -v "${REPO_DIR}:/backup:Z" \
        -w /backup \
        -e PYTHONPATH=/backup \
        -e PYTHONDONTWRITEBYTECODE=1 \
        "${IMAGE}" "$@"
}

LINT_PATHS=(noisebackup tests)

case "$1" in
    test)
        shift
        container python -m pytest -p no:cacheprovider "$@"
        ;;
    lint)
        container bash -c "black --check ${LINT_PATHS[*]} \
            && isort --check-only --diff --recursive ${LINT_PATHS[*]} \
            && flake8 ${LINT_PATHS[*]} \
            && echo 'lint: all clean'"
        ;;
    fmt)
        container bash -c "isort --recursive ${LINT_PATHS[*]} && black ${LINT_PATHS[*]}"
        ;;
    shell)
        container /bin/bash
        ;;
    *)
        container "$@"
        ;;
esac
