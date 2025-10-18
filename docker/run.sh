#!/bin/bash

# This script runs the container, starting a JupyterLab server in that container.
# You should run this each time you start working in order to enter the necessary environment for this project.

# Go to this script's location.
SCRIPT_DIR="$(dirname "$0")"
cd "${SCRIPT_DIR}"

# Assume the repository root is one level up from here
REPO_PATH="$(realpath ..)"

# Load Docker configuration variables from config.env.
CONFIG_FILE="config.env"
source "${CONFIG_FILE}"

# Check that the DATA_PATH variable from config.env is valid.
if [ -z "${DATA_PATH}" ]; then
    echo "Error: DATA_PATH is not set in ${CONFIG_FILE}."
    echo "Please specify the path to the data directory in ${CONFIG_FILE}."
    exit 1
fi
if [ ! -d "${DATA_PATH}" ]; then
    echo "Error: The data directory ${DATA_PATH} specified in ${CONFIG_FILE} does not exist."
    exit 1
fi

echo "REPO_PATH=${REPO_PATH}"
echo "DATA_PATH=${DATA_PATH}"

# Give the container a unique name
CONTAINER_NAME="dagict-container"

# You can use `run.sh --cpu` to run with CPU-only.
# Our code is designed with GPU availability expected, though, so this will only give limited functionality.
GPU_FLAG=""
if [[ "${1:-}" == "--cpu" ]]; then
    # Remove --cpu from arguments
    shift  
else
    # Default behavior is to try to use GPU.
    if command -v nvidia-smi &>/dev/null; then
        GPU_FLAG="--gpus all"
    else
        echo "Error: Failed to run nvidia-smi, so cannot expose GPUs to container."
        echo "Make sure you have appropriate NVIDIA drivers for your computer installed."
        echo "You can rerun this script with '--cpu' as the first argument to use only CPU."
        exit 1
    fi
fi

# Reminder message about what gets deleted when the container is removed
reminder_message() {
    echo "REMINDER: Once you close the container '${CONTAINER_NAME}', the following will be lost:"
    echo "  - Any files saved outside the mounted volumes (/repo/ and /data/)."
    echo "  - The container's internal state (such as installed packages)."
    echo "Make sure to write all data to /data/, and keep record of any new package installations you had to make to aid in reproducibility."
    echo "If you are running Jupyter Lab (i.e. the default behavior):"
    echo "  - You can access it via a web browser at localhost:8888/lab"
    echo "  - You can exit it by pressing Ctrl+C"
}

# Case 1 - The container is already running.
if docker ps --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' is already running - not going to start a new container."
    exit 1
fi

# Case 2 - The container does not exist yet, so we need to run it.
# `-it` : Gives you interactive mode pseudo-terminal (necessary for an interactive shell session inside the container)
# `--rm` : Deletes the container after you close it
# `-v` : Mounts a local filesystem directory to a directory inside the container
# `-w` : Sets your working directory inside the container
# `-p` : Exposes port 8888 of your local filesystem to the container for Jupyter to use
# "$@" : If you give this script more arguments, then they'll get passed directly to `docker run`
#   to override the default CMD behavior specified in the Dockerfile.
echo "Starting a new container named '${CONTAINER_NAME}'..."
echo "Your laptop's ${REPO_PATH} will be available in the container at /repo/"
echo "Your laptop's ${DATA_PATH} will be available in the container at /data/"
reminder_message
MSYS_NO_PATHCONV=1 docker run \
    -it \
    --rm \
    --name "${CONTAINER_NAME}" \
    $GPU_FLAG \
    -v "${REPO_PATH}":"/repo/" \
    -v "${DATA_PATH}":"/data/" \
    -w "/repo/" \
    -p 8888:8888 \
    dagict-image \
    "$@"
