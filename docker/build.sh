#!/bin/bash

# This script builds the Docker image for this project based on the Dockerfile.
# You only need to run this one time for your initial setup.

# Go to this script's location.
SCRIPT_DIR="$(dirname "$0")"
cd "${SCRIPT_DIR}"

# Build the Docker image based on `Dockerfile`.
# Allow the build process to use this folder as its context.
docker build --force-rm --no-cache --platform linux/amd64 -t dagict-image -f Dockerfile .
