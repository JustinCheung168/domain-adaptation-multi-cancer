# Multi cancer project to be named

## Environment Setup

### Setting up Docker

This project uses Docker to control requirements and aid in cross-platform compatibility.

#### Prerequisites

1. **Install Docker:**
   - For Linux: Follow the instructions at https://docs.docker.com/engine/install/.
   - For Mac & Windows: Download Docker Desktop from https://www.docker.com/products/docker-desktop/.
1. **(Windows Only) Install Git Bash:**
   - Download and install Git Bash from https://git-scm.com/downloads.

#### Building the Docker Image (one-time setup)

1. Open a terminal (if you are using Windows, this terminal should be a Git Bash terminal).
1. Open the Docker application. Do any first-time setup it prompts for.
1. Clone this repository.
1. Navigate to the project directory:
   ```bash
   cd ./domain-adaptation-multi-cancer
   ```
1. Run the `build.sh` script to build the Docker image:
   ```bash
   ./docker/build.sh
   ```

#### Setting your data path

Open `./docker/config.env` and set the path to where your data will live.

#### Running the Environment

1. Start the Docker container and open an interactive Bash session inside it by running the `run.sh` script:
```bash
./docker/run.sh bash
```

2. Install the Python dependencies for this project:
```bash
pip install -e .
```

Our preferred workflow is to attach a VSCode session to the container produced by the above (https://code.visualstudio.com/docs/devcontainers/attach-container).

If you prefer to use Jupyter Lab instead, you can run the below script. When the Jupyter Lab server comes up, you can start running code and editing in the Jupyter Lab environment by going to `localhost:8888/lab` in your browser.

3. (Optional) Open Jupyter Lab
```bash
./scripts/start_jupyter_lab.sh
```

#### Hardware Recommendations

You likely need at least 32GB of RAM to work with this project, due to the size of the preprocessed training data file.

This project expects to have NVIDIA GPU access.

#### Troubleshooting

In one Windows system, we observed a failure in the `docker/run.sh` script that is potentially resolved by replacing the final docker run command with the following:
```bash
MSYS_NO_PATHCONV=1 winpty docker run \
    -it \
    --rm \
    --name "${CONTAINER_NAME}" \
    $GPU_FLAG \
    -v "C:\Users\myname\domain-adaptation-multi-cancer":"/repo/" \
    -v "C:\Users\myname\domain-adaptation-multi-cancer":"/data/" \
    -p 8888:8888 \
    dact-image \
    "$@"
```

## Contents

TBD

## Usage

2. Specify details of your experiment in `experiment_configs/`. Example config values are specified as arguments to the commands below.
3. Run training:
```bash
./scripts/run_training.py experiment_configs/dann_train_quickcheck_config.yaml
```
4. Run evaluation:
```bash
./scripts/run_evaluation.py experiment_configs/dann_test_quickcheck_config.yaml
```


## TODO

Restore the image manipulation notebooks.
Restore the OrganAMNIST preprocessing code.
Update this documentation.
