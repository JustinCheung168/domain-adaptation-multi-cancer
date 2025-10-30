# Improving Artifact Robustness for CT Deep Learning Models Without Labeled Artifact Images via Domain Adaptation

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
   cd ./domain-adaptation-ct
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
    -v "C:\Users\myname\domain-adaptation-ct":"/repo/" \
    -v "C:\Users\myname\domain-adaptation-ct":"/data/" \
    -p 8888:8888 \
    dact-image \
    "$@"
```

## Contents

- `src/`: Source code for this project.
   - `domain_adaptation_ct/`: Module for this project.
      - `config/`: Classes for config files.
      - `dataset/`: Classes for reading and interacting with datasets.
      - `learn/`: Classes & functions for model training & evaluation.
      - `logging/`: Classes for logging.
      - `preprocess/`: Functions for preprocessing data.
      - `visualize/`: Classes for visualizing images.
- `results/`: Raw training/validation/test result outputs.
- `src/`: Source code files. Primarily contains sinogram manipulation code right now.
- `*_pipeline*.ipynb`: Jupyter Notebooks used for model training for each experiment on OrganAMNIST data. In our convention expanding on notation used by Geirhos et al., "A" models are trained on single distortions, "C" models are trained on all-but-one distortion, and "D" models are based on Ganin & Lempitsky (2015)'s domain adaptation architecture.
- `GaninDALoss.ipynb`: Quick demonstration that the loss function component used for the label predictor successfully excludes influence of target domain instances.
- `Image_Manipulation*.ipynb`: Jupyter Notebooks for producing distorted data.
- `*resnet*.py`: Classes and script for our custom ResNet-50 configuration based on Ganin & Lempitsky (2015), and for a comparable unmodified ResNet-50.
- `ct_projection_proto.ipynb`: Exploration of sinogram manipulation.
- `evaluate_experiment.ipynb`: Model evaluation code.
- `medmnist_eda.ipynb`: Exploratory data analysis of MedMNIST datasets.
- `view_test_results.ipynb`: Model training/validation curve and test matrix visualization code. 

## Usage

1. Download data from [huggingface.co/datasets/NNDLCLASS](huggingface.co/datasets/NNDLCLASS).
2. Specify details of your experiment in `experiment_configs/`. Example config values are specified as arguments to the commands below.
3. Run training:
```bash
./scripts/run_training.py experiment_configs/dann_train_quickcheck_config.yaml
```
4. Run evaluation:
```bash
./scripts/run_evaluation.py experiment_configs/dann_test_quickcheck_config.yaml
```

## Citation

Cheung, J., Savine, S., Nguyen, C., Lu, L., & Yasin, A. S. (2025). Improving Artifact Robustness for CT Deep Learning Models Without Labeled Artifact Images via Domain Adaptation. arXiv preprint arXiv:2510.06584.

## TODO

Restore the image manipulation notebooks.
Restore the OrganAMNIST preprocessing code.
Update this documentation.
