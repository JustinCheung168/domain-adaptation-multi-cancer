# domain-generalization-ct

## Usage

### Setting up Docker

This project uses Docker to control requirements and aid in cross-platform compatibility.

#### Prerequisites

1. **Install Docker:**
   - For Linux: Follow the instructions at https://docs.docker.com/engine/install/.
   - For Mac & Windows: Download Docker Desktop from https://www.docker.com/products/docker-desktop/.
1. **Windows Only - Install Git Bash:**
   - Download and install Git Bash from https://git-scm.com/downloads.

#### Building the Docker Image (one-time setup)

1. Open a terminal (if you are using Windows, this terminal should be a Git Bash terminal).
1. Open the Docker application. Do any first-time setup it prompts for.
1. Clone this repository.
1. Navigate to the project directory:
   ```bash
   cd ./domain-generalization-ct
   ```
1. Run the `build.sh` script to build the Docker image:
   ```bash
   ./docker/build.sh
   ```

#### Setting your data path

Open `./docker/config.env` and set the path to where your data will live.

#### Running the Project

1. Start the Docker container by running the `run.sh` script:
   ```bash
   ./docker/run.sh
   ```

The default behavior of this script is to open Jupyter Lab. When the Jupyter Lab server comes up, you can start running code and editing in the Jupyter Lab environment by going to `localhost:8888/lab` in your browser.


Alternatively, you can open an interactive Bash session instead of Jupyter Lab via:

```bash
./docker/run.sh bash
```

You can consider attaching a VSCode session to the resulting container.

#### Troubleshooting

In one Windows system, we observed a failure in the `docker/run.sh` script that is potentially resolved by replacing the final docker run command with the following:
```bash
MSYS_NO_PATHCONV=1 winpty docker run \
    -it \
    --rm \
    --name "${CONTAINER_NAME}" \
    $GPU_FLAG \
    -v "C:\Users\myname\domain-generalization-ct":"/repo/" \
    -v "C:\Users\myname\domain-generalization-ct":"/data/" \
    -p 8888:8888 \
    dagict-image \
    "$@"
```

## Contents

- `geirhos/`: Preprocessing & training code for reproducing results from Geirhos et al. (2018).
- `metric_csvs/`: Raw training/validation/test result outputs.
- `src/`: Source code files. Primarily contains sinogram manipulation code right now.
- `*_pipeline*.ipynb`: Jupyter Notebooks used for model training for each experiment on OrganAMNIST data. In our convention expanding on notation used by Geirhos et al., "A" models are trained on single distortions, "C" models are trained on all-but-one distortion, and "D" models are based on Ganin & Lempitsky (2015)'s domain adaptation architecture.
- `GaninDALoss.ipynb`: Quick demonstration that the loss function component used for the label predictor successfully excludes influence of target domain instances.
- `Image_Manipulation*.ipynb`: Jupyter Notebooks for producing distorted data.
- `*resnet*.py`: Classes and script for our custom ResNet-50 configuration based on Ganin & Lempitsky (2015), and for a comparable unmodified ResNet-50.
- `ct_projection_proto.ipynb`: Exploration of sinogram manipulation.
- `evaluate_experiment.ipynb`: Model evaluation code.
- `medmnist_eda.ipynb`: Exploratory data analysis of MedMNIST datasets.
- `view_test_results.ipynb`: Model training/validation curve and test matrix visualization code. 

## References

- Ganin, Y., & Lempitsky, V. (2015, June). Unsupervised domain adaptation by backpropagation. In International conference on machine learning (pp. 1180-1189). PMLR.
- Geirhos, R., Temme, C. R., Rauber, J., Schütt, H. H., Bethge, M., & Wichmann, F. A. (2018). Generalisation in humans and deep neural networks. Advances in neural information processing systems, 31.

## TODO

Check docker process for various platforms
