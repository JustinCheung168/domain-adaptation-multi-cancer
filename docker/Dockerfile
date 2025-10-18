# Base our Docker image on an official PyTorch-provided one.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime 

# Set the working directory of the image build.
WORKDIR /repo/

# Copy necessary files from the build context to the working directory.
COPY requirements/linux-requirements.txt requirements/python-environment.yml ./

# Install useful Linux packages.
# Remove apt's cached package lists at the end to reduce final image size.
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list && \
    apt-get update && \
    sed -i 's/\r$//' linux-requirements.txt && \
    xargs -a linux-requirements.txt apt-get install -y && \
    rm -rf /var/lib/apt/lists/*

# Install useful Python packages.
RUN conda env update -n base -f python-environment.yml

# By default, running a container based on this image will open JupyterLab.
# `--ip=0.0.0.0` is needed to let you use the JupyterLab server running the container at localhost.
# `--allow-root` allows JupyterLab to open if the Docker container is root
#   (which currently is the case, since this Dockerfile provides no USER instruction).
# `--no-browser` prevents the browser from trying to open inside the container.
#   You would instead want to open JupyterLab on your local computer.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
