# Base our Docker image on an official PyTorch-provided one.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime 

# Set the working directory of the image build.
WORKDIR /repo/

# Copy necessary files from the build context to the working directory.
COPY requirements/linux-requirements.txt ./

# Install useful Linux packages.
# Remove apt's cached package lists at the end to reduce final image size.
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list && \
    apt-get update && \
    sed -i 's/\r$//' linux-requirements.txt && \
    xargs -a linux-requirements.txt apt-get install -y && \
    rm -rf /var/lib/apt/lists/*

# Not including Python packages - run `pip install -e .` within container instead.