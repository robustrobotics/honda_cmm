# Use an official Python runtime as a parent image
FROM nvidia/cuda:9.0-runtime-ubuntu16.04

# Copy requirements contents into the container for installation
COPY requirements.txt .

# Install any needed packages specified in requirements.txt in order
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pip
RUN apt-get install -y git
RUN xargs -n1 pip3 install --trusted-host pypi.python.org < requirements.txt

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget https://dl.min.io/client/mc/release/linux-amd64/mc && chmod +x mc
COPY run.sh .
COPY run_active_train.sh .
