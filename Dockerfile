# Use an official Python runtime as a parent image
FROM nvidia/cuda:9.0-runtime-ubuntu16.04

# Set the working directory
WORKDIR /honda_cmm

# Copy requirements contents into the container for installation
COPY requirements.txt /honda_cmm

# Install any needed packages specified in requirements.txt in order
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pip
RUN apt-get install -y git
RUN xargs -n1 pip3 install --trusted-host pypi.python.org < requirements.txt

# volume where host honda_cmm repo will be mounted
VOLUME /honda_cmm

# run with the following command
# docker run --runtime=nvidia -itv PATH_TO_HONDA_CMM_REPO:/honda_cmm IMAGE bash
# if in honda_cmm directory, can replace PATH_TO_HONDA_CMM_REPO with $(pwd)
