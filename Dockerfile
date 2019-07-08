# Use an official Python runtime as a parent image
FROM python:3.6

# Set the working directory
WORKDIR /honda_cmm

# Copy requirements contents into the container for installation
COPY requirements.txt /honda_cmm

# Install any needed packages specified in requirements.txt in order
RUN xargs -n1 pip3 install --trusted-host pypi.python.org < requirements.txt

# volume where host honda_cmm repo will be mounted
VOLUME /honda_cmm

# run with the following command
# docker run -itv PATH_TO_HONDA_CMM_REPO:/honda_cmm IMAGE bash
# if in honda_cmm directory, can replace PATH_TO_HONDA_CMM_REPO with $(pwd)
