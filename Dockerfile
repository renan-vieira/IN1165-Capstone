FROM pytorch/pytorch:latest

MAINTAINER Renan Vieira && Gabriel Guimarães

# Set the working directory to /app
WORKDIR /app
# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN cat requirements.txt | xargs -n 1 pip3 install

# install libsndfile
RUN apt-get update
RUN apt-get install libsndfile1 -y

# Jupyter and Tensorboard ports
EXPOSE 8888 6006

