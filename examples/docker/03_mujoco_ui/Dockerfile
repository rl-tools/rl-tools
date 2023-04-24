FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y cmake build-essential
RUN apt-get update && apt-get install -y libglfw3-dev
RUN apt-get install -y git
RUN apt-get install -y libhdf5-dev libboost-all-dev
RUN mkdir build
WORKDIR /build
