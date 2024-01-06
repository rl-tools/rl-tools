FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y cmake build-essential libglfw3-dev libhdf5-dev libboost-all-dev git # git is required for MuJoCo (it uses "findOrFetch" in its CMakeLists.txt)
RUN apt-get update && apt-get install -y libxinerama-dev libxcursor-dev libxi-dev
RUN mkdir build
WORKDIR /build
