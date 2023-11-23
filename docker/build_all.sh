set -e
#PLATFORM=linux/amd64
#OS=ubuntu
#OS_VERSION=20.04
#BACKEND=mkl
#TOOLCHAIN=gcc

build() {
  PLATFORM=$1
  OS=$2
  OS_VERSION=$3
  BACKEND=$4
  TOOLCHAIN=$5

  BACKEND_IMAGE_NAME=rltools/rltools:${OS}${OS_VERSION}_${BACKEND}_base
  TOOLCHAIN_IMAGE_NAME=rltools/rltools:${OS}${OS_VERSION}_${BACKEND}_${TOOLCHAIN}_base
  FINAL_IMAGE_NAME=rltools/rltools:${OS}${OS_VERSION}_${BACKEND}_${TOOLCHAIN}
  docker build -t ${BACKEND_IMAGE_NAME}     -f ${OS}/backend/Dockerfile.${BACKEND}     --build-arg OS=${OS} --build-arg OS_VERSION=${OS_VERSION} --platform ${PLATFORM} .
  docker build -t ${TOOLCHAIN_IMAGE_NAME}   -f ${OS}/toolchain/Dockerfile.${TOOLCHAIN} --build-arg OS=${OS} --build-arg OS_VERSION=${OS_VERSION} --build-arg BASE_IMAGE=${BACKEND_IMAGE_NAME} --platform ${PLATFORM} .
  docker build -t ${FINAL_IMAGE_NAME}       -f ${OS}/Dockerfile                        --build-arg OS=${OS} --build-arg OS_VERSION=${OS_VERSION} --build-arg BASE_IMAGE=${TOOLCHAIN_IMAGE_NAME} --platform ${PLATFORM} .
  docker build -t ${FINAL_IMAGE_NAME}_build -f ${OS}/Dockerfile.build                  --build-arg OS=${OS} --build-arg OS_VERSION=${OS_VERSION} --build-arg BASE_IMAGE=${FINAL_IMAGE_NAME} --platform ${PLATFORM} .
}

#build linux/amd64 ubuntu 20.04 mkl gcc
build linux/amd64 ubuntu 22.04 cuda gcc
#build linux/amd64 ubuntu 22.04 cuda_mkl gcc

#wait $pid1 $pid2 $pid3

echo final ${FINAL_IMAGE_NAME}
docker tag ${FINAL_IMAGE_NAME} rltools/rltools:latest


docker run -it --rm -v $(cd .. && pwd):/rl_tools ${FINAL_IMAGE_NAME}_build