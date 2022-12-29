#!/usr/bin/env bash
set -e
# if folder context does not exist, create it
if [ ! -d context ]; then
    mkdir context
    cd context
    git clone --recursive https://git.jonas.es/studium/phd/projects/rl-for-multirotor-control/layer-in-c.git
    git clone --recursive https://git.jonas.es/studium/phd/multirotor-torch.git
    cd ..
fi
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
docker build -t layer_in_c -f base_dockerfile context
docker build -t layer_in_c:tests -f tests_dockerfile context

#--mount type=bind,source=$SCRIPTPATH/../..,target=/layer_in_c,ro
docker run -it --rm layer_in_c:tests
