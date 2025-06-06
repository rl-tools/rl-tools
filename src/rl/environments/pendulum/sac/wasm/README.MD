## Docker
```
cd src/rl/environments/pendulum/sac/wasm
```
```
docker build -f ../../../../../../examples/docker/05_wasm/Dockerfile -t rltools/wasm_build . --no-cache
```
```
docker run --rm -it --mount type=bind,source=$(cd ../../../../../..; pwd),target=/rl_tools,readonly --mount type=bind,source=$(cd ../../../../../..; pwd)/static/rl.tools/build,target=/build --platform linux/amd64 --entrypoint /bin/bash  rltools/wasm_build
```
```
RL_TOOLS_INCLUDE_DIR=/rl_tools/include RL_TOOLS_BUILD_DIR=/build /rl_tools/src/rl/environments/pendulum/sac/wasm/build.sh
```

## Native
```
source ~/git/emsdk/emsdk_env.sh
```
```
mkdir build
./build.sh
```

## Host
```
git submodule update --init -- static/media
```
```
python3 -m http.server --directory static
```

## Deploy
```
scp -r static/* server3.jonas.es:DockerContainer/rl_tools/static/
```