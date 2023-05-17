## Docker
```
cd src/rl/environments/pendulum/td3/wasm
```
```
docker build -f ../../../../../../examples/docker/05_wasm/Dockerfile -t backprop_tools:wasm_build .
```
```
docker run --rm -it -p 8000:8000 --mount type=bind,source=$(cd ../../../../../..; pwd),target=/backprop_tools,readonly --mount type=bind,source=$(pwd)/static/build,target=/build backprop_tools:wasm
```
```
BACKPROP_TOOLS_INCLUDE_DIR=/backprop_tools/include BACKPROP_TOOLS_BUILD_DIR=/build /backprop_tools/src/rl/environments/pendulum/td3/wasm/build.sh
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
python3 -m http.server --directory static
```

## Deploy
```
scp -r static/* server3.jonas.es:DockerContainer/backprop_tools/static/
```