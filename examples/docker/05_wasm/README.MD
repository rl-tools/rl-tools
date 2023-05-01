Change to the current directory
```
cd backprop_tools/examples/docker/05_wasm
```
Build the docker image (which is tagged as `backprop_tools:wasm`)
```
./build.sh
```
Run the image just built (exposing port 8000 for the web server and mounting the source code from the host in readonly mode)
```
docker run --rm -it -p 8000:8000 --mount type=bind,source=$(cd ../../..; pwd),target=/backprop_tools,readonly backprop_tools:wasm
```
Copy the static files needed for the website
```
cp -r /backprop_tools/src/rl/environments/pendulum/td3/wasm/static .
```
Build the WASM code (which is placed in the static directory where the other files of the website expect it)
```
BACKPROP_TOOLS_INCLUDE_DIR=/backprop_tools/include BACKPROP_TOOLS_BUILD_DIR=/build/static/build /backprop_tools/src/rl/environments/pendulum/td3/wasm/build.sh
```
Run a basic http server to serve the website
```
python3 -m http.server --directory static
```
On your host machine, navigate to http://localhost:8000/ using your web browser and enjoy the Pendulum-v1 learning to swing up using TD3
