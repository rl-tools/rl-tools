Change to the current directory
```
cd examples/docker/05_wasm
```
Build the docker image (which is tagged as `rl_tools:wasm`)
```
docker build -f Dockerfile -t rl_tools:wasm . --no-cache
````
Run the image just built (exposing port 8000 for the web server and mounting the source code from the host in readonly mode)
```
docker run --rm -it -p 8000:8000 --mount type=bind,source=$(cd ../../..; pwd),target=/rl_tools,readonly rl_tools:wasm
```
Copy the static files needed for the website
```
cp -r /rl_tools/src/rl/environments/pendulum/sac/wasm/static .
```
Build the WASM code (which is placed in the static directory where the other files of the website expect it)
```
RL_TOOLS_INCLUDE_DIR=/rl_tools/include RL_TOOLS_BUILD_DIR=/build/static/build /rl_tools/src/rl/environments/pendulum/sac/wasm/build.sh
```
Run a basic http server to serve the website
```
python3 -m http.server --directory static
```
On your host machine, navigate to http://localhost:8000/ using your web browser and enjoy the Pendulum-v1 learning to swing up using SAC 
