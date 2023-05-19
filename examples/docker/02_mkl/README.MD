Change to the current directory
```
cd examples/docker/02_mkl
```
Build the docker image (saved as `backprop_tools:mkl`)
```
docker build -f Dockerfile -t backprop_tools:mkl . --no-cache
```
Run a docker container based on the image just built (exposing port `6006` for Tensorboard and mounting the source code from the host in readonly mode)
```
docker run --rm -it -p 6006:6006 --mount type=bind,source=$(cd ../../..; pwd),target=/backprop_tools,readonly --cap-add=sys_nice backprop_tools:mkl
```
Configure BackpropTools using CMake
```
cmake /backprop_tools/ -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON
```
Building the targets
```
cmake --build . -j$(nproc)
```
Execute the built binary that uses MKL (indicated by the `_blas`) with a high priority
```
nice -n -20 ./src/rl/environments/pendulum/td3/cpu/rl_environments_pendulum_td3_training_blas
```
Alternatively: You can run the binary including Tensorboard logging (which might slow down the training). For that start tensorboard in the background first:
```
tensorboard --logdir logs --bind_all 2> /dev/null > /dev/null &
```
Then run the executable:
```
nice -n -20 ./src/rl/environments/pendulum/td3/cpu/rl_environments_pendulum_td3_training_blas_tensorboard
```
You can observe the training using the browser on your host machine to access Tensorboard at `http://localhost:6006/
