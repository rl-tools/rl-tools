Change to the current directory
```
cd examples/docker/02_mkl
```
Build the docker image (saved as `rl_tools:mkl`)
```
docker build -f Dockerfile -t rl_tools:mkl . --no-cache
```
Run a docker container based on the image just built (exposing port `6006` for Tensorboard and mounting the source code from the host in readonly mode)
```
docker run --rm -it -p 6006:6006 --mount type=bind,source=$(cd ../../..; pwd),target=/rl_tools,readonly --cap-add=sys_nice rl_tools:mkl
```
Configure RLtools using CMake
```
cmake /rl_tools/ -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_ENABLE_TARGETS:BOOL=ON -DRL_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DRL_TOOLS_ENABLE_HDF5:BOOL=ON -DRL_TOOLS_ENABLE_TENSORBOARD:BOOL=ON
```
Building the targets
```
cmake --build . -j$(nproc)
```
Execute the built binary that uses MKL (indicated by the `_blas`) with a high priority
```
nice -n -20 ./src/rl/environments/pendulum/td3/cpu/rl_environments_pendulum_td3_blas
```
Alternatively: You can run the binary including Tensorboard logging (which might slow down the training). For that start tensorboard in the background first:
```
tensorboard --logdir runs --bind_all 2> /dev/null > /dev/null &
```
Then run the executable:
```
nice -n -20 ./src/rl/environments/pendulum/td3/cpu/rl_environments_pendulum_td3_blas_tensorboard
```
You can observe the training using the browser on your host machine to access Tensorboard at `http://localhost:6006/


Expected output:
```
Mean return: -1269.58
step_i: 1000 0.275554s
Mean return: -1677.3
step_i: 2000 0.632844s
Mean return: -1341.02
step_i: 3000 1.00824s
Mean return: -1224.27
step_i: 4000 1.38705s
Mean return: -1018.41
step_i: 5000 1.77338s
Mean return: -681.685
step_i: 6000 2.1696s
Mean return: -250.108
step_i: 7000 2.56791s
Mean return: -157.664
step_i: 8000 2.9712s
Mean return: -98.1297
step_i: 9000 3.37329s
Mean return: -196.552
total time: 3.79653s
```

Compared to the example without vectorized CPU acceleration (MKL) the training should be much faster but yield similar returns. The returns are numerically different because for floating point numbers e.g. the order of operations matters.