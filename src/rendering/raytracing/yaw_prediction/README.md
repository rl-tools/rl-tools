```
./build/src/rendering/raytracing/yaw_prediction/rendering_raytracing_yaw_prediction --scene-dir /home/jonas/git/ai2thor-hab/glb --num-scenes 80 --num-scenes-per-batch 8 --num-iterations 1000000
```

### Big Run
```
CUDACXX=/home/jonas/.local/opt/cuda-12.8.2/bin/nvcc cmake -B build   -DCMAKE_BUILD_TYPE=Release   -DRL_TOOLS_ENABLE_TESTS=ON   -DRL_TOOLS_EXPERIMENTAL=OFF   -DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO=ON   -DRL_TOOLS_NUMERIC_TYPES_ENABLE_BF16=ON   -DRL_TOOLS_ENABLE_TAR=ON -DRL_TOOLS_RENDERING_ENABLE_RAYTRACING=ON -DCMAKE_PREFIX_PATH=/home/jonas/.local/opt/cudnn
```

```
ulimit -n 65536 && ./build/src/rendering/raytracing/yaw_prediction/rendering_raytracing_yaw_prediction --scene-dir /scr/jonas/ai2thor-hab/glb --num-load-threads 16 --num-scenes 1000 --num-val-scenes 100 --num-iterations 10000000
```
