```
./build/src/rendering/raytracing/yaw_prediction/rendering_raytracing_yaw_prediction --scene-dir /home/jonas/git/ai2thor-hab/glb --num-scenes 80 --num-scenes-per-batch 8 --num-iterations 1000000
```

### Big Run
```
CUDACXX=/home/jonas/.local/opt/cuda-12.8.2/bin/nvcc cmake -B build   -DCMAKE_BUILD_TYPE=Release   -DRL_TOOLS_DISABLE_EXPERIMENTAL=ON   -DCMAKE_PREFIX_PATH=/home/jonas/.local/opt/cudnn
```
```
cmake --build build -j16 --target rendering_raytracing_yaw_prediction_base rendering_raytracing_yaw_prediction_crossconv rendering_raytracing_yaw_prediction_base_2x rendering_raytracing_yaw_prediction_crossconv_2x rendering_raytracing_yaw_prediction_base_128 rendering_raytracing_yaw_prediction_crossconv_128
```

```
ulimit -n 65536 && ./build/src/rendering/raytracing/yaw_prediction/rendering_raytracing_yaw_prediction_base --scene-dir /scr/jonas/ai2thor-hab/glb --num-load-threads 16 --num-scenes 1000 --num-val-scenes 100 --num-iterations 2000000
```
