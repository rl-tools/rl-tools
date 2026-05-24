Detect CUDA arch and optimize build
```
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j --target rl_environments_l2f_visual_yaw_cuda
/rl-tools/build/src/rl/environments/l2f_visual/rl_environments_l2f_visual_yaw_cuda /rl-tools/glb
```


```
watch -n 120 -- "rsync -aP --append-verify -e 'ssh -p 63240' root@70.30.158.46:/rl-tools/experiments/ ."
```
