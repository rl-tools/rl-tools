# Simple 2D Car Environment
GTK3 is required for the UI:
```
sudo apt-get install libgtk-3-dev
```
Optionally, if you want to try to drive the car using a gamepad, you can also install SDL2:
```
sudo apt-get install libsdl2-dev
```

In the root directory of the repo (switch the backend according to your platform, e.g. `MKL`, `ACCELERATE` or `OPEN_BLAS`):
```
mkdir build
cd build
MKL_ROOT=/opt/intel/oneapi/mkl/latest cmake .. -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_BACKEND_ENABLE_MKL=ON -DRL_TOOLS_ENABLE_GTK=ON -DRL_TOOLS_ENABLE_SDL2=ON
```

Then build the ui and training targets:
```
cmake --build . --target rl_environments_car_ui -j$(nproc)
```

Back to the root directory of the repo:
```
cd ..
./build/src/rl/environments/car/rl_environments_car_ui
```
Wiggle your gamepad sticks to see the axes ids and restart the ui with the correct ids:
```
./build/src/rl/environments/car/rl_environments_car_ui [throttle_id] [steering_id]
```
If the axis should be inverted, add a minus in front of the id.

To run the training execute:
```
./build/src/rl/environments/car/rl_environments_car_training
```


