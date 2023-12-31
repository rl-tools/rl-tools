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
cmake --build . --target rl_environments_car_ui rl_environments_car_training -j$(nproc)
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
It should learn to navigate the corners after around `100000-200000` steps which should take a couple of minutes. The training is running in the background (which is why the UI freezes for some time) and every N steps the evaluation of the current policy is run and displayed in the UI. The hyperparameters and reward function are not tuned at all. I believe it could train much faster and more reliably with better tuning.

If you would like to change the layout of the track, you can modify the `.bmp` image in `src/rl/environments/track.bmp` using e.g. Gimp (all non-black pixels are converted to drivable space).


