

```
mkdir build_rl_tools
cd build_rl_tools
```
Maximal config
```
cmake ../../.. -DCMAKE_BUILD_TYPE=Release  -DRL_TOOLS_ENABLE_HDF5=ON  -DRL_TOOLS_ENABLE_BOOST_BEAST=ON -DRL_TOOLS_ENABLE_TENSORBOARD=ON -DRL_TOOLS_ENABLE_JSON=ON -DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO=ON -DRL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI=ON -DRL_TOOLS_BACKEND_ENABLE_ACCELERATE=ON -DRL_TOOLS_ENABLE_CLI11=ON
```
Miniamal config
```
cmake ../../..
```
```
cmake --build . -j8 
cmake --install . --prefix ../install
cd ..
```
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=$(pwd)/../install -DCMAKE_BUILD_TYPE=Release
cmake --build .
```
```
./user_test_pendulum
```