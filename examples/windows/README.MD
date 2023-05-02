## Windows Examples
### Basic Requirements
Install the following software:
* [git](https://git-scm.com/download/win)
  * You can use the default settings for all options
* [CMake](https://cmake.org/install/)
* [Visual Studio](https://visualstudio.microsoft.com/)
  * Select the component for Desktop C++ development
  * This installs the required C++ compiler

Using the MinGW based `Git Bash` terminal, clone the repository:
```
git clone --recursive https://github.com/BackpropTools/BackpropTools.git
```
```
cd BackpropTools
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --target rl_environments_pendulum_td3_training_standalone -j$(nproc)
./src/rl/environments/penduolum/td3/cpu/Release/rl_environments_pendulum_td3_training_standalone.exe
```


### Extended Requirements
The extended requirements can be mixed and matched as needed they are not necessarily needed for every example (depending on the CMake configuration flags used)

#### [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html) (tested with oneMKL 2023.1.0)
BLAS library that accelerates matrix multiplications on Intel CPUs
After installing MKL use `CMD.exe` to source the `setvars.bat` (which sets the search paths for the MKL libraries):
```
C:\Program Files (x86)\Intel\oneAPI\setvars.bat
```
Now configure the BackpropTools to use MKL:
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON
```
After that build and run the target as before:
```
cmake --build . --config Release --target rl_environments_pendulum_td3_training_standalone -j$(nproc)
```
And execute the built binary:
```
src\rl\environments\pendulum\td3\cpu\Release\rl_environments_pendulum_td3_training_standalone.exe
```

* [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) (we tested this on CUDA 12.1)
* [Protocol Buffers](https://github.com/protocolbuffers/protobuf/releases/tag/v22.3)
    * Data exchange format used by Tensorboard
    * Download binaries for windows
    * Unzip and move to `C:\protoc` (so that the `protoc.exe` is located at `C:\protoc\bin\protoc.exe`)