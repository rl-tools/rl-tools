## Docker Examples
### Mandatory
The most deterministic way to get started using BackpropTools is to use Docker. Mandatory prerequesite for this is:
1. Docker is installed ([official installation instructions](https://docs.docker.com/engine/install/))

### Optional
Furthermore, if you want to use GPU acceleration you need the NVIDIA Docker runtime to be able to acceess the GPU from within the container
1. NVIDIA Container Toolkit ([official installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
2. And an up to date NVIDIA driver
    * In the examples we are e.g. using CUDA 11.x hence a driver >= 450.80.02 is required

For each example we provide a readme file with a handful of commands to build the container, start it and to build and run the code inside. In our view this is prefarable to hiding the commands in a script because by exposing them the user can understand the different steps, has more agency and can directly start tinkering with the code.

Once these prequesites are installed you can continue with the instructions for the particular examples:

### [00 - Basic MNIST](00_basic_mnist/README.MD): Training a fully connected neural network to classify handwritten digits with > 90% accuracy
### [01 - Basic RL](01_basic_rl/README.MD): Dependency-free RL example
### [02 - MKL](02_mkl/README.MD): Examples running on CPU but accelerated using Intel MKL
### [03 - CUDA](03_cuda/README.MD): Examples running on the GPU
### [04 - MuJoCo UI](04_mujoco_ui/README.MD): Using checkpoints saved from the previous example to visualize the trained policy in the MuJoCo Simulator
### [05 - WASM](05_wasm/README.MD): Examples running in the browser using WebAssembly
