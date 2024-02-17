### This example shows the deep learning capabilities of the RLtools library
#### Here we are training a fully connected neural network to classify the MNIST handwritten digit dataset to reach > 90% accuracy
You can verify the lack of dependencies in the [Dockerfile](Dockerfile) used for this example.
```
cd examples/docker/00_basic_mnist
```
Build the docker image (tagged as `rl_tools:basic_mnist`)
```
docker build -f Dockerfile -t rl_tools:basic_mnist . --no-cache
```
Run the image just built (mounting the source code from the host in readonly mode)
```
docker run --rm -it --mount type=bind,source=$(cd ../../..; pwd),target=/rl_tools,readonly rl_tools:basic_mnist
```
Configure the RLtools using CMake (we enable HDF5 because the MNIST dataset is stored in the HDF5 format)
```
cmake /rl_tools/ -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_ENABLE_TARGETS:BOOL=ON -DRL_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DRL_TOOLS_ENABLE_HDF5:BOOL=ON
```
Build the nn_mnist_training target which trains to classify the MNIST digit dataset
```
cmake --build . --target nn_mnist_training -j$(nproc)
```
Run the built binary. This will train a fully connected neural network with two hidden layers of 50 neurons with ReLU activations each. The input are flattened `28*28=784` dimensional images from the MNIST handwritten digit dataset. After the epoch we classify and print a number of examples from the held-out validation set on the console. The input is displayed using ASCII art which consists of the predicted class (digit) of the trained classifier.
```
RL_TOOLS_NN_MNIST_DATA_FILE=mnist.hdf5 ./src/nn/mnist/nn_mnist_training
```

You should see plenty of outputs similar to this one classifying the handwritten four as a four (scroll up to see more):
```
                                                        
                                                        
               4 4                 4 4                  
               4 4                 4 4 4                
               4 4                 4 4 4                
               4 4 4               4 4 4 4              
               4 4 4               4 4 4 4              
               4 4 4               4 4 4 4              
               4 4 4               4 4 4 4 4 4          
               4 4 4               4 4 4 4 4 4          
               4 4 4               4 4 4 4 4 4          
               4 4 4             4 4 4 4 4 4            
               4 4 4 4 4 4 4 4 4 4 4 4 4 4              
               4 4 4 4 4 4 4 4 4 4 4 4 4 4              
                 4 4 4 4 4 4 4 4 4 4 4 4                
                   4 4 4 4 4       4 4 4                
                                   4 4 4                
                                   4 4 4                
                                   4 4 4                
                                   4 4 4                
                                   4 4 4                
                                   4 4 4                
                                                        
                                                        
```
