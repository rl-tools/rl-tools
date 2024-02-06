template<typename DEV_SPEC, typename SPEC, typename devices::CUDA<DEV_SPEC>::index_t BATCH_SIZE, typename devices::CUDA<DEV_SPEC>::index_t BLOCK_SIZE>
__global__ void
evaluate_batch_kernel(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC> layer, const typename SPEC::T* input, typename SPEC::T* output) {
    using T = typename SPEC::T;
    using TI = typename devices::CUDA<DEV_SPEC>::index_t;
    constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
    constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;

    __shared__ T shared_input[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ T shared_weights[BLOCK_SIZE * BLOCK_SIZE];

    assert(BLOCK_SIZE == blockDim.x);
    assert(BLOCK_SIZE == blockDim.y);

    TI block_output_pos = blockIdx.x * BLOCK_SIZE;
    TI block_batch_pos = blockIdx.y * BLOCK_SIZE;
    TI thread_output_pos = block_output_pos + threadIdx.y;
    TI thread_batch_pos = block_batch_pos + threadIdx.y;

    TI thread_block_index = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    TI print_block_idx = 0;
    TI print_block_idy = 1;
    TI print_thread_idx = 4;
    TI print_thread_idy = 0;

    T acc = 0;
    for(TI block_reduction_i = 0; block_reduction_i < RL_TOOLS_DEVICES_CUDA_CEIL(INPUT_DIM, BLOCK_SIZE) * BLOCK_SIZE; block_reduction_i += BLOCK_SIZE){
        TI thread_input_pos = block_reduction_i + threadIdx.x;
        if(thread_input_pos < INPUT_DIM && thread_batch_pos < BATCH_SIZE){
            shared_input[thread_block_index] = input[thread_batch_pos * INPUT_DIM + thread_input_pos];
        }
        else{
            shared_input[thread_block_index] = 0;
        }
        if(thread_input_pos < INPUT_DIM && thread_output_pos < OUTPUT_DIM){
            shared_weights[thread_block_index] = get(layer.weights, thread_output_pos, thread_input_pos);
        }
        else{
            shared_weights[thread_block_index] = 0;
        }
        __syncthreads();
//                if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                    printf("input:\n");
//                    for(TI i = 0; i < BLOCK_SIZE; i++){
//                        for(TI j = 0; j < BLOCK_SIZE; j++){
//                            printf(" %f", shared_input[i * BLOCK_SIZE + j]);
//                        }
//                        printf("\n");
//                    }
//                    printf("weights:\n");
//                    for(TI i = 0; i < BLOCK_SIZE; i++){
//                        for(TI j = 0; j < BLOCK_SIZE; j++){
//                            printf(" %f", shared_weights[i * BLOCK_SIZE + j]);
//                        }
//                        printf("\n");
//                    }
//                }
        // x: output, y: batch
        for(TI reduction_i = 0; reduction_i < BLOCK_SIZE; reduction_i++){
            T a = shared_weights[threadIdx.x * BLOCK_SIZE + reduction_i];
            T b = shared_input[threadIdx.y * BLOCK_SIZE + reduction_i];
            acc += a * b;
//                    if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                        printf("a: %f, b: %f\n", a, b);
//                    }
        }
    }
//            if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                printf("result: %f\n",  acc);
//            }
    T b = get(layer.biases, 0, blockIdx.x * BLOCK_SIZE + threadIdx.x);
    acc += b;
//            if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                printf("bias: %f\n",  b);
//            }
//            if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                printf("result: %f\n",  acc);
//            }
    acc = activation<typename devices::CUDA<DEV_SPEC>::SPEC::MATH, typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(acc);
//            if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                printf("result: %f\n",  acc);
//            }

    if(blockIdx.y * BLOCK_SIZE + threadIdx.y < BATCH_SIZE && blockIdx.x * BLOCK_SIZE + threadIdx.x < OUTPUT_DIM){
//                if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                    printf("writing %f to: %d %d\n", acc, blockIdx.y * BLOCK_SIZE + threadIdx.y, blockIdx.x * BLOCK_SIZE + threadIdx.x);
//                }
        output[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * OUTPUT_DIM + blockIdx.x * BLOCK_SIZE + threadIdx.x] = acc;
    }
}
// invocation code
//        {
//            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_BATCH = 32;
//            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_BATCH = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_BATCH);
//            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_OUTPUT = 32;
//            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_OUTPUT = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_OUTPUT);
//            dim3 grid(N_BLOCKS_OUTPUT, N_BLOCKS_BATCH);
//            dim3 block(BLOCKSIZE_OUTPUT, BLOCKSIZE_BATCH);
//            nn::dense::cuda::evaluate_batch_kernel<DEV_SPEC, LAYER_SPEC, BATCH_SIZE, BLOCKSIZE_BATCH><<<grid, block, 0, device.stream>>>(device, layer, input.data, output.data);
////          handle cuda error
//            cudaDeviceSynchronize();
//            auto err = cudaGetLastError();
//            if(err != cudaSuccess){
//                std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
//
//            }
//        }
