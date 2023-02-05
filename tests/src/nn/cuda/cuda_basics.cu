// Group 1
#include <layer_in_c/operations/cpu/group_1.h>
#include <layer_in_c/operations/cuda/group_1.h>

// Group 2
#include <layer_in_c/operations/cpu/group_2.h>
#include <layer_in_c/operations/cuda/group_2.h>

// Group 3
#include <layer_in_c/operations/cpu/group_3.h>
#include <layer_in_c/operations/cuda/group_3.h>

#include <layer_in_c/nn/operations_cuda.h>
#include <layer_in_c/nn/loss_functions/mse/operations_cuda.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/nn_models/operations_cpu.h>

namespace lic = layer_in_c;

#include <gtest/gtest.h>

namespace copy{
    using DTYPE = float;
    using DEVICE_CPU = lic::devices::DefaultCPU;
    using DEVICE_CUDA = lic::devices::DefaultCUDA;

    constexpr DEVICE_CPU::index_t BATCH_SIZE = 100;
    constexpr DEVICE_CPU::index_t HIDDEN_DIM = BATCH_SIZE;

    template <typename T, typename TI, lic::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
    using StructureSpecification = lic::nn_models::mlp::StructureSpecification<T, TI, HIDDEN_DIM, HIDDEN_DIM, 3, HIDDEN_DIM, ACTIVATION_FUNCTION, ACTIVATION_FUNCTION, BATCH_SIZE>;

    template <typename T, typename TI, lic::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
    using InferenceSpecification = lic::nn_models::mlp::AdamSpecification<StructureSpecification<T, TI, ACTIVATION_FUNCTION>, lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;

    constexpr DEVICE_CPU::index_t ITERATIONS = 1;
    constexpr DEVICE_CPU::index_t NAIVE_ITERATIONS = 1;
}

TEST(LAYER_IN_C_NN_CUDA, COPY) {
    using NetworkTypeCPU = lic::nn_models::mlp::NeuralNetworkAdam<copy::InferenceSpecification<copy::DTYPE, copy::DEVICE_CPU::index_t, lic::nn::activation_functions::RELU>>;
    using NetworkTypeCUDA = lic::nn_models::mlp::NeuralNetworkAdam<copy::InferenceSpecification<copy::DTYPE, copy::DEVICE_CUDA::index_t, lic::nn::activation_functions::RELU>>;
    copy::DEVICE_CPU::SPEC::LOGGING cpu_logger;
    copy::DEVICE_CUDA::SPEC::LOGGING cuda_logger;
    copy::DEVICE_CPU device_cpu(cpu_logger);
    copy::DEVICE_CUDA device_cuda(cuda_logger);
    NetworkTypeCPU network_cpu;
    NetworkTypeCPU network_cpu_2;
    NetworkTypeCUDA network_cuda;
    lic::malloc(device_cpu, network_cpu);
    lic::malloc(device_cpu, network_cpu_2);
    lic::malloc(device_cuda, network_cuda);

    auto rng = lic::random::default_engine(copy::DEVICE_CPU::SPEC::RANDOM());

    lic::init_weights(device_cpu, network_cpu, rng);
    lic::init_weights(device_cpu, network_cpu_2, rng);
    auto cpu_network_diff = lic::abs_diff(device_cpu, network_cpu, network_cpu_2);
    std::cout << "CPU network diff: " << cpu_network_diff << std::endl;
    ASSERT_GT(cpu_network_diff, 0);

    lic::copy(device_cuda, device_cpu, network_cuda, network_cpu);
    lic::copy(device_cpu, device_cuda, network_cpu_2, network_cuda);
    auto cpu_network_diff_round_trip = lic::abs_diff(device_cpu, network_cpu, network_cpu_2);
    std::cout << "CPU network round-trip: " << cpu_network_diff_round_trip << std::endl;
    ASSERT_FLOAT_EQ(cpu_network_diff_round_trip, 0);

    network_cpu.hidden_layers[0].weights.data[50] += 5;
    std::cout << "CPU network weights: " << network_cpu.hidden_layers[0].weights.data[50] << std::endl;

    cpu_network_diff = lic::abs_diff(device_cpu, network_cpu, network_cpu_2);
    std::cout << "CPU network diff: " << cpu_network_diff << std::endl;
    ASSERT_FLOAT_EQ(cpu_network_diff, 5);

    lic::copy(device_cuda, device_cpu, network_cuda, network_cpu);
    lic::copy(device_cpu, device_cuda, network_cpu_2, network_cuda);
    std::cout << "CPU network weights: " << network_cpu_2.hidden_layers[0].weights.data[50] << std::endl;
    cpu_network_diff_round_trip = lic::abs_diff(device_cpu, network_cpu, network_cpu_2);
    ASSERT_FLOAT_EQ(cpu_network_diff_round_trip, 0);
    std::cout << "CPU network round-trip: " << cpu_network_diff_round_trip << std::endl;

    lic::free(device_cpu, network_cpu);
    lic::free(device_cpu, network_cpu_2);
    lic::free(device_cuda, network_cuda);
}

template <typename T, typename TI, TI BATCH_SIZE, TI ITERATIONS>
void GEMM() {
    using DEVICE_CPU = lic::devices::DefaultCPU;
    using DEVICE_CUDA = lic::devices::DefaultCUDA;

    constexpr DEVICE_CPU::index_t HIDDEN_DIM = BATCH_SIZE;

    constexpr auto ACTIVATION_FUNCTION = lic::nn::activation_functions::IDENTITY;
    using StructureSpecification = lic::nn_models::mlp::StructureSpecification<T, TI, HIDDEN_DIM, HIDDEN_DIM, 3, HIDDEN_DIM, ACTIVATION_FUNCTION, lic::nn::activation_functions::RELU, BATCH_SIZE>;

    using NNSpecification = lic::nn_models::mlp::AdamSpecification<StructureSpecification, lic::nn::optimizers::adam::DefaultParametersTorch<T>>;

    std::cout << "GEMM<" << (lic::utils::typing::is_same_v<T, float> ? "float" : "double") << ", " << BATCH_SIZE << ">" << std::endl;
    using NetworkTypeCPU = lic::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    using NetworkTypeCUDA = lic::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    DEVICE_CPU::SPEC::LOGGING cpu_logger;
    DEVICE_CUDA::SPEC::LOGGING cuda_logger;
    DEVICE_CPU device_cpu(cpu_logger);
    DEVICE_CUDA device_cuda(cuda_logger);
    lic::init(device_cuda);
    NetworkTypeCPU network_cpu;
    typename NetworkTypeCPU::template Buffers<BATCH_SIZE> network_cpu_buffers;
    NetworkTypeCUDA network_cuda;
    typename NetworkTypeCPU::template Buffers<BATCH_SIZE> network_cuda_buffers;
    lic::malloc(device_cpu, network_cpu);
    lic::malloc(device_cpu, network_cpu_buffers);
    lic::malloc(device_cuda, network_cuda);
    lic::malloc(device_cpu, network_cuda_buffers);

    auto rng = lic::random::default_engine(DEVICE_CPU::SPEC::RANDOM());

    lic::init_weights(device_cpu, network_cpu, rng);
    lic::copy(device_cuda, device_cpu, network_cuda, network_cpu);

    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cpu;
    lic::malloc(device_cpu, input_cpu);
    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM>> output_first_layer_cpu;
    lic::malloc(device_cpu, output_first_layer_cpu);
    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM>> output_first_layer_cuda_cpu;
    lic::malloc(device_cpu, output_first_layer_cuda_cpu);

    for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE * NetworkTypeCPU::INPUT_DIM; ++i)
    {
        input_cpu.data[i] = lic::random::normal_distribution(DEVICE_CPU::SPEC::RANDOM(), (T)0, (T)1, rng);
    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Input:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::INPUT_DIM; ++j)
//            {
//                std::cout << input_cpu.data[i * NetworkTypeCPU::INPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Weights:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::INPUT_DIM; ++j)
//            {
//                std::cout << network_cpu.input_layer.weights.data[i * NetworkTypeCPU::INPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Biases:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM; ++i)
//        {
//            std::cout << network_cpu.input_layer.biases.data[i] << " ";
//        }
//        std::cout << std::endl;
//    }


    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cuda;
    lic::malloc(device_cuda, input_cuda);
    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM>> output_first_layer_cuda;
    lic::malloc(device_cuda, output_first_layer_cuda);

    lic::copy(device_cuda, device_cpu, input_cuda, input_cpu);

    lic::evaluate(device_cpu, network_cpu.input_layer, input_cpu, output_first_layer_cpu);
    lic::evaluate(device_cuda, network_cuda.input_layer, input_cuda, output_first_layer_cuda);
    cudaDeviceSynchronize();

    lic::copy(device_cpu, device_cuda, output_first_layer_cuda_cpu, output_first_layer_cuda);
    auto evaluation_diff = lic::abs_diff(device_cpu, output_first_layer_cuda_cpu, output_first_layer_cpu)/(BATCH_SIZE * NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM);

//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM <= 10){
//        std::cout << "cpu output:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM; ++j)
//            {
//                std::cout << output_first_layer_cpu.data[i * NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM <= 10){
//        std::cout << "cuda output:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i){
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM; ++j){
//                std::cout << output_first_layer_cuda_cpu.data[i * NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM <= 10){
//        std::cout << "cuda diff:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM; ++j)
//            {
//                T diff = output_first_layer_cpu.data[i * NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM + j] - output_first_layer_cuda_cpu.data[i * NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM + j];
//                diff = std::abs(diff) > 1e-7 ? diff : 0;
//                std::cout << diff << " ";
//            }
//            std::cout << std::endl;
//        }
//    }

    std::cout << "Evaluation diff: " << evaluation_diff << std::endl;
    auto threshold = (lic::utils::typing::is_same_v<T, float> ? 1e-6 : 1e-15);
    if(evaluation_diff > threshold){
        ASSERT_LT(evaluation_diff, threshold);
    }

    {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < ITERATIONS; ++i)
        {
            lic::evaluate(device_cuda, network_cuda.input_layer, input_cuda, output_first_layer_cuda);
            cudaDeviceSynchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "CUDA evaluation time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((T)ITERATIONS) << "us" << std::endl;
    }
}
//TEST(LAYER_IN_C_NN_CUDA, GEMM) {
//    using DEFAULT_DTYPE = float;
//    GEMM<DEFAULT_DTYPE, unsigned int, 1, 1>();
//    GEMM<DEFAULT_DTYPE, unsigned int, 2, 1>();
//    GEMM<DEFAULT_DTYPE, unsigned int, 32, 1>();
////    GEMM<DEFAULT_DTYPE, unsigned int, 1024, 1>();
//    GEMM<DEFAULT_DTYPE, unsigned int, 10, 1>();
//    GEMM<DEFAULT_DTYPE, unsigned int, 9, 1>();
//    GEMM<double, unsigned int, 200, 1>();
//    GEMM<DEFAULT_DTYPE, unsigned int, 200, 1>();
//    GEMM<DEFAULT_DTYPE, unsigned int, 64, 1000>();
//    GEMM<DEFAULT_DTYPE, unsigned int, 256, 1000>();
//}

template <typename T, typename TI, TI BATCH_SIZE, TI ITERATIONS>
void FORWARD() {
    using DEVICE_CPU = lic::devices::DefaultCPU;
    using DEVICE_CUDA = lic::devices::DefaultCUDA;

    constexpr DEVICE_CPU::index_t HIDDEN_DIM = BATCH_SIZE;

    constexpr auto ACTIVATION_FUNCTION = lic::nn::activation_functions::IDENTITY;
    using StructureSpecification = lic::nn_models::mlp::StructureSpecification<T, TI, HIDDEN_DIM, HIDDEN_DIM, 3, HIDDEN_DIM, ACTIVATION_FUNCTION, lic::nn::activation_functions::RELU, BATCH_SIZE>;

    using NNSpecification = lic::nn_models::mlp::AdamSpecification<StructureSpecification, lic::nn::optimizers::adam::DefaultParametersTorch<T>>;

    std::cout << "GEMM<" << (lic::utils::typing::is_same_v<T, float> ? "float" : "double") << ", " << BATCH_SIZE << ">" << std::endl;
    using NetworkTypeCPU = lic::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    using NetworkTypeCUDA = lic::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    DEVICE_CPU::SPEC::LOGGING cpu_logger;
    DEVICE_CUDA::SPEC::LOGGING cuda_logger;
    DEVICE_CPU device_cpu(cpu_logger);
    DEVICE_CUDA device_cuda(cuda_logger);
    lic::init(device_cuda);
    NetworkTypeCPU network_cpu;
    typename NetworkTypeCPU::template Buffers<BATCH_SIZE> network_cpu_buffers;
    NetworkTypeCUDA network_cuda;
    typename NetworkTypeCPU::template Buffers<BATCH_SIZE> network_cuda_buffers;
    lic::malloc(device_cpu, network_cpu);
    lic::malloc(device_cpu, network_cpu_buffers);
    lic::malloc(device_cuda, network_cuda);
    lic::malloc(device_cpu, network_cuda_buffers);

    auto rng = lic::random::default_engine(DEVICE_CPU::SPEC::RANDOM());

    lic::init_weights(device_cpu, network_cpu, rng);
    lic::copy(device_cuda, device_cpu, network_cuda, network_cpu);

    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cpu;
    lic::malloc(device_cpu, input_cpu);
    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cpu;
    lic::malloc(device_cpu, output_cpu);
    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cuda_cpu;
    lic::malloc(device_cpu, output_cuda_cpu);

    for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE * NetworkTypeCPU::INPUT_DIM; ++i)
    {
        input_cpu.data[i] = lic::random::normal_distribution(DEVICE_CPU::SPEC::RANDOM(), (T)0, (T)1, rng);
    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Input:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::INPUT_DIM; ++j)
//            {
//                std::cout << input_cpu.data[i * NetworkTypeCPU::INPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Weights:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < NetworkTypeCPU::OUTPUT_DIM; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::INPUT_DIM; ++j)
//            {
//                std::cout << network_cpu.input_layer.weights.data[i * NetworkTypeCPU::INPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Biases:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < NetworkTypeCPU::OUTPUT_DIM; ++i)
//        {
//            std::cout << network_cpu.input_layer.biases.data[i] << " ";
//        }
//        std::cout << std::endl;
//    }


    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cuda;
    lic::malloc(device_cuda, input_cuda);
    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cuda;
    lic::malloc(device_cuda, output_cuda);

    lic::copy(device_cuda, device_cpu, input_cuda, input_cpu);

    lic::forward(device_cpu, network_cpu, input_cpu, output_cpu);
    lic::forward(device_cuda, network_cuda, input_cuda, output_cuda);
    cudaDeviceSynchronize();

    lic::copy(device_cpu, device_cuda, output_cuda_cpu, output_cuda);
    auto evaluation_diff = lic::abs_diff(device_cpu, output_cuda_cpu, output_cpu)/(BATCH_SIZE * NetworkTypeCPU::OUTPUT_DIM);

//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::OUTPUT_DIM <= 10){
//        std::cout << "cpu output:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::OUTPUT_DIM; ++j)
//            {
//                std::cout << output_cpu.data[i * NetworkTypeCPU::OUTPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::OUTPUT_DIM <= 10){
//        std::cout << "cuda output:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i){
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::OUTPUT_DIM; ++j){
//                std::cout << output_cuda_cpu.data[i * NetworkTypeCPU::OUTPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::OUTPUT_DIM <= 10){
//        std::cout << "cuda diff:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::OUTPUT_DIM; ++j)
//            {
//                T diff = output_cpu.data[i * NetworkTypeCPU::OUTPUT_DIM + j] - output_cuda_cpu.data[i * NetworkTypeCPU::OUTPUT_DIM + j];
//                diff = std::abs(diff) > 1e-7 ? diff : 0;
//                std::cout << diff << " ";
//            }
//            std::cout << std::endl;
//        }
//    }

    std::cout << "Evaluation diff: " << evaluation_diff << std::endl;
    auto threshold = (lic::utils::typing::is_same_v<T, float> ? 1e-7 : 1e-15);
    if(evaluation_diff > threshold){
        ASSERT_LT(evaluation_diff, threshold);
    }

    {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < ITERATIONS; ++i)
        {
            lic::evaluate(device_cuda, network_cuda.input_layer, input_cuda, output_cuda);
            cudaDeviceSynchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "CUDA evaluation time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((T)ITERATIONS) << "us" << std::endl;
    }
}

//TEST(LAYER_IN_C_NN_CUDA, FORWARD) {
//    FORWARD<float, unsigned int, 1, 1>();
//    FORWARD<float, unsigned int, 2, 1>();
//    FORWARD<float, unsigned int, 32, 1>();
////    FORWARD<float, unsigned int, 1024, 1>();
//    FORWARD<float, unsigned int, 10, 1>();
//    FORWARD<float, unsigned int, 9, 1>();
//    FORWARD<double, unsigned int, 200, 1>();
//    FORWARD<float, unsigned int, 200, 1>();
//    FORWARD<float, unsigned int, 64, 10000>();
//    FORWARD<float, unsigned int, 256, 100000>();
//}

template <typename T, typename TI, TI BATCH_SIZE, TI ITERATIONS>
void BACKWARD() {
    using DEVICE_CPU = lic::devices::DefaultCPU;
    using DEVICE_CUDA = lic::devices::DefaultCUDA;

    constexpr DEVICE_CPU::index_t HIDDEN_DIM = BATCH_SIZE;

    constexpr auto ACTIVATION_FUNCTION = lic::nn::activation_functions::IDENTITY;
    using StructureSpecification = lic::nn_models::mlp::StructureSpecification<T, TI, HIDDEN_DIM, HIDDEN_DIM, 3, HIDDEN_DIM, ACTIVATION_FUNCTION, lic::nn::activation_functions::RELU, BATCH_SIZE>;

    using NNSpecification = lic::nn_models::mlp::AdamSpecification<StructureSpecification, lic::nn::optimizers::adam::DefaultParametersTorch<T>>;

    std::cout << "GEMM<" << (lic::utils::typing::is_same_v<T, float> ? "float" : "double") << ", " << BATCH_SIZE << ">" << std::endl;
    using NetworkTypeCPU = lic::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    using NetworkTypeCUDA = lic::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    DEVICE_CPU::SPEC::LOGGING cpu_logger;
    DEVICE_CUDA::SPEC::LOGGING cuda_logger;
    DEVICE_CPU device_cpu(cpu_logger);
    DEVICE_CUDA device_cuda(cuda_logger);
    lic::init(device_cuda);
    NetworkTypeCPU network_cpu;
    NetworkTypeCPU network_cpu_pre;
    NetworkTypeCPU network_cuda_cpu;
    typename NetworkTypeCPU::template BuffersForwardBackward<BATCH_SIZE> network_cpu_buffers;
    NetworkTypeCUDA network_cuda;
    typename NetworkTypeCPU::template BuffersForwardBackward<BATCH_SIZE> network_cuda_buffers;
    lic::malloc(device_cpu, network_cpu);
    lic::malloc(device_cpu, network_cpu_pre);
    lic::malloc(device_cpu, network_cuda_cpu);
    lic::malloc(device_cpu, network_cpu_buffers);
    lic::malloc(device_cuda, network_cuda);
    lic::malloc(device_cuda, network_cuda_buffers);

    auto rng = lic::random::default_engine(DEVICE_CPU::SPEC::RANDOM());

    lic::init_weights(device_cpu, network_cpu, rng);
    lic::zero_gradient(device_cpu, network_cpu);
    lic::reset_optimizer_state(device_cpu, network_cpu);
    lic::copy(device_cpu, device_cpu, network_cpu_pre, network_cpu);

    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cpu;
    lic::malloc(device_cpu, input_cpu);
    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cpu;
    lic::malloc(device_cpu, output_cpu);
    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_target_cpu;
    lic::malloc(device_cpu, output_target_cpu);
    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cuda_cpu;
    lic::malloc(device_cpu, output_cuda_cpu);

    for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE * NetworkTypeCPU::INPUT_DIM; ++i)
    {
        input_cpu.data[i] = lic::random::normal_distribution(DEVICE_CPU::SPEC::RANDOM(), (T)0, (T)1, rng);
    }
    for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE * NetworkTypeCPU::OUTPUT_DIM; ++i)
    {
        output_target_cpu.data[i] = lic::random::normal_distribution(DEVICE_CPU::SPEC::RANDOM(), (T)0, (T)1, rng);
    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Input:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::INPUT_DIM; ++j)
//            {
//                std::cout << input_cpu.data[i * NetworkTypeCPU::INPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Weights:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < NetworkTypeCPU::OUTPUT_DIM; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::INPUT_DIM; ++j)
//            {
//                std::cout << network_cpu.input_layer.weights.data[i * NetworkTypeCPU::INPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Biases:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < NetworkTypeCPU::OUTPUT_DIM; ++i)
//        {
//            std::cout << network_cpu.input_layer.biases.data[i] << " ";
//        }
//        std::cout << std::endl;
//    }

    lic::forward_backward_mse(device_cpu, network_cpu, input_cpu, output_target_cpu, network_cpu_buffers);
    lic::copy(device_cuda, device_cpu, network_cuda, network_cpu);


    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cuda;
    lic::malloc(device_cuda, input_cuda);
    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cuda;
    lic::malloc(device_cuda, output_cuda);
    lic::Matrix<lic::MatrixSpecification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_target_cuda;
    lic::malloc(device_cuda, output_target_cuda);

    lic::copy(device_cuda, device_cpu, input_cuda, input_cpu);
    lic::copy(device_cuda, device_cpu, output_target_cuda, output_target_cpu);

    lic::zero_gradient(device_cpu, network_cpu);
    lic::zero_gradient(device_cuda, network_cuda);
    lic::forward_backward_mse(device_cpu, network_cpu, input_cpu, output_target_cpu, network_cpu_buffers);
    lic::forward_backward_mse(device_cuda, network_cuda, input_cuda, output_target_cuda, network_cuda_buffers);
    cudaDeviceSynchronize();

    lic::copy(device_cpu, device_cuda, network_cuda_cpu, network_cuda);
    auto evaluation_diff_pre = lic::abs_diff(device_cpu, network_cuda_cpu, network_cpu_pre)/(BATCH_SIZE * NetworkTypeCPU::OUTPUT_DIM);
    auto evaluation_diff = lic::abs_diff(device_cpu, network_cuda_cpu, network_cpu)/(BATCH_SIZE * NetworkTypeCPU::OUTPUT_DIM);

//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::OUTPUT_DIM <= 10){
//        std::cout << "cpu output:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::OUTPUT_DIM; ++j)
//            {
//                std::cout << output_cpu.data[i * NetworkTypeCPU::OUTPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::OUTPUT_DIM <= 10){
//        std::cout << "cuda output:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i){
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::OUTPUT_DIM; ++j){
//                std::cout << output_cuda_cpu.data[i * NetworkTypeCPU::OUTPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::OUTPUT_DIM <= 10){
//        std::cout << "cuda diff:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::OUTPUT_DIM; ++j)
//            {
//                T diff = output_cpu.data[i * NetworkTypeCPU::OUTPUT_DIM + j] - output_cuda_cpu.data[i * NetworkTypeCPU::OUTPUT_DIM + j];
//                diff = std::abs(diff) > 1e-7 ? diff : 0;
//                std::cout << diff << " ";
//            }
//            std::cout << std::endl;
//        }
//    }

    std::cout << "Evaluation diff: " << evaluation_diff << std::endl;
    auto threshold = (lic::utils::typing::is_same_v<T, float> ? 1e-6 : 1e-14);
    if(evaluation_diff > threshold){
        ASSERT_LT(evaluation_diff, threshold);
    }

    {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < ITERATIONS; ++i)
        {
            lic::evaluate(device_cuda, network_cuda.input_layer, input_cuda, output_cuda);
            cudaDeviceSynchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "CUDA evaluation time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((T)ITERATIONS) << "us" << std::endl;
    }
}

TEST(LAYER_IN_C_NN_CUDA, BACKWARD) {
    using DEFAULT_DTYPE = float;
    BACKWARD<DEFAULT_DTYPE, unsigned int, 1, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int, 2, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int, 32, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int, 1024, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int, 10, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int, 9, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int, 200, 1>();
    BACKWARD<double, unsigned int, 200, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int, 64, 10000>();
    BACKWARD<DEFAULT_DTYPE, unsigned int, 256, 100000>();
}
