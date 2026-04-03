#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/max_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>

#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/max_pool2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn/layers/flatten/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>

#include <rl_tools/persist/backends/hdf5/operations_cpu.h>

#include <iostream>
#include <string>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

template <typename DEVICE_T, typename MODEL, typename INPUT_SPEC, typename OUTPUT_SPEC>
void save_single_input_fixture(DEVICE_T& device, MODEL& model, rlt::Tensor<INPUT_SPEC>& input, rlt::Tensor<OUTPUT_SPEC>& output, const std::string& path){
    auto file = rl_tools::persist::backends::hdf5::File(path, rl_tools::persist::backends::hdf5::Mode::WRITE);
    auto actor_group = rlt::create_group(device, file, "actor");
    rlt::save(device, model, actor_group);
    auto example_group = rlt::create_group(device, file, "example");
    rlt::save(device, input, example_group, "input");
    rlt::save(device, output, example_group, "output");
    std::cout << "Saved fixture: " << path << std::endl;
}

void generate_conv2d_avgpool(DEVICE& device, const std::string& output_dir){
    constexpr TI BATCH_SIZE = 2;
    constexpr TI HEIGHT = 8;
    constexpr TI WIDTH = 8;
    constexpr TI INPUT_CHANNELS = 3;

    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, HEIGHT, WIDTH, INPUT_CHANNELS>;
    using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 8, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
    using MAXPOOL_CONFIG = rlt::nn::layers::max_pool2d::Configuration<TYPE_POLICY, TI, 2, 2, 2, 2>;
    using MAXPOOL = rlt::nn::layers::max_pool2d::BindConfiguration<MAXPOOL_CONFIG>;
    using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 16, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
    using AVGPOOL_CONFIG = rlt::nn::layers::avg_pool2d::Configuration<TYPE_POLICY, TI>;
    using AVGPOOL = rlt::nn::layers::avg_pool2d::BindConfiguration<AVGPOOL_CONFIG>;
    using DENSE_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using DENSE = rlt::nn::layers::dense::BindConfiguration<DENSE_CONFIG>;

    using MODULE_CHAIN = rlt::nn_models::sequential::Module<CONV1, MAXPOOL, CONV2, AVGPOOL, DENSE>;
    using CAPABILITY = rlt::nn::capability::Forward<>;
    using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;

    MODEL model;
    typename MODEL::Buffer<> buffer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::OUTPUT_SHAPE>> output;

    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);

    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, output);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);
    rlt::evaluate(device, model, input, output, buffer, rng);

    save_single_input_fixture(device, model, input, output, output_dir + "/rltools_js_conv2d.h5");

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, output);
    rlt::free(device, rng);
}

void generate_conv2d_flatten(DEVICE& device, const std::string& output_dir){
    constexpr TI BATCH_SIZE = 2;
    constexpr TI HEIGHT = 6;
    constexpr TI WIDTH = 6;
    constexpr TI INPUT_CHANNELS = 3;

    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, HEIGHT, WIDTH, INPUT_CHANNELS>;
    using CONV_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 4, 3, 3, 2, 2, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV = rlt::nn::layers::conv2d::BindConfiguration<CONV_CONFIG>;
    using FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using FLATTEN = rlt::nn::layers::flatten::BindConfiguration<FLATTEN_CONFIG>;
    using DENSE_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using DENSE = rlt::nn::layers::dense::BindConfiguration<DENSE_CONFIG>;

    using MODULE_CHAIN = rlt::nn_models::sequential::Module<CONV, FLATTEN, DENSE>;
    using CAPABILITY = rlt::nn::capability::Forward<>;
    using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;

    MODEL model;
    typename MODEL::Buffer<> buffer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::OUTPUT_SHAPE>> output;

    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 99);

    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, output);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);
    rlt::evaluate(device, model, input, output, buffer, rng);

    save_single_input_fixture(device, model, input, output, output_dir + "/rltools_js_flatten.h5");

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, output);
    rlt::free(device, rng);
}

void generate_parallel(DEVICE& device, const std::string& output_dir){
    constexpr TI BATCH_SIZE = 2;
    constexpr TI INPUT_DIM_A = 10;
    constexpr TI INPUT_DIM_B = 8;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, INPUT_DIM_A>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, INPUT_DIM_B>;

    using DENSE_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 16, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using DENSE_A = rlt::nn::layers::dense::BindConfiguration<DENSE_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<DENSE_A>;

    using DENSE_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 16, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using DENSE_B = rlt::nn::layers::dense::BindConfiguration<DENSE_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<DENSE_B>;

    using HEAD_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using HEAD = rlt::nn::layers::dense::BindConfiguration<HEAD_CONFIG>;
    using HEAD_MODULE = rlt::nn_models::sequential::Module<HEAD>;

    using CAPABILITY = rlt::nn::capability::Forward<>;
    using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B, HEAD_MODULE>;

    MODEL model;
    typename MODEL::Buffer<> buffer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A>> input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B>> input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::OUTPUT_SHAPE>> output;

    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 123);

    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, output);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input_a, rng);
    rlt::randn(device, input_b, rng);
    rlt::evaluate(device, model, input_a, input_b, output, buffer, rng);

    auto file = rl_tools::persist::backends::hdf5::File(output_dir + "/rltools_js_parallel.h5", rl_tools::persist::backends::hdf5::Mode::WRITE);
    auto actor_group = rlt::create_group(device, file, "actor");
    rlt::save(device, model, actor_group);
    auto example_group = rlt::create_group(device, file, "example");
    rlt::save(device, input_a, example_group, "input_a");
    rlt::save(device, input_b, example_group, "input_b");
    rlt::save(device, output, example_group, "output");
    std::cout << "Saved fixture: " << output_dir << "/rltools_js_parallel.h5" << std::endl;

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input_a);
    rlt::free(device, input_b);
    rlt::free(device, output);
    rlt::free(device, rng);
}

int main(int argc, char* argv[]){
    std::string output_dir = "tests/data";
    if(argc > 1){
        output_dir = argv[1];
    }

    DEVICE device;

    generate_conv2d_avgpool(device, output_dir);
    generate_conv2d_flatten(device, output_dir);
    generate_parallel(device, output_dir);

    return 0;
}
