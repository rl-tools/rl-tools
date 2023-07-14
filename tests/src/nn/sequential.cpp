#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/nn/optimizers/adam/operations_generic.h>
#include <backprop_tools/nn/operations_cpu.h>
#include <backprop_tools/nn_models/operations_cpu.h>

namespace bpt = backprop_tools;

#include <gtest/gtest.h>


//template <typename T_CONTENT>
//struct OutputModule{
//    using CONTENT = T_CONTENT;
//    static constexpr auto MAX_HIDDEN_DIM = CONTENT::INPUT_DIM;
//    CONTENT content;
//};
//
//template <typename T_CONTENT, typename T_NEXT_MODULE>
//struct Specification{
//    using CONTENT = T_CONTENT;
//    using NEXT_MODULE = T_NEXT_MODULE;
//    static constexpr auto NEXT_MODULE_INPUT_DIM = NEXT_MODULE::CONTENT::INPUT_DIM;
//    static_assert(NEXT_MODULE_INPUT_DIM == CONTENT::OUTPUT_DIM);
//    static constexpr auto NEXT_MODULE_INPUT_DIM = NEXT_MODULE::CONTENT::INPUT_DIM;
//};
//

namespace backprop_tools::nn_models::sequential{
    struct OutputModule{
        struct CONTENT{
            static constexpr auto INPUT_DIM = 0;
            static constexpr auto BATCH_SIZE = 0;
        };
    };

    // Required fields on CONTENT:
    // compile-time types
    //     T
    //     TI
    // compile-time constants
    //     INPUT_DIM
    //     OUTPUT_DIM
    //     BATCH_SIZE
    // run-time containers
    //     output (just required for forward and backward)
    // containers
    //     operations
    //     malloc
    //     free
    //     init_weights
    //     forward

    template <typename SPEC>
    constexpr auto find_output_dim() {
        if constexpr (bpt::utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>){
            return SPEC::CONTENT::OUTPUT_DIM;
        } else {
            return find_output_dim<typename SPEC::NEXT_MODULE>();
        }
    }
    template <typename TI, typename SPEC>
    constexpr auto find_max_hiddend_dim(TI current_max = 0){
        current_max = current_max > SPEC::CONTENT::OUTPUT_DIM ? current_max : SPEC::CONTENT::OUTPUT_DIM;
        if constexpr (utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>){
            return 0;
        } else {
            auto max_downstream = find_max_hiddend_dim<TI, typename SPEC::NEXT_MODULE>();
            return max_downstream > current_max ? max_downstream : current_max;
        }
    }
    template <typename MODULE>
    constexpr bool check_batch_size_consistency_f(){
        if constexpr (utils::typing::is_same_v<typename MODULE::NEXT_MODULE, OutputModule>){
            return true;
        } else {
            return MODULE::CONTENT::BATCH_SIZE == MODULE::NEXT_MODULE::CONTENT::BATCH_SIZE && check_batch_size_consistency_f<typename MODULE::NEXT_MODULE>();
        }
    }

    template <typename MODULE>
    constexpr bool check_batch_size_consistency = check_batch_size_consistency_f<MODULE>();

    template <typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    constexpr bool check_input_output = INPUT_SPEC::COLS == SPEC::CONTENT::INPUT_DIM && OUTPUT_SPEC::ROWS == INPUT_SPEC::ROWS && OUTPUT_SPEC::COLS == find_output_dim<SPEC>() && OUTPUT_SPEC::ROWS == INPUT_SPEC::ROWS;

    template <typename BUFFER_SPEC, typename MODULE_SPEC>
    constexpr bool buffer_compatible = BUFFER_SPEC::SPEC::MAX_HIDDEN_DIM >= MODULE_SPEC::MAX_HIDDEN_DIM;

    template <typename T_CONTENT, typename T_NEXT_MODULE = OutputModule>
    struct Specification{
        using CONTENT = T_CONTENT;
        using NEXT_MODULE = T_NEXT_MODULE;
        using T = typename CONTENT::T;
        using TI = typename CONTENT::TI;
        static constexpr auto INPUT_DIM = CONTENT::INPUT_DIM;
        static constexpr auto OUTPUT_DIM = find_output_dim<Specification<T_CONTENT, T_NEXT_MODULE>>();
        static constexpr auto MAX_HIDDEN_DIM = find_max_hiddend_dim<typename CONTENT::TI, Specification<T_CONTENT, T_NEXT_MODULE>>();
        static_assert(utils::typing::is_same_v<NEXT_MODULE, OutputModule> || CONTENT::OUTPUT_DIM == NEXT_MODULE::CONTENT::INPUT_DIM);
    };

    template <typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE, typename T_CONTAINER_TYPE_TAG, typename T_MEMORY_LAYOUT>
    struct ModuleDoubleBufferSpecification {
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        using MEMORY_LAYOUT = T_MEMORY_LAYOUT;
    };
    template <typename T_BUFFER_SPEC>
    struct ModuleDoubleBuffer{
        using BUFFER_SPEC = T_BUFFER_SPEC;
        using SPEC = typename BUFFER_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BUFFER_SPEC::BATCH_SIZE;
        using TICK_TOCK_CONTAINER_SPEC = matrix::Specification<T, TI, BATCH_SIZE, SPEC::MAX_HIDDEN_DIM, typename BUFFER_SPEC::MEMORY_LAYOUT>;
        using TICK_TOCK_CONTAINER_TYPE = typename BUFFER_SPEC::CONTAINER_TYPE_TAG::template type<TICK_TOCK_CONTAINER_SPEC>;
        TICK_TOCK_CONTAINER_TYPE tick;
        TICK_TOCK_CONTAINER_TYPE tock;
    };
    template <typename T_SPEC>
    struct ModuleInternal{
        using SPEC = T_SPEC;
        using CONTENT = typename SPEC::CONTENT;
        using NEXT_MODULE = typename SPEC::NEXT_MODULE;
        CONTENT content;
        NEXT_MODULE next_module;

        template <typename SPEC::TI BATCH_SIZE, typename CONTAINER_TYPE_TAG = MatrixDynamicTag, typename MEMORY_LAYOUT = matrix::layouts::DEFAULT<typename SPEC::TI>>
        using DoubleBuffer = ModuleDoubleBuffer<ModuleDoubleBufferSpecification<SPEC, BATCH_SIZE, CONTAINER_TYPE_TAG, MEMORY_LAYOUT>>;
    };

    namespace interface{
        template <typename T_CONTENT, typename T_NEXT_MODULE = OutputModule>
        struct Module: ModuleInternal<Specification<T_CONTENT, T_NEXT_MODULE>>{};
    }
}

namespace backprop_tools{
    template <typename DEVICE, typename MODULE_SPEC>
    void malloc(DEVICE& device, nn_models::sequential::ModuleInternal<MODULE_SPEC>& module){
        using namespace nn_models::sequential;
        malloc(device, module.content);
        if constexpr(!bpt::utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            malloc(device, module.next_module);
        }
    }
    template <typename DEVICE, typename MODULE_SPEC>
    void free(DEVICE& device, nn_models::sequential::ModuleInternal<MODULE_SPEC>& module){
        using namespace nn_models::sequential;
        free(device, module.content);
        if constexpr(!bpt::utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            free(device, module.next_module, module.content.output);
        }
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    void malloc(DEVICE& device, nn_models::sequential::ModuleDoubleBuffer<BUFFER_SPEC>& buffers){
        using namespace nn_models::sequential;
        malloc(device, buffers.tick);
        malloc(device, buffers.tock);
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    void free(DEVICE& device, nn_models::sequential::ModuleDoubleBuffer<BUFFER_SPEC>& buffers){
        using namespace nn_models::sequential;
        free(device, buffers.tick);
        free(device, buffers.tock);
    }
    template <typename DEVICE, typename MODULE_SPEC, typename RNG>
    void init_weights(DEVICE& device, nn_models::sequential::ModuleInternal<MODULE_SPEC>& module, RNG& rng){
        using namespace nn_models::sequential;
        init_weights(device, module.content, rng);
        if constexpr(!bpt::utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            init_weights(device, module.next_module, rng);
        }
    }
    template<typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, bool TICK = true>
    void evaluate(DEVICE& device, const nn_models::sequential::ModuleInternal<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::sequential::ModuleDoubleBuffer<BUFFER_SPEC>& buffers){
        static_assert(nn_models::sequential::buffer_compatible<BUFFER_SPEC, MODULE_SPEC>);
        static_assert(BUFFER_SPEC::BATCH_SIZE == OUTPUT_SPEC::ROWS);
        static_assert(nn_models::sequential::check_input_output<MODULE_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using DOUBLE_BUFFER_TYPE = decltype(buffers.tick);

//        static_assert(TEMP_SPEC::ROWS >= BATCH_SIZE);
//        static_assert(TEMP_SPEC::COLS >= MODEL_SPEC::HIDDEN_DIM);

        if constexpr(utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            evaluate(device, model.content, input, output);
        }
        else{
            DOUBLE_BUFFER_TYPE& output_buffer = TICK ? buffers.tick : buffers.tock;
            evaluate(device, model.content, input, output_buffer);
            evaluate<DEVICE, typename MODULE_SPEC::NEXT_MODULE::SPEC, typename DOUBLE_BUFFER_TYPE::SPEC, OUTPUT_SPEC, BUFFER_SPEC, !TICK>(device, model.next_module, output_buffer, output, buffers);
        }
    }
    template <typename DEVICE, typename MODULE_SPEC, typename INPUT, typename OUTPUT>
    void forward(DEVICE& device, nn_models::sequential::ModuleInternal<MODULE_SPEC>& module, INPUT& input, OUTPUT& output){
        using namespace nn_models::sequential;
        forward(device, module.content, input);
        if constexpr(!bpt::utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            forward(device, module.next_module, module.content.output, output);
        }
        else{
            bpt::copy(device, device, output, module.content.output);
        }
    }
    template<typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, bool TICK = true>
    void backward(DEVICE& device, nn_models::sequential::ModuleInternal<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::sequential::ModuleDoubleBuffer<BUFFER_SPEC> buffers) {
        static_assert(nn_models::sequential::buffer_compatible<BUFFER_SPEC, MODULE_SPEC>);
        using DOUBLE_BUFFER_TYPE = decltype(buffers.tick);

        if constexpr(utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            backward(device, model.content, input, d_output, d_input);
        }
        else{
            DOUBLE_BUFFER_TYPE& current_d_output_buffer = TICK ? buffers.tick : buffers.tock;
            backward<DEVICE, typename MODULE_SPEC::NEXT_MODULE::SPEC, typename decltype(model.content.output)::SPEC, D_OUTPUT_SPEC, typename DOUBLE_BUFFER_TYPE::SPEC, BUFFER_SPEC, !TICK>(device, model.next_module, model.content.output, d_output, current_d_output_buffer, buffers);
            backward(device, model.content, input, current_d_output_buffer, d_input);
        }

    }
}


TEST(BACKPROP_TOOLS_NN_MODELS_MLP_VARI, TEST_SEQUENTIAL_STATIC){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    {
        using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 20, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;

        using namespace bpt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1>;

        static_assert(bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 0);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 0);
        static_assert(bpt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 10);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 10);
    }
    {
        using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 1, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 1, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;

        using namespace bpt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2>>;

        static_assert(bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 10);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 10);
        static_assert(bpt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 1);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 1);
    }
    {
        using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 100, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 100, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;

        using namespace bpt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2>>;

        static_assert(bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 10);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 10);
        static_assert(bpt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 100);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 100);
    }
    {
        using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 20, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 11, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
        using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, 11, 11, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
        using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;
        using LAYER_4_SPEC = bpt::nn::layers::dense::Specification<T, TI, 11, 20, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
        using LAYER_4 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_4_SPEC>;

        using namespace bpt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3, Module<LAYER_4>>>>;

        static_assert(bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 11);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 11);
        static_assert(bpt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 20);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 20);
    }
    {
        using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 20, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 11, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
        using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, 11, 11, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
        using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;
        using LAYER_4_SPEC = bpt::nn::layers::dense::Specification<T, TI, 11, 100, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
        using LAYER_4 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_4_SPEC>;
        using LAYER_5_SPEC = bpt::nn::layers::dense::Specification<T, TI, 100, 20, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
        using LAYER_5 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_5_SPEC>;

        using namespace bpt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3, Module<LAYER_4, Module<LAYER_5>>>>>;

        static_assert(bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 100);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 100);
        static_assert(bpt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 20);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 20);
    }

}

TEST(BACKPROP_TOOLS_NN_MODELS_MLP_VARI, TEST_FORWARD){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    using MLP_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, 5, 2, 3, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using MLP_SPEC = bpt::nn_models::mlp::AdamSpecification<MLP_STRUCTURE_SPEC>;
    using MLP = bpt::nn_models::mlp::NeuralNetworkAdam<MLP_SPEC>;

    using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 5, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
    using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 2, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
    using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

    using namespace bpt::nn_models::sequential::interface;
    using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;

    std::cout << "Max hidden dim: " << bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() << std::endl;

    DEVICE device;
    MLP mlp;
    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 1);

    LAYER_1 layer_1;
    LAYER_2 layer_2;
    LAYER_3 layer_3;

    SEQUENTIAL sequential;

    bpt::malloc(device, mlp);
    bpt::malloc(device, layer_1);
    bpt::malloc(device, layer_2);
    bpt::malloc(device, layer_3);

    bpt::malloc(device, sequential);

    bpt::init_weights(device, mlp, rng);
    bpt::copy(device, device, layer_1, mlp.input_layer);
    bpt::copy(device, device, layer_2, mlp.hidden_layers[0]);
    bpt::copy(device, device, layer_3, mlp.output_layer);

    bpt::copy(device, device, sequential.content, mlp.input_layer);
    bpt::copy(device, device, sequential.next_module.content, mlp.hidden_layers[0]);
    bpt::copy(device, device, sequential.next_module.next_module.content, mlp.output_layer);

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 5>> input;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> hidden_tick;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> hidden_tock;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_mlp;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_chain;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_sequential;
    bpt::malloc(device, input);
    bpt::malloc(device, hidden_tick);
    bpt::malloc(device, hidden_tock);
    bpt::malloc(device, output_mlp);
    bpt::malloc(device, output_chain);
    bpt::malloc(device, output_sequential);

    bpt::randn(device, input, rng);
    bpt::print(device, input);

    for(TI i = 0; i < 2; i++){
        bpt::forward(device, mlp, input, output_mlp);
        bpt::print(device, output_mlp);

        bpt::forward(device, layer_1, input, hidden_tick);
        bpt::forward(device, layer_2, hidden_tick, hidden_tock);
        bpt::forward(device, layer_3, hidden_tock, output_chain);
        bpt::print(device, output_chain);

        bpt::forward(device, sequential.content                        , input, hidden_tick);
        bpt::forward(device, sequential.next_module.content            , hidden_tick, hidden_tock);
        bpt::forward(device, sequential.next_module.next_module.content, hidden_tock, output_sequential);
        bpt::print(device, output_sequential);

        bpt::forward(device, sequential, input, output_sequential);
        bpt::print(device, output_sequential);

        auto abs_diff_sequential = bpt::abs_diff(device, output_mlp, output_sequential);
        auto abs_diff_chain = bpt::abs_diff(device, output_mlp, output_sequential);

        std::cout << "Abs diff sequential: " << abs_diff_sequential << std::endl;
        std::cout << "Abs diff chain: " << abs_diff_chain << std::endl;

        ASSERT_LT(abs_diff_sequential, 1e-8);
        ASSERT_LT(abs_diff_chain, 1e-8);

        bpt::init_weights(device, sequential, rng);
        bpt::copy(device, device, mlp.input_layer, sequential.content);
        bpt::copy(device, device, mlp.hidden_layers[0], sequential.next_module.content);
        bpt::copy(device, device, mlp.output_layer, sequential.next_module.next_module.content);
    }
}

TEST(BACKPROP_TOOLS_NN_MODELS_MLP_VARI, TEST_INCOMPATIBLE_DEFINITION){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;
    using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 1, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam, bpt::MatrixDynamicTag, 10>;
    using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 1, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam, bpt::MatrixDynamicTag, 10>;
    using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;

    using namespace bpt::nn_models::sequential::interface;
    using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2>>;

    DEVICE device;
    SEQUENTIAL model;
    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 1);

    bpt::malloc(device, model);
    bpt::init_weights(device, model, rng);

    static_assert(bpt::nn_models::sequential::check_batch_size_consistency<SEQUENTIAL>);
}

TEST(BACKPROP_TOOLS_NN_MODELS_MLP_VARI, TEST_EVALUATE){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 5, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
    using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 2, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
    using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

    using namespace bpt::nn_models::sequential::interface;
    using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;

    std::cout << "Max hidden dim: " << bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() << std::endl;

    DEVICE device;
    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 1);

    LAYER_1 layer_1;
    LAYER_2 layer_2;
    LAYER_3 layer_3;

    SEQUENTIAL sequential;
    typename SEQUENTIAL::DoubleBuffer<1> buffer;

    bpt::malloc(device, sequential);
    bpt::malloc(device, buffer);
    bpt::init_weights(device, sequential, rng);

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 5>> input;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_sequential;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_sequential_evaluate;
    bpt::malloc(device, input);
    bpt::malloc(device, output_sequential);
    bpt::malloc(device, output_sequential_evaluate);

    bpt::randn(device, input, rng);
    bpt::print(device, input);

    bpt::forward(device, sequential, input, output_sequential);
    bpt::print(device, output_sequential);
    bpt::evaluate(device, sequential, input, output_sequential_evaluate, buffer);
    bpt::print(device, output_sequential_evaluate);

    auto abs_diff = bpt::abs_diff(device, output_sequential_evaluate, output_sequential);

    std::cout << "Abs diff evaluate: " << abs_diff << std::endl;

    ASSERT_LT(abs_diff, 1e-8);

}

TEST(BACKPROP_TOOLS_NN_MODELS_MLP_VARI, TEST_BACKWARD){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    constexpr T THRESHOLD = 1e-8;

    using MLP_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, 5, 2, 3, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using MLP_SPEC = bpt::nn_models::mlp::AdamSpecification<MLP_STRUCTURE_SPEC>;
    using MLP = bpt::nn_models::mlp::NeuralNetworkAdam<MLP_SPEC>;

    using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 5, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
    using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 2, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
    using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

    using namespace bpt::nn_models::sequential::interface;
    using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;

    std::cout << "Max hidden dim: " << bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() << std::endl;

    DEVICE device;
    MLP mlp;
    typename MLP::Buffers<1> mlp_buffers;
    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 1);

    LAYER_1 layer_1;
    LAYER_2 layer_2;
    LAYER_3 layer_3;

    SEQUENTIAL sequential;
    SEQUENTIAL::DoubleBuffer<1> buffer_sequential;

    bpt::malloc(device, mlp);
    bpt::malloc(device, layer_1);
    bpt::malloc(device, layer_2);
    bpt::malloc(device, layer_3);
    bpt::malloc(device, mlp_buffers);

    bpt::malloc(device, sequential);
    bpt::malloc(device, buffer_sequential);

    bpt::init_weights(device, mlp, rng);
    bpt::copy(device, device, layer_1, mlp.input_layer);
    bpt::copy(device, device, layer_2, mlp.hidden_layers[0]);
    bpt::copy(device, device, layer_3, mlp.output_layer);

    bpt::copy(device, device, sequential.content, mlp.input_layer);
    bpt::copy(device, device, sequential.next_module.content, mlp.hidden_layers[0]);
    bpt::copy(device, device, sequential.next_module.next_module.content, mlp.output_layer);

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 5>> input;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 5>> d_input_mlp, d_input_chain, d_input_sequential;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> hidden_tick;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> hidden_tock;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> d_hidden_tick;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> d_hidden_tock;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_mlp;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_chain;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_sequential;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> d_output;
    bpt::malloc(device, input);
    bpt::malloc(device, d_input_mlp);
    bpt::malloc(device, d_input_chain);
    bpt::malloc(device, d_input_sequential);
    bpt::malloc(device, hidden_tick);
    bpt::malloc(device, hidden_tock);
    bpt::malloc(device, d_hidden_tick);
    bpt::malloc(device, d_hidden_tock);
    bpt::malloc(device, output_mlp);
    bpt::malloc(device, output_chain);
    bpt::malloc(device, output_sequential);
    bpt::malloc(device, d_output);

    bpt::randn(device, input, rng);
    bpt::randn(device, d_output, rng);
    bpt::print(device, input);

    bpt::forward(device, mlp, input, output_mlp);
    bpt::backward(device, mlp, input, d_output, d_input_mlp, mlp_buffers);

    bpt::print(device, d_input_mlp);

    bpt::zero_gradient(device, layer_1);
    bpt::zero_gradient(device, layer_2);
    bpt::zero_gradient(device, layer_3);
    bpt::forward(device, layer_1, input, hidden_tick);
    bpt::forward(device, layer_2, hidden_tick, hidden_tock);
    bpt::forward(device, layer_3, hidden_tock, output_chain);
    bpt::backward(device, layer_3, hidden_tock, d_output, d_hidden_tick);
    bpt::backward(device, layer_2, hidden_tick, d_hidden_tick, d_hidden_tock);
    bpt::backward(device, layer_1, input, d_hidden_tock, d_input_chain);

    bpt::print(device, d_input_chain);

    {
        auto abs_diff_d_input = bpt::abs_diff(device, d_input_mlp, d_input_chain);
        auto abs_diff_grad_W_1 = bpt::abs_diff(device, mlp.input_layer.weights.gradient, layer_1.weights.gradient);
        auto abs_diff_grad_b_1 = bpt::abs_diff(device, mlp.input_layer.biases.gradient, layer_1.biases.gradient);
        auto abs_diff_grad_W_2 = bpt::abs_diff(device, mlp.hidden_layers[0].weights.gradient, layer_2.weights.gradient);
        auto abs_diff_grad_b_2 = bpt::abs_diff(device, mlp.hidden_layers[0].biases.gradient, layer_2.biases.gradient);
        auto abs_diff_grad_W_3 = bpt::abs_diff(device, mlp.output_layer.weights.gradient, layer_3.weights.gradient);
        auto abs_diff_grad_b_3 = bpt::abs_diff(device, mlp.output_layer.biases.gradient, layer_3.biases.gradient);

        ASSERT_LT(abs_diff_d_input, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_1, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_1, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_2, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_2, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_3, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_3, THRESHOLD);
    }

    bpt::forward(device, sequential.content                        , input, hidden_tick);
    bpt::forward(device, sequential.next_module.content            , hidden_tick, hidden_tock);
    bpt::forward(device, sequential.next_module.next_module.content, hidden_tock, output_sequential);

    bpt::forward(device, sequential, input, output_sequential);
    bpt::backward(device, sequential, input, d_output, d_input_sequential, buffer_sequential);

    bpt::print(device, d_input_sequential);

    {
        auto abs_diff_d_input = bpt::abs_diff(device, d_input_mlp, d_input_chain);
        auto abs_diff_grad_W_1 = bpt::abs_diff(device, sequential.content.weights.gradient, layer_1.weights.gradient);
        auto abs_diff_grad_b_1 = bpt::abs_diff(device, sequential.content.biases.gradient, layer_1.biases.gradient);
        auto abs_diff_grad_W_2 = bpt::abs_diff(device, sequential.next_module.content.weights.gradient, layer_2.weights.gradient);
        auto abs_diff_grad_b_2 = bpt::abs_diff(device, sequential.next_module.content.biases.gradient, layer_2.biases.gradient);
        auto abs_diff_grad_W_3 = bpt::abs_diff(device, sequential.next_module.next_module.content.weights.gradient, layer_3.weights.gradient);
        auto abs_diff_grad_b_3 = bpt::abs_diff(device, sequential.next_module.next_module.content.biases.gradient, layer_3.biases.gradient);

        ASSERT_LT(abs_diff_d_input, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_1, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_1, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_2, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_2, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_3, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_3, THRESHOLD);
    }
}
