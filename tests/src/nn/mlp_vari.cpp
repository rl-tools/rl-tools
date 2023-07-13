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
    struct OutputModule{};

    template <typename SPEC>
    constexpr auto find_output_dim() {
        if constexpr (bpt::utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>){
            return SPEC::CONTENT::OUTPUT_DIM;
        } else {
            return find_output_dim<typename SPEC::NEXT_MODULE>();
        }
    }
    template <typename TI, typename SPEC>
    constexpr auto find_max_hiddend_dim(TI current_max = 0) {
        current_max = current_max > SPEC::CONTENT::OUTPUT_DIM ? current_max : SPEC::CONTENT::OUTPUT_DIM;
        if constexpr (bpt::utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>){
            return 0;
        } else {
            auto max_downstream = find_max_hiddend_dim<TI, typename SPEC::NEXT_MODULE>();
            return max_downstream > current_max ? max_downstream : current_max;
        }
    }

    template <typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    constexpr bool check_input_output = INPUT_SPEC::COLS == SPEC::CONTENT::INPUT_DIM && OUTPUT_SPEC::ROWS == INPUT_SPEC::ROWS && OUTPUT_SPEC::COLS == find_output_dim<SPEC>();

    template <typename T_CONTENT, typename T_NEXT_MODULE = OutputModule>
    struct Specification{
        using CONTENT = T_CONTENT;
        using NEXT_MODULE = T_NEXT_MODULE;
        static constexpr auto OUTPUT_DIM = find_output_dim<Specification<T_CONTENT, T_NEXT_MODULE>>();
        static constexpr auto MAX_HIDDEN_DIM = find_max_hiddend_dim<typename CONTENT::TI, Specification<T_CONTENT, T_NEXT_MODULE>>();
    };
    template <typename T_SPEC>
    struct ModuleInternal{
        using SPEC = T_SPEC;
        using CONTENT = typename SPEC::CONTENT;
        using NEXT_MODULE = typename SPEC::NEXT_MODULE;
        CONTENT content;
        NEXT_MODULE next_module;
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
    template <typename DEVICE, typename MODULE_SPEC, typename RNG>
    void init_weights(DEVICE& device, nn_models::sequential::ModuleInternal<MODULE_SPEC>& module, RNG& rng){
        using namespace nn_models::sequential;
        init_weights(device, module.content);
        if constexpr(!bpt::utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            init_weights(device, module.next_module);
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

TEST(BACKPROP_TOOLS_NN_MODELS_MLP_VARI, TEST){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    using MLP_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, 5, 2, 3, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using MLP_SPEC = bpt::nn_models::mlp::AdamSpecification<MLP_STRUCTURE_SPEC>;
    using MLP = bpt::nn_models::mlp::NeuralNetworkAdam<MLP_SPEC>;

    using OPTIMIZER_SPEC = bpt::nn::optimizers::adam::DefaultParametersTF<T>;
    using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_SPEC>;

    using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 5, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
    using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 2, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
    using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

    using namespace bpt::nn_models::sequential::interface;
    using MLP_VARI = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;

    std::cout << "Max hidden dim: " << bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename MLP_VARI::SPEC>() << std::endl;

    DEVICE device;
    MLP mlp;
    OPTIMIZER optimizer;
    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 1);

    LAYER_1 layer_1;
    LAYER_2 layer_2;
    LAYER_3 layer_3;

    MLP_VARI mlp_vari;

    bpt::malloc(device, mlp);
    bpt::malloc(device, layer_1);
    bpt::malloc(device, layer_2);
    bpt::malloc(device, layer_3);

    bpt::malloc(device, mlp_vari);

    bpt::init_weights(device, mlp, rng);
    bpt::copy(device, device, layer_1, mlp.input_layer);
    bpt::copy(device, device, layer_2, mlp.hidden_layers[0]);
    bpt::copy(device, device, layer_3, mlp.output_layer);

    bpt::copy(device, device, mlp_vari.content, mlp.input_layer);
    bpt::copy(device, device, mlp_vari.next_module.content, mlp.hidden_layers[0]);
    bpt::copy(device, device, mlp_vari.next_module.next_module.content, mlp.output_layer);

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 5>> input;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> hidden_tick;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> hidden_tock;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_mlp;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_chain;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_mlp_vari;
    bpt::malloc(device, input);
    bpt::malloc(device, hidden_tick);
    bpt::malloc(device, hidden_tock);
    bpt::malloc(device, output_mlp);
    bpt::malloc(device, output_chain);
    bpt::malloc(device, output_mlp_vari);

    bpt::randn(device, input, rng);
    bpt::print(device, input);

//    bpt::forward(device, mlp, input, output_mlp);
//    bpt::print(device, output_mlp);

//    bpt::forward(device, layer_1, input, hidden_tick);
//    bpt::forward(device, layer_2, hidden_tick, hidden_tock);
//    bpt::forward(device, layer_3, hidden_tock, output_chain);
//    bpt::print(device, output_chain);

//    bpt::forward(device, mlp_vari.content                        , input, hidden_tick);
//    bpt::forward(device, mlp_vari.next_module.content            , hidden_tick, hidden_tock);
//    bpt::forward(device, mlp_vari.next_module.next_module.content, hidden_tock, output_mlp_vari);
//    bpt::print(device, output_mlp_vari);

    bpt::forward(device, mlp_vari, input, output_mlp_vari);
    bpt::print(device, output_mlp_vari);
}

