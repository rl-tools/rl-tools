#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/sgd/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/lamb/instance/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/sgd/operations_generic.h>
#include <rl_tools/nn/optimizers/lamb/operations_generic.h>
#include <rl_tools/numeric_types/persist_code.h>
#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/parallel/persist_code.h>
#include <fstream>
#include <metra/metra.h>
#include <gtest/gtest.h>

namespace rl_tools::tests::groups{
    using DEVICE = devices::DefaultCPU;
    using TI = DEVICE::index_t;
    using T = double;
    using TYPE_POLICY = numeric_types::Policy<T>;
    using CAPABILITY = nn::capability::Gradient<nn::parameters::Adam>;
    using SHAPE = tensor::Shape<TI, 1, 3, 2>;
    using DENSE = nn::layers::dense::BindConfiguration<nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, nn::activation_functions::TANH>>;
    struct Fixed{};
    struct Learned{};
    struct Parent{};
    struct Missing{};
    template<typename T_BASE, typename T_SPEC>
    struct Inspect: T_BASE{
        using BASE = T_BASE;
        using SPEC = T_SPEC;
    };
    struct LearnedMatrixWeights{
        template<typename TARGET>
        static constexpr bool MATCH = nn::group::Only<Learned>::MATCH<TARGET> && nn::group::Category<nn::parameters::categories::Weights>::MATCH<TARGET> && length(typename TARGET::PARAMETER_SPEC::SHAPE{}) == 2;
    };
    using CHAIN = nn_models::sequential::Module<nn_models::Group<Fixed, nn_models::sequential::Module<DENSE, DENSE>>, nn_models::Group<Learned, DENSE>, nn_models::Group<Fixed, DENSE>>;
    using MODEL = nn_models::sequential::Build<CAPABILITY, CHAIN, SHAPE>;
    using PLAIN = nn_models::sequential::Build<CAPABILITY, nn_models::sequential::Module<DENSE, DENSE, DENSE, DENSE>, SHAPE>;
    using FREEZE_FIXED = Mode<nn::optimizers::mode::Freeze<mode::Default<>, nn::group::Only<Fixed>>>;
    using WEIGHTS = Mode<nn::optimizers::mode::Update<mode::Default<>, nn::group::Every<nn::group::Only<Learned>, nn::group::Category<nn::parameters::categories::Weights>>>>;
    struct OPTIMIZER_PARAMETERS: nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{
        static constexpr bool ENABLE_WEIGHT_DECAY = true;
        static constexpr T WEIGHT_DECAY = 0.1;
    };
    using OPTIMIZER = nn::optimizers::Adam<nn::optimizers::adam::Specification<TYPE_POLICY, TI, OPTIMIZER_PARAMETERS>>;
    using TENSOR = Tensor<tensor::Specification<T, TI, SHAPE>>;

    struct Fixture: ::testing::Test{
        DEVICE device;
        DEVICE::SPEC::RANDOM::ENGINE<> rng;
        MODEL model;
        PLAIN reference;
        MODEL::Buffer<> buffer;
        PLAIN::Buffer<> reference_buffer;
        TENSOR input, d_output, reference_d_output, d_input, reference_d_input;
        OPTIMIZER optimizer;

        void SetUp() override{
            malloc(device, rng);
            init(device, rng, 19);
            malloc(device, model);
            malloc(device, reference);
            malloc(device, buffer);
            malloc(device, reference_buffer);
            malloc(device, input);
            malloc(device, d_output);
            malloc(device, reference_d_output);
            malloc(device, d_input);
            malloc(device, reference_d_input);
            malloc(device, optimizer);
            init(device, optimizer);
            init_weights(device, model, rng);
            reset_optimizer_state(device, optimizer, model);
            zero_gradient(device, model);
            copy(device, device, model, reference);
            set_all(device, input, T(0.3));
            set_all(device, d_output, T(0.7));
            copy(device, device, d_output, reference_d_output);
        }
        void TearDown() override{
            free(device, optimizer);
            free(device, reference_d_input);
            free(device, d_input);
            free(device, reference_d_output);
            free(device, d_output);
            free(device, input);
            free(device, reference_buffer);
            free(device, buffer);
            free(device, reference);
            free(device, model);
            free(device, rng);
        }
        void forward_both(){
            forward(device, model, input, buffer, rng);
            forward(device, reference, input, reference_buffer, rng);
            EXPECT_DOUBLE_EQ(abs_diff(device, output(device, model), output(device, reference)), 0);
        }
        template<auto I = 0>
        void check_gradients(){
            if constexpr(I < MODEL::SPEC::NUM_LAYERS){
                auto& layer = get_layer<I>(model);
                auto& expected = get_layer<I>(reference);
                EXPECT_NEAR(abs_diff(device, layer.weights.gradient, expected.weights.gradient), 0, 1e-12);
                EXPECT_NEAR(abs_diff(device, layer.biases.gradient, expected.biases.gradient), 0, 1e-12);
                EXPECT_GT(gradient_norm(device, layer), 0);
                check_gradients<I + 1>();
            }
        }
    };

    TEST_F(Fixture, GroupingPreservesFullBackward){
        static_assert(MODEL::SPEC::NUM_LAYERS == PLAIN::SPEC::NUM_LAYERS);
        static_assert(sizeof(MODEL) == sizeof(PLAIN));
        forward_both();
        backward(device, model, input, d_output, buffer);
        backward(device, reference, input, reference_d_output, reference_buffer);
        check_gradients();
        metra::log("nn_groups/gradient_error", abs_diff(device, get_layer<2>(model).weights.gradient, get_layer<2>(reference).weights.gradient));
    }

    TEST_F(Fixture, InputGradientTraversesFrozenPrefixAndSuffix){
        forward_both();
        backward_full(device, model, input, d_output, d_input, buffer);
        backward_full(device, reference, input, reference_d_output, reference_d_input, reference_buffer);
        check_gradients();
        EXPECT_NEAR(abs_diff(device, d_input, reference_d_input), 0, 1e-12);
        EXPECT_GT(squared_sum(device, d_input), 0);
    }

    TEST_F(Fixture, FreezeAndUnfreezePreserveMomentsAndParameters){
        forward(device, model, input, buffer, rng);
        backward(device, model, input, d_output, buffer);
        step(device, optimizer, model);
        EXPECT_GT(squared_sum(device, get_layer<0>(model).weights.gradient_first_order_moment), 0);
        copy(device, device, model, reference);
        zero_gradient(device, model);
        set_all(device, d_output, T(0.7));
        forward(device, model, input, buffer, rng);
        backward(device, model, input, d_output, buffer);
        step(device, optimizer, model, FREEZE_FIXED{});
        EXPECT_DOUBLE_EQ(abs_diff(device, get_layer<0>(model).weights.parameters, get_layer<0>(reference).weights.parameters), 0);
        EXPECT_DOUBLE_EQ(abs_diff(device, get_layer<0>(model).weights.gradient_first_order_moment, get_layer<0>(reference).weights.gradient_first_order_moment), 0);
        EXPECT_DOUBLE_EQ(abs_diff(device, get_layer<0>(model).weights.gradient_second_order_moment, get_layer<0>(reference).weights.gradient_second_order_moment), 0);
        EXPECT_DOUBLE_EQ(abs_diff(device, get_layer<3>(model).weights.parameters, get_layer<3>(reference).weights.parameters), 0);
        EXPECT_GT(abs_diff(device, get_layer<2>(model).weights.parameters, get_layer<2>(reference).weights.parameters), 0);
        zero_gradient(device, model);
        set_all(device, d_output, T(0.7));
        forward(device, model, input, buffer, rng);
        backward(device, model, input, d_output, buffer);
        step(device, optimizer, model);
        EXPECT_GT(abs_diff(device, get_layer<0>(model).weights.parameters, get_layer<0>(reference).weights.parameters), 0);
    }

    TEST_F(Fixture, ParameterCategorySelection){
        forward_both();
        backward(device, model, input, d_output, buffer);
        backward(device, reference, input, reference_d_output, reference_buffer);
        EXPECT_NEAR(abs_diff(device, get_layer<2>(model).weights.gradient, get_layer<2>(reference).weights.gradient), 0, 1e-12);
        EXPECT_GT(squared_sum(device, get_layer<2>(model).biases.gradient), 0);
        set_all(device, get_layer<2>(model).biases.gradient_first_order_moment, T(1));
        set_all(device, get_layer<2>(model).biases.gradient_second_order_moment, T(1));
        step(device, optimizer, model, WEIGHTS{});
        EXPECT_DOUBLE_EQ(abs_diff(device, get_layer<2>(model).biases.parameters, get_layer<2>(reference).biases.parameters), 0);
        EXPECT_DOUBLE_EQ(get(device, get_layer<2>(model).biases.gradient_first_order_moment, 0), 1);
    }

    TEST_F(Fixture, NestedGroupsAndDefaultMembership){
        using NESTED_CHAIN = nn_models::sequential::Module<DENSE, nn_models::Group<Parent, nn_models::Group<Learned, nn_models::sequential::Module<DENSE, DENSE>>>>;
        using NESTED = nn_models::sequential::Build<CAPABILITY, NESTED_CHAIN, SHAPE>;
        using FIRST = typename nn_models::sequential::layer_spec_t<0, NESTED::SPEC>::CONTENT::WEIGHTS_PARAMETER_SPEC;
        using SECOND = typename nn_models::sequential::layer_spec_t<1, NESTED::SPEC>::CONTENT::WEIGHTS_PARAMETER_SPEC;
        static_assert(NESTED::SPEC::NUM_LAYERS == 3);
        static_assert(nn::group::Only<nn::group::Default>::MATCH<FIRST>);
        static_assert(!nn::group::Only<nn::group::Default>::MATCH<SECOND>);
        static_assert(nn::group::Only<Parent>::MATCH<SECOND>);
        static_assert(nn::group::Only<Learned>::MATCH<SECOND>);
        static_assert(nn::group::ParameterGroup<nn::parameters::groups::Normal>::MATCH<SECOND>);
        using STACKED = mode::Evaluation<nn::optimizers::mode::Update<mode::Default<>, nn::group::Only<Learned>>>;
        static_assert(mode::is<STACKED, mode::Evaluation>);
        static_assert(nn::optimizers::UpdateSelection<STACKED>::TYPE::MATCH<nn::optimizers::UpdateTarget<SECOND>>);
        static_assert(!nn::optimizers::UpdateSelection<STACKED>::TYPE::MATCH<nn::optimizers::UpdateTarget<FIRST>>);
        using NONE = Mode<nn::optimizers::mode::Update<mode::Default<>, nn::group::Only<Missing>>>;
        forward_both();
        backward(device, model, input, d_output, buffer);
        EXPECT_GT(gradient_norm(device, model), 0);
        step(device, optimizer, model, NONE{});
        EXPECT_DOUBLE_EQ(abs_diff(device, get_layer<0>(model).weights.parameters, get_layer<0>(reference).weights.parameters), 0);
    }

    TEST_F(Fixture, UpdateTargetAndIndependentEffects){
        struct INSPECTION_TARGET{
            using MODEL_GROUPS = nn::group::Membership<Learned>;
        };
        static_assert(nn::group::Only<Learned>::MATCH<INSPECTION_TARGET>);
        static_assert(!nn::group::Only<Fixed>::MATCH<INSPECTION_TARGET>);
        using OTHER_EFFECT = Inspect<mode::Default<>, nn::group::Only<Fixed>>;
        using UPDATE = nn::optimizers::mode::Update<OTHER_EFFECT, LearnedMatrixWeights>;
        using COMBINED = Inspect<UPDATE, nn::group::Only<Missing>>;
        static_assert(mode::is<COMBINED, Inspect>);
        static_assert(mode::is<COMBINED, nn::optimizers::mode::Update>);
        static_assert(utils::typing::is_same_v<nn::optimizers::UpdateSelection<COMBINED>::TYPE, LearnedMatrixWeights>);
        using OVERRIDE = nn::optimizers::mode::Update<COMBINED, nn::group::All>;
        static_assert(utils::typing::is_same_v<nn::optimizers::UpdateSelection<OVERRIDE>::TYPE, nn::group::All>);
        forward(device, model, input, buffer, rng);
        backward(device, model, input, d_output, buffer);
        step(device, optimizer, model, Mode<COMBINED>{});
        EXPECT_EQ(abs_diff(device, get_layer<0>(model).weights.parameters, get_layer<0>(reference).weights.parameters), 0);
        EXPECT_EQ(abs_diff(device, get_layer<2>(model).biases.parameters, get_layer<2>(reference).biases.parameters), 0);
        EXPECT_GT(abs_diff(device, get_layer<2>(model).weights.parameters, get_layer<2>(reference).weights.parameters), 0);
    }

    TEST_F(Fixture, DefaultAndUnrelatedModesUpdateAll){
        OPTIMIZER reference_optimizer;
        malloc(device, reference_optimizer);
        init(device, reference_optimizer);
        reset_optimizer_state(device, reference_optimizer, reference);
        forward_both();
        backward(device, model, input, d_output, buffer);
        backward(device, reference, input, reference_d_output, reference_buffer);
        step(device, optimizer, model);
        step(device, reference_optimizer, reference, Mode<mode::Default<>>{});
        EXPECT_EQ(abs_diff(device, model, reference), 0);
        using OTHER_EFFECT = Inspect<mode::Default<>, nn::group::Only<Missing>>;
        step(device, optimizer, model, Mode<OTHER_EFFECT>{});
        step(device, reference_optimizer, reference);
        EXPECT_EQ(abs_diff(device, model, reference), 0);
        free(device, reference_optimizer);
    }

    TEST_F(Fixture, FreezeMatchesAndModePrecedence){
        using FIRST = typename nn_models::sequential::layer_spec_t<0, MODEL::SPEC>::CONTENT::WEIGHTS_PARAMETER_SPEC;
        using THIRD = typename nn_models::sequential::layer_spec_t<2, MODEL::SPEC>::CONTENT::WEIGHTS_PARAMETER_SPEC;
        using UNGROUPED = typename nn_models::sequential::layer_spec_t<0, PLAIN::SPEC>::CONTENT::WEIGHTS_PARAMETER_SPEC;
        using STACKED = Inspect<FREEZE_FIXED::MODE, nn::group::Only<Learned>>;
        using SELECTION = nn::optimizers::UpdateSelection<STACKED>::TYPE;
        static_assert(mode::is<STACKED, nn::optimizers::mode::Freeze>);
        static_assert(!SELECTION::MATCH<nn::optimizers::UpdateTarget<FIRST>>);
        static_assert(SELECTION::MATCH<nn::optimizers::UpdateTarget<THIRD>>);
        static_assert(SELECTION::MATCH<nn::optimizers::UpdateTarget<UNGROUPED>>);
        using OUTER_UPDATE = nn::optimizers::mode::Update<STACKED, nn::group::All>;
        static_assert(nn::optimizers::UpdateSelection<OUTER_UPDATE>::TYPE::MATCH<nn::optimizers::UpdateTarget<FIRST>>);
        using OUTER_FREEZE = nn::optimizers::mode::Freeze<OUTER_UPDATE, LearnedMatrixWeights>;
        static_assert(nn::optimizers::UpdateSelection<OUTER_FREEZE>::TYPE::MATCH<nn::optimizers::UpdateTarget<FIRST>>);
        static_assert(!nn::optimizers::UpdateSelection<OUTER_FREEZE>::TYPE::MATCH<nn::optimizers::UpdateTarget<THIRD>>);
        forward(device, model, input, buffer, rng);
        backward(device, model, input, d_output, buffer);
        step(device, optimizer, model);
        copy(device, device, model, reference);
        step(device, optimizer, model, Mode<nn::optimizers::mode::Freeze<>>{});
        EXPECT_EQ(abs_diff(device, model, reference), 0);
        step(device, optimizer, model, Mode<nn::optimizers::mode::Freeze<mode::Default<>, nn::group::Only<Missing>>>{});
        EXPECT_GT(abs_diff(device, get_layer<0>(model).weights.parameters, get_layer<0>(reference).weights.parameters), 0);
        EXPECT_GT(abs_diff(device, get_layer<2>(model).weights.parameters, get_layer<2>(reference).weights.parameters), 0);
    }
}

namespace rl_tools::nn::group{
    template<> struct CodeName<tests::groups::Fixed>{
        static constexpr const char* VALUE = "rl_tools::tests::groups::Fixed";
        static constexpr const char* DEFINITION = "namespace rl_tools::tests::groups{ struct Fixed; }";
    };
    template<> struct CodeName<tests::groups::Learned>{
        static constexpr const char* VALUE = "rl_tools::tests::groups::Learned";
        static constexpr const char* DEFINITION = "namespace rl_tools::tests::groups{ struct Learned; }";
    };
}
namespace rl_tools::tests::groups{
    struct HEAD_CONFIG: nn_models::mlp::Configuration<TYPE_POLICY, TI, 2, 2, 4, nn::activation_functions::TANH, nn::activation_functions::IDENTITY>{
        using LOG_STD_GROUP = Learned;
    };
    using HEAD = nn_models::Group<Fixed, nn_models::mlp_unconditional_stddev::BindConfiguration<HEAD_CONFIG>>;
    using BRANCH_FIXED = nn_models::parallel::Branch<nn_models::Group<Fixed, nn_models::sequential::Module<DENSE>>, SHAPE>;
    using BRANCH_LEARNED = nn_models::parallel::Branch<nn_models::Group<Learned, nn_models::sequential::Module<DENSE>>, SHAPE>;
    using PARALLEL = nn_models::parallel::Build<CAPABILITY, HEAD, BRANCH_FIXED, BRANCH_LEARNED>;

    TEST_F(Fixture, ParallelBranchesAndExploration){
        PARALLEL actor, snapshot;
        PARALLEL::Buffer<> actor_buffer;
        malloc(device, actor);
        malloc(device, snapshot);
        malloc(device, actor_buffer);
        init_weights(device, actor, rng);
        reset_optimizer_state(device, optimizer, actor);
        zero_gradient(device, actor);
        auto inputs = nn_models::parallel::pack_inputs(input, input);
        forward(device, actor, inputs, actor_buffer, rng);
        backward(device, actor, inputs, d_output, actor_buffer);
        set_all(device, actor.head.log_std.gradient, T(1));
        step(device, optimizer, actor);
        copy(device, device, actor, snapshot);
        using FREEZE_HEAD = Mode<nn::optimizers::mode::Freeze<mode::Default<>, nn::group::Every<nn::group::Only<Fixed>, nn::group::Not<nn::group::Only<Learned>>>>>;
        step(device, optimizer, actor, FREEZE_HEAD{});
        EXPECT_EQ(abs_diff(device, get_layer<0>(get<0>(actor.pipelines)).weights, get_layer<0>(get<0>(snapshot.pipelines)).weights), 0);
        EXPECT_GT(abs_diff(device, get_layer<0>(get<1>(actor.pipelines)).weights.parameters, get_layer<0>(get<1>(snapshot.pipelines)).weights.parameters), 0);
        EXPECT_EQ(abs_diff(device, actor.head.input_layer.weights, snapshot.head.input_layer.weights), 0);
        EXPECT_EQ(abs_diff(device, actor.head.output_layer.weights, snapshot.head.output_layer.weights), 0);
        EXPECT_GT(abs_diff(device, actor.head.log_std.parameters, snapshot.head.log_std.parameters), 0);
        free(device, actor_buffer);
        free(device, snapshot);
        free(device, actor);
    }

    TEST_F(Fixture, ExportCode){
        PARALLEL actor;
        malloc(device, actor);
        init_weights(device, actor, rng);
        reset_optimizer_state(device, optimizer, actor);
        zero_gradient(device, actor);
        forward(device, model, input, buffer, rng);
        PARALLEL::Buffer<> actor_buffer;
        malloc(device, actor_buffer);
        auto inputs = nn_models::parallel::pack_inputs(input, input);
        forward(device, actor, inputs, actor_buffer, rng);
        std::ofstream file("groups_export.h");
        ASSERT_TRUE(file.is_open());
        file << save_code(device, model, "grouped_chain") << "\n";
        file << save_code(device, actor, "grouped_parallel") << "\n";
        ASSERT_TRUE(file.good());
        free(device, actor_buffer);
        free(device, actor);
    }

    TEST_F(Fixture, LambSelection){
        using LAMB = nn::optimizers::Lamb<nn::optimizers::lamb::Specification<TYPE_POLICY, TI, OPTIMIZER_PARAMETERS>>;
        LAMB lamb;
        malloc(device, lamb);
        init(device, lamb);
        reset_optimizer_state(device, lamb, model);
        forward(device, model, input, buffer, rng);
        backward(device, model, input, d_output, buffer);
        step(device, lamb, model);
        copy(device, device, model, reference);
        step(device, lamb, model, FREEZE_FIXED{});
        EXPECT_EQ(abs_diff(device, get_layer<0>(model).weights, get_layer<0>(reference).weights), 0);
        EXPECT_GT(abs_diff(device, get_layer<2>(model).weights.parameters, get_layer<2>(reference).weights.parameters), 0);
        free(device, lamb);
    }

    TEST_F(Fixture, SgdSelection){
        using SGD_CAPABILITY = nn::capability::Gradient<nn::parameters::SGD>;
        using SGD_MODEL = nn_models::sequential::Build<SGD_CAPABILITY, CHAIN, SHAPE>;
        using SGD = nn::optimizers::SGD<nn::optimizers::sgd::Specification<TYPE_POLICY, TI>>;
        SGD_MODEL network, snapshot;
        SGD_MODEL::Buffer<> network_buffer;
        SGD sgd;
        malloc(device, network);
        malloc(device, snapshot);
        malloc(device, network_buffer);
        malloc(device, sgd);
        init(device, sgd);
        init_weights(device, network, rng);
        reset_optimizer_state(device, sgd, network);
        zero_gradient(device, network);
        forward(device, network, input, network_buffer, rng);
        backward(device, network, input, d_output, network_buffer);
        step(device, sgd, network);
        copy(device, device, network, snapshot);
        step(device, sgd, network, FREEZE_FIXED{});
        EXPECT_EQ(abs_diff(device, get_layer<0>(network).weights, get_layer<0>(snapshot).weights), 0);
        EXPECT_GT(abs_diff(device, get_layer<2>(network).weights.parameters, get_layer<2>(snapshot).weights.parameters), 0);
        free(device, sgd);
        free(device, network_buffer);
        free(device, snapshot);
        free(device, network);
    }
}
