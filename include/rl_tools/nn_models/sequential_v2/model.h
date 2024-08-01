#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_SEQUENTIAL_V2_MODEL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_SEQUENTIAL_V2_MODEL_H

#include "../../utils/generic/typing.h"
#include "../../nn/nn.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn_models::sequential_v2{
    struct OutputModule{
        struct CONTENT{
            using INPUT_SHAPE = tensor::Shape<decltype(0), 0>;
            static constexpr auto BATCH_SIZE = 0;
        };
        template <typename>
        using CHANGE_CAPABILITY = OutputModule;
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
        if constexpr (utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>){
            return SPEC::CONTENT::OUTPUT_DIM;
        } else {
            return find_output_dim<typename SPEC::NEXT_MODULE>();
        }
    }
    template <typename SPEC>
    constexpr auto find_output_shape() {
        if constexpr (utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>){
            return typename SPEC::CONTENT::OUTPUT_SHAPE{};
        } else {
            return find_output_dim<typename SPEC::NEXT_MODULE>();
        }
    }
    template <typename TI, typename SPEC>
    constexpr auto find_max_hiddend_dim(TI current_max = 0){
        constexpr TI CONTENT_OUTPUT_DIM = product(typename SPEC::CONTENT::OUTPUT_SHAPE{});
        current_max = current_max > CONTENT_OUTPUT_DIM ? current_max : CONTENT_OUTPUT_DIM;
        if constexpr (utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>){
            return 0;
        } else {
            TI max_downstream = find_max_hiddend_dim<TI, typename SPEC::NEXT_MODULE>();
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

//    template <typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
//    constexpr bool check_input_output_f(){
//        static_assert(INPUT_SPEC::COLS == SPEC::CONTENT::INPUT_DIM);
//        static_assert(OUTPUT_SPEC::ROWS == INPUT_SPEC::ROWS);
//        static_assert(OUTPUT_SPEC::COLS == find_output_dim<SPEC>());
//        static_assert(OUTPUT_SPEC::ROWS == INPUT_SPEC::ROWS);
//        return true;
//    }
//    template <typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
//    constexpr bool check_input_output = check_input_output_f<SPEC, INPUT_SPEC, OUTPUT_SPEC>();


    template <typename BUFFER_SPEC, typename MODULE_SPEC>
    constexpr bool buffer_compatible = BUFFER_SPEC::SPEC::MAX_HIDDEN_DIM >= MODULE_SPEC::MAX_HIDDEN_DIM;

    template <typename T_CONTENT, typename T_NEXT_MODULE = OutputModule>
    struct Specification{
        using CONTENT = T_CONTENT;
        using NEXT_MODULE = T_NEXT_MODULE;
        using T = typename CONTENT::T;
        using TI = typename CONTENT::TI;
        using CONTAINER_TYPE_TAG = typename CONTENT::CONTAINER_TYPE_TAG;
        using INPUT_SHAPE = typename CONTENT::INPUT_SHAPE;
        using OUTPUT_SHAPE = decltype(find_output_shape<Specification<T_CONTENT, T_NEXT_MODULE>>());
        static constexpr TI MAX_HIDDEN_DIM = find_max_hiddend_dim<typename CONTENT::TI, Specification<T_CONTENT, T_NEXT_MODULE>>();
        static constexpr bool NEXT_IS_OUTPUT = utils::typing::is_same_v<NEXT_MODULE, OutputModule>;
        static_assert(NEXT_IS_OUTPUT || tensor::same_dimensions_shape<typename CONTENT::OUTPUT_SHAPE, utils::typing::conditional_t<NEXT_IS_OUTPUT, typename CONTENT::OUTPUT_SHAPE, typename NEXT_MODULE::CONTENT::INPUT_SHAPE>>());
    };

    template <typename T_CAPABILITY, typename T_SPEC>
    struct CapabilitySpecification: T_SPEC{
        using CAPABILITY = T_CAPABILITY;
        using PARAMETER_TYPE = typename CAPABILITY::PARAMETER_TYPE;
    };
    template <typename T_SPEC>
    struct ModuleForward;
    template <typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE, typename T_CONTAINER_TYPE_TAG>
    struct ContentBufferSpecification {
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;
        using CONTENT = typename SPEC::CONTENT;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        using CONTENT_BUFFER = typename CONTENT::template Buffer<BATCH_SIZE>;
        static constexpr bool IS_FINAL = utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>;
        using NEXT_MODULE = utils::typing::conditional_t<!IS_FINAL, typename SPEC::NEXT_MODULE, ModuleForward<SPEC>>;
        using NEXT_SPEC = utils::typing::conditional_t<
                !IS_FINAL,
                ContentBufferSpecification<typename NEXT_MODULE::SPEC, BATCH_SIZE, CONTAINER_TYPE_TAG>,
                OutputModule
        >;
    };
    template <typename T_SPEC>
    struct ContentBuffer{
        using SPEC = T_SPEC;
        using CONTENT_BUFFER = typename SPEC::CONTENT_BUFFER;
        using NEXT_SPEC = typename SPEC::NEXT_SPEC;
        CONTENT_BUFFER buffer;
        using NEXT_CONTENT_BUFFER = utils::typing::conditional_t<utils::typing::is_same_v<NEXT_SPEC, OutputModule>,
                OutputModule,
                ContentBuffer<NEXT_SPEC>>;
        NEXT_CONTENT_BUFFER next_content_buffer;
    };

    template <typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE, typename T_CONTAINER_TYPE_TAG, typename T_MEMORY_LAYOUT>
    struct ModuleBufferSpecification {
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;
        using CONTENT = typename SPEC::CONTENT;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        using MEMORY_LAYOUT = T_MEMORY_LAYOUT;
        using CONTENT_BUFFER_SPEC = ContentBufferSpecification<SPEC, BATCH_SIZE, CONTAINER_TYPE_TAG>;
    };
    template <typename T_BUFFER_SPEC>
    struct ModuleBuffer{
        using BUFFER_SPEC = T_BUFFER_SPEC;
        using SPEC = typename BUFFER_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BUFFER_SPEC::BATCH_SIZE;
        using TICK_TOCK_CONTAINER_SHAPE = tensor::Shape<TI, SPEC::MAX_HIDDEN_DIM>;
        using TICK_TOCK_CONTAINER_SPEC = tensor::Specification<T, TI, TICK_TOCK_CONTAINER_SHAPE, tensor::RowMajorStride<TICK_TOCK_CONTAINER_SHAPE>>;
        using TICK_TOCK_CONTAINER_TYPE = Tensor<TICK_TOCK_CONTAINER_SPEC>;
//        using TICK_TOCK_CONTAINER_SPEC = matrix::Specification<T, TI, BATCH_SIZE, SPEC::MAX_HIDDEN_DIM, typename BUFFER_SPEC::MEMORY_LAYOUT>;
//        using TICK_TOCK_CONTAINER_TYPE = typename BUFFER_SPEC::CONTAINER_TYPE_TAG::template type<TICK_TOCK_CONTAINER_SPEC>;
        TICK_TOCK_CONTAINER_TYPE tick;
        TICK_TOCK_CONTAINER_TYPE tock;
        using CONTENT_BUFFER = ContentBuffer<typename BUFFER_SPEC::CONTENT_BUFFER_SPEC>;
        CONTENT_BUFFER content_buffer;
    };
    template <typename T_SPEC>
    struct ModuleForward{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using CONTENT = typename SPEC::CONTENT;
        using NEXT_MODULE = typename SPEC::NEXT_MODULE;
        CONTENT content;
        NEXT_MODULE next_module;

//        static constexpr auto INPUT_DIM = SPEC::INPUT_DIM;
//        static constexpr auto OUTPUT_DIM = SPEC::OUTPUT_DIM;

        // We have one module Buffer for the whole module and possible ContentBuffers for the intermediate steps (that are unwrapped recursively in tandem with the module/content)
        template <typename SPEC::TI BATCH_SIZE, typename CONTAINER_TYPE_TAG=typename SPEC::CONTAINER_TYPE_TAG, typename MEMORY_LAYOUT = matrix::layouts::DEFAULT<typename SPEC::TI>>
        using Buffer = ModuleBuffer<ModuleBufferSpecification<SPEC, BATCH_SIZE, CONTAINER_TYPE_TAG, MEMORY_LAYOUT>>;
    };

    template <typename T_SPEC>
    struct ModuleBackward: public ModuleForward<T_SPEC>{};
    template <typename T_SPEC>
    struct ModuleGradient: public ModuleBackward<T_SPEC>{
        using TI = typename T_SPEC::TI;
        static constexpr TI BATCH_SIZE = T_SPEC::CAPABILITY::BATCH_SIZE;
    };

    template <typename CAPABILITY, template <typename> typename CONTENT, typename NEXT_MODULE>
    using _Module =
        utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Forward,
                ModuleForward<CapabilitySpecification<CAPABILITY, Specification<CONTENT<CAPABILITY>, NEXT_MODULE>>>,
        utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Backward,
                ModuleBackward<CapabilitySpecification<CAPABILITY, Specification<CONTENT<CAPABILITY>, NEXT_MODULE>>>,
        utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Gradient,
                ModuleGradient<CapabilitySpecification<CAPABILITY, Specification<CONTENT<CAPABILITY>, NEXT_MODULE>>>, void>>>;

    template <typename T_CAPABILITY, template <typename> typename T_CONTENT, typename T_NEXT_MODULE = OutputModule>
    struct Module: _Module<T_CAPABILITY, T_CONTENT, T_NEXT_MODULE>{
        template <typename TT_CAPABILITY>
        using CHANGE_CAPABILITY = Module<TT_CAPABILITY, T_CONTENT, typename T_NEXT_MODULE::template CHANGE_CAPABILITY<TT_CAPABILITY>>;
    };

    template <typename T_CAPABILITY>
    struct Interface{
        template <template <typename> typename T_CONTENT, typename T_NEXT_MODULE = OutputModule>
        using Module = sequential_v2::Module<T_CAPABILITY, T_CONTENT, T_NEXT_MODULE>;
    };

    template <typename CAPABILITY>
    using OutputModuleTemplate = OutputModule;
    template <template <typename> typename CONTENT, template <typename> typename NEXT_MODULE = OutputModuleTemplate>
    struct Bind{
        template <typename CAPABILITY>
        using Module = sequential_v2::Module<CAPABILITY, CONTENT, NEXT_MODULE<CAPABILITY>>;
    };

}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
