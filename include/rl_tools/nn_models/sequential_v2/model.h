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
            return find_output_shape<typename SPEC::NEXT_MODULE>();
        }
    }
    template <typename TI, typename SPEC>
    constexpr auto find_max_hiddend_dim(TI current_max = 0){
        constexpr TI CONTENT_OUTPUT_DIM = product(typename SPEC::CONTENT::OUTPUT_SHAPE{});
        current_max = current_max > CONTENT_OUTPUT_DIM ? current_max : CONTENT_OUTPUT_DIM;
        if constexpr (utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>){
            return current_max;
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

    template <typename BUFFER_SPEC, typename MODULE_SPEC>
    constexpr bool buffer_compatible = BUFFER_SPEC::SPEC::MAX_HIDDEN_DIM >= MODULE_SPEC::MAX_HIDDEN_DIM;

    template <typename T_ORIGINAL_ROOT, typename T_CONTENT, typename T_NEXT_MODULE = OutputModule>
    struct Specification{
        using ORIGINAL_ROOT = T_ORIGINAL_ROOT; // saving this such that we can reuse ::Build to change capabilities and/or input shapes
        using CONTENT = T_CONTENT;
        using NEXT_MODULE = T_NEXT_MODULE;
        using T = typename CONTENT::T;
        using TI = typename CONTENT::TI;
        using INPUT_SHAPE = typename CONTENT::INPUT_SHAPE;
        using OUTPUT_SHAPE = decltype(find_output_shape<Specification<T_ORIGINAL_ROOT, T_CONTENT, T_NEXT_MODULE>>());
        static constexpr TI MAX_HIDDEN_DIM = find_max_hiddend_dim<typename CONTENT::TI, Specification<T_ORIGINAL_ROOT, T_CONTENT, T_NEXT_MODULE>>();
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
    template <typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE, bool T_DYNAMIC_ALLOCATION>
    struct ContentBufferSpecification {
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;
        using CONTENT = typename SPEC::CONTENT;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
        using CONTENT_BUFFER = typename CONTENT::template Buffer<BATCH_SIZE, DYNAMIC_ALLOCATION>;
        static constexpr bool IS_FINAL = utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>;
        using NEXT_MODULE = utils::typing::conditional_t<!IS_FINAL, typename SPEC::NEXT_MODULE, ModuleForward<SPEC>>;
        using NEXT_SPEC = utils::typing::conditional_t<
                !IS_FINAL,
                ContentBufferSpecification<typename NEXT_MODULE::SPEC, BATCH_SIZE, DYNAMIC_ALLOCATION>,
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

    template <typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE, bool T_DYNAMIC_ALLOCATION = true>
    struct ModuleBufferSpecification {
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;
        using CONTENT = typename SPEC::CONTENT;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
        using CONTENT_BUFFER_SPEC = ContentBufferSpecification<SPEC, BATCH_SIZE, DYNAMIC_ALLOCATION>;
    };
    template <typename T_BUFFER_SPEC>
    struct ModuleBuffer{
        using BUFFER_SPEC = T_BUFFER_SPEC;
        using SPEC = typename BUFFER_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BUFFER_SPEC::BATCH_SIZE;
        static_assert(SPEC::MAX_HIDDEN_DIM > 0);
        static_assert(BATCH_SIZE > 0);
        using TICK_TOCK_CONTAINER_SHAPE = tensor::Shape<TI, SPEC::MAX_HIDDEN_DIM * BATCH_SIZE>; // TODO: check if this is overkill
        using TICK_TOCK_CONTAINER_SPEC = tensor::Specification<T, TI, TICK_TOCK_CONTAINER_SHAPE, BUFFER_SPEC::DYNAMIC_ALLOCATION, tensor::RowMajorStride<TICK_TOCK_CONTAINER_SHAPE>>;
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
        using INPUT_SHAPE = typename SPEC::INPUT_SHAPE;
        using OUTPUT_SHAPE = typename SPEC::OUTPUT_SHAPE;

        // We have one module Buffer for the whole module and possible ContentBuffers for the intermediate steps (that are unwrapped recursively in tandem with the module/content)
        template <typename SPEC::TI BATCH_SIZE, bool DYNAMIC_ALLOCATION=true>
        using Buffer = ModuleBuffer<ModuleBufferSpecification<SPEC, BATCH_SIZE, DYNAMIC_ALLOCATION>>;
    };

    template <typename T_SPEC>
    struct ModuleBackward: public ModuleForward<T_SPEC>{};
    template <typename T_SPEC>
    struct ModuleGradient: public ModuleBackward<T_SPEC>{
        using TI = typename T_SPEC::TI;
        static constexpr TI BATCH_SIZE = T_SPEC::CAPABILITY::BATCH_SIZE;
    };

    template <typename T_CONTENT, typename T_NEXT_MODULE = OutputModule>
    struct Module{
        using CONTENT = T_CONTENT;
        using NEXT_CARRIER_MODULE = T_NEXT_MODULE;
    };

    template <typename CAPABILITY, typename T_MODULE, typename INPUT_SHAPE>
    struct _Chain{
        using T_CONTENT = typename T_MODULE::CONTENT;
        using CONTENT = typename T_CONTENT::template Layer<CAPABILITY, INPUT_SHAPE>;
        using OUTPUT_SHAPE = typename CONTENT::SPEC::OUTPUT_SHAPE;
        using NEXT_MODULE = utils::typing::conditional_t<utils::typing::is_same_v<typename T_MODULE::NEXT_CARRIER_MODULE, OutputModule>, OutputModule, typename _Chain<CAPABILITY, typename T_MODULE::NEXT_CARRIER_MODULE, OUTPUT_SHAPE>::MODULE>;
        using MODULE_FORWARD = ModuleForward<CapabilitySpecification<CAPABILITY, Specification<T_MODULE, CONTENT, NEXT_MODULE>>>;
        using MODULE_BACKWARD = ModuleBackward<CapabilitySpecification<CAPABILITY, Specification<T_MODULE, CONTENT, NEXT_MODULE>>>;
        using MODULE_GRADIENT = ModuleGradient<CapabilitySpecification<CAPABILITY, Specification<T_MODULE, CONTENT, NEXT_MODULE>>>;
        using MODULE = utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Forward, MODULE_FORWARD,
        utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Backward, MODULE_BACKWARD,
        utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Gradient, MODULE_GRADIENT, void>>>;
    };
    template <typename CAPABILITY, typename INPUT_SHAPE>
    struct _Chain<CAPABILITY, OutputModule, INPUT_SHAPE>{
        using MODULE = OutputModule;
    };


    // The user specifies a sequence of Modules using the Module type and then builds them for a particular capability and input shape
    // When e.g. saving checkpoints the user can rebuild the model with a different capability (e.g. without gradient and optimizers state) and then save the checkpoint
    template <typename CAPABILITY, typename MODULE, typename INPUT_SHAPE>
    using Build = typename _Chain<CAPABILITY, MODULE, INPUT_SHAPE>::MODULE;

//    template <typename T_MODULE, typename T_INPUT_SHAPE>
//    struct Chain{
//
//    };

    template <typename CAPABILITY, typename T_CONTENT, typename INPUT_SHAPE, typename NEXT_MODULE>
    struct _Module{

    };

//    template <typename T_CAPABILITY>
//    struct Interface{
//        template <typename T_CONTENT, typename T_NEXT_MODULE = OutputModule, typename T_INPUT_SHAPE = void>
//        using Module = sequential_v2::Module<T_CAPABILITY, T_CONTENT, T_INPUT_SHAPE, T_NEXT_MODULE>;
//    };
//
//    template <typename CAPABILITY>
//    using OutputModuleTemplate = OutputModule;
//    template <typename CONTENT, template <typename> typename NEXT_MODULE = OutputModuleTemplate>
//    struct Bind{
//        template <typename CAPABILITY>
//        using Module = sequential_v2::Module<CAPABILITY, CONTENT, NEXT_MODULE<CAPABILITY>>;
//    };

}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
