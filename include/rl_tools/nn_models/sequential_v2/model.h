#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_SEQUENTIAL_V2_MODEL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_SEQUENTIAL_V2_MODEL_H

#include "../../utils/generic/typing.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn_models::sequential_v2{
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
    constexpr auto find_output_shape(){
        if constexpr(utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>){
            return SPEC::CONTENT::OUTPUT_SHAPE;
        } else {
            return find_output_shape<typename SPEC::NEXT_MODULE>();
        }
    }
    template <typename TI, typename SPEC>
    constexpr auto find_max_hiddend_size(TI current_max = 0){
        constexpr TI OUTPUT_SIZE = get<0>(tensor::Product<typename SPEC::CONTENT::OUTPUT_SHAPE>{});
        current_max = current_max > OUTPUT_SIZE ? current_max : OUTPUT_SIZE;
        if constexpr(utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>){
            return 0;
        } else {
            auto max_downstream = find_max_hiddend_size<TI, typename SPEC::NEXT_MODULE>();
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

//    template <typename MODULE>
//    constexpr bool check_batch_size_consistency = check_batch_size_consistency_f<MODULE>();

//    template <typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
//    constexpr bool check_input_output_f(){
//        static_assert(INPUT_SPEC::COLS == SPEC::CONTENT::INPUT_DIM);
//        static_assert(OUTPUT_SPEC::ROWS == INPUT_SPEC::ROWS);
//        static_assert(OUTPUT_SPEC::COLS == find_output_size<SPEC>());
//        static_assert(OUTPUT_SPEC::ROWS == INPUT_SPEC::ROWS);
//        return true;
//    }
//    template <typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
//    constexpr bool check_input_output = check_input_output_f<SPEC, INPUT_SPEC, OUTPUT_SPEC>();


//    template <typename BUFFER_SPEC, typename MODULE_SPEC>
//    constexpr bool buffer_compatible = BUFFER_SPEC::SPEC::MAX_HIDDEN_DIM >= MODULE_SPEC::MAX_HIDDEN_DIM;

    template <typename T_CONTENT, typename T_NEXT_MODULE = OutputModule>
    struct Specification{
        using CONTENT = T_CONTENT;
        using NEXT_MODULE = T_NEXT_MODULE;
        using T = typename CONTENT::T;
        using TI = typename CONTENT::TI;
        using CONTAINER_TYPE_TAG = typename CONTENT::CONTAINER_TYPE_TAG;
        static constexpr TI INPUT_SHAPE = CONTENT::INPUT_SHAPE;
        static constexpr TI OUTPUT_SHAPE = find_output_shape<Specification<T_CONTENT, T_NEXT_MODULE>>();
        static constexpr TI MAX_HIDDEN_SIZE = find_max_hiddend_size<typename CONTENT::TI, Specification<T_CONTENT, T_NEXT_MODULE>>();
//        static_assert(utils::typing::is_same_v<NEXT_MODULE, OutputModule> || CONTENT::OUTPUT_DIM == NEXT_MODULE::CONTENT::INPUT_DIM);
    };

    template <typename T_SPEC, bool T_STATIC>
    struct ModuleBufferSpecification {
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;
        static constexpr TI SIZE = SPEC::MAX_HIDDEN_SIZE;
        static constexpr bool STATIC = T_STATIC;
    };
    template <typename T_BUFFER_SPEC>
    struct ModuleBuffer{
        using BUFFER_SPEC = T_BUFFER_SPEC;
        using SPEC = typename BUFFER_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI SIZE = BUFFER_SPEC::SIZE;
        using CONTAINER_SHAPE = tensor::Shape<TI, SIZE>;
        using CONTAINER_SPEC = tensor::Specification<T, TI, CONTAINER_SHAPE, tensor::RowMajorStride<CONTAINER_SHAPE>>;
        using CONTAINER_TYPE = Tensor<CONTAINER_SPEC>;
        CONTAINER_TYPE tick;
        CONTAINER_TYPE tock;
    };
    template <typename T_SPEC>
    struct Module{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using CONTENT = typename SPEC::CONTENT;
        using BUFFER_EVALUATION = typename CONTENT::BufferEvaluation;
        using NEXT_MODULE = typename SPEC::NEXT_MODULE;
        CONTENT content;
        BUFFER_EVALUATION buffer_evaluation;

        NEXT_MODULE next_module;

        static constexpr auto INPUT_SHAPE = SPEC::INPUT_SHAPE;
        static constexpr auto OUTPUT_SHAPE = SPEC::OUTPUT_SHAPE;

        template <bool STATIC=false>
        using Buffer = ModuleBuffer<ModuleBufferSpecification<SPEC, STATIC>>;
    };

    namespace interface{
        template <typename T_CONTENT, typename T_NEXT_MODULE = OutputModule>
        struct Module: rl_tools::nn_models::sequential_v2::Module<Specification<T_CONTENT, T_NEXT_MODULE>>{};
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
